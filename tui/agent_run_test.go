package tui

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/stack-bound/stackllm/agent"
	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/provider"
	"github.com/stack-bound/stackllm/tools"
)

// scriptedProvider feeds one canned event stream per Complete call,
// mirroring the mockProvider pattern used by the agent package tests.
type scriptedProvider struct {
	mu        sync.Mutex
	responses [][]provider.Event
	callIndex int
}

func (p *scriptedProvider) Complete(_ context.Context, _ provider.Request) (<-chan provider.Event, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	ch := make(chan provider.Event, 64)
	if p.callIndex >= len(p.responses) {
		close(ch)
		return ch, fmt.Errorf("scripted: no more responses")
	}
	events := p.responses[p.callIndex]
	p.callIndex++
	go func() {
		defer close(ch)
		for _, ev := range events {
			ch <- ev
		}
	}()
	return ch, nil
}

func (p *scriptedProvider) Models(_ context.Context) ([]provider.ModelMeta, error) {
	return nil, nil
}

func textEvents(text string) []provider.Event {
	blk := conversation.Block{Type: conversation.BlockText, Text: text}
	return []provider.Event{
		{Type: provider.EventTypeBlockStart, BlockType: conversation.BlockText},
		{Type: provider.EventTypeBlockDelta, BlockType: conversation.BlockText, Content: text},
		{Type: provider.EventTypeBlockEnd, BlockType: conversation.BlockText, Block: &blk},
	}
}

func thinkingEvents(text string) []provider.Event {
	blk := conversation.Block{Type: conversation.BlockThinking, Text: text}
	return []provider.Event{
		{Type: provider.EventTypeBlockStart, BlockType: conversation.BlockThinking},
		{Type: provider.EventTypeBlockDelta, BlockType: conversation.BlockThinking, Content: text},
		{Type: provider.EventTypeBlockEnd, BlockType: conversation.BlockThinking, Block: &blk},
	}
}

func toolUseEvents(id, name, args string) []provider.Event {
	blk := conversation.Block{
		Type:         conversation.BlockToolUse,
		ToolCallID:   id,
		ToolName:     name,
		ToolArgsJSON: args,
	}
	return []provider.Event{
		{Type: provider.EventTypeBlockStart, BlockType: conversation.BlockToolUse},
		{Type: provider.EventTypeBlockEnd, BlockType: conversation.BlockToolUse, Block: &blk},
		{Type: provider.EventTypeToolCall, Call: &conversation.ToolCall{ID: id, Name: name, Arguments: args}},
	}
}

func concatEvents(groups ...[]provider.Event) []provider.Event {
	var out []provider.Event
	for _, g := range groups {
		out = append(out, g...)
	}
	return out
}

// runModelWithProvider builds a TUI model around an agent that talks to
// the scripted provider, with a user message already in the session.
func runModelWithProvider(t *testing.T, p provider.Provider, store *fullFakeStore, agentOpts ...agent.Option) *Model {
	t.Helper()
	a := agent.New(p, agentOpts...)
	m := New(a, store)
	m.width = 100
	m.height = 40
	m.session.AppendMessage(conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "question"}},
	})
	return m
}

func TestRunAgent_StreamsThinkingAndText(t *testing.T) {
	t.Parallel()

	p := &scriptedProvider{responses: [][]provider.Event{
		concatEvents(
			thinkingEvents("let me reason"),
			textEvents("the final answer"),
			[]provider.Event{
				{Type: provider.EventTypeUsage, Usage: &provider.TokenUsage{PromptTokens: 42, CompletionTokens: 7, TotalTokens: 49}},
				{Type: provider.EventTypeDone},
			},
		),
	}}
	store := newFullFakeStore()
	m := runModelWithProvider(t, p, store)
	m.state = stateRunning

	msg := m.runAgent()()
	if _, ok := msg.(agentDoneMsg); !ok {
		t.Fatalf("expected agentDoneMsg, got %T", msg)
	}

	// The session must carry the full evolved conversation.
	if len(m.session.Messages) != 2 {
		t.Fatalf("session messages = %d, want 2 (user + assistant)", len(m.session.Messages))
	}
	asst := m.session.Messages[1]
	if asst.Role != conversation.RoleAssistant {
		t.Errorf("messages[1] role = %q, want assistant", asst.Role)
	}
	if got := asst.TextContent(); got != "the final answer" {
		t.Errorf("assistant text = %q, want %q", got, "the final answer")
	}
	if got := asst.ThinkingText(); got != "let me reason" {
		t.Errorf("assistant thinking = %q, want %q", got, "let me reason")
	}

	// Usage lands on the session for the status line.
	if m.session.LastUsage == nil || m.session.LastUsage.PromptTokens != 42 {
		t.Errorf("LastUsage = %+v, want PromptTokens 42", m.session.LastUsage)
	}

	// Streaming output: thinking prefix and the text, thinking first.
	out := m.output.String()
	thinkIdx := strings.Index(out, "let me reason")
	textIdx := strings.Index(out, "the final answer")
	if !strings.Contains(out, "thinking:") {
		t.Errorf("expected thinking: marker in output:\n%s", out)
	}
	if thinkIdx < 0 || textIdx < 0 || thinkIdx > textIdx {
		t.Errorf("expected thinking before text (think=%d text=%d):\n%s", thinkIdx, textIdx, out)
	}

	// Feeding the done msg back through Update persists the session.
	updated, _ := m.Update(msg)
	m = updated.(*Model)
	if m.state != stateIdle {
		t.Errorf("expected stateIdle after done, got %v", m.state)
	}
	saved, err := store.Load(context.Background(), m.session.ID)
	if err != nil {
		t.Fatalf("session not persisted: %v", err)
	}
	if len(saved.Messages) != 2 {
		t.Errorf("persisted messages = %d, want 2", len(saved.Messages))
	}
}

func TestRunAgent_ToolCallRoundTrip(t *testing.T) {
	t.Parallel()

	p := &scriptedProvider{responses: [][]provider.Event{
		concatEvents(
			toolUseEvents("call_1", "echo", `{"text":"tool says hi"}`),
			[]provider.Event{{Type: provider.EventTypeDone}},
		),
		concatEvents(
			textEvents("done with the tool"),
			[]provider.Event{{Type: provider.EventTypeDone}},
		),
	}}

	type echoArgs struct {
		Text string `json:"text"`
	}
	reg := tools.NewRegistry()
	if err := reg.Register("echo", "echo text back", func(_ context.Context, args echoArgs) (string, error) {
		return args.Text, nil
	}); err != nil {
		t.Fatalf("register tool: %v", err)
	}

	m := runModelWithProvider(t, p, newFullFakeStore(), agent.WithTools(reg))
	m.state = stateRunning

	msg := m.runAgent()()
	if _, ok := msg.(agentDoneMsg); !ok {
		t.Fatalf("expected agentDoneMsg, got %T", msg)
	}

	// user + assistant(tool_use) + tool(result) + assistant(text)
	if len(m.session.Messages) != 4 {
		t.Fatalf("session messages = %d, want 4: %+v", len(m.session.Messages), m.session.Messages)
	}
	toolMsg := m.session.Messages[2]
	if toolMsg.Role != conversation.RoleTool {
		t.Errorf("messages[2] role = %q, want tool", toolMsg.Role)
	}
	results := toolMsg.ToolResults()
	if len(results) != 1 || results[0].Text != "tool says hi" || results[0].ToolCallID != "call_1" {
		t.Errorf("tool result = %+v, want text %q for call_1", results, "tool says hi")
	}
	if got := m.session.Messages[3].TextContent(); got != "done with the tool" {
		t.Errorf("final assistant text = %q", got)
	}

	out := m.output.String()
	if !strings.Contains(out, "⚡ echo") {
		t.Errorf("expected tool call marker in output:\n%s", out)
	}
	if !strings.Contains(out, "→ tool says hi") {
		t.Errorf("expected tool result preview in output:\n%s", out)
	}
}

func TestRunAgent_ProviderErrorSurfaces(t *testing.T) {
	t.Parallel()

	// No scripted responses: the first Complete call fails.
	p := &scriptedProvider{}
	m := runModelWithProvider(t, p, newFullFakeStore())
	m.state = stateRunning

	msg := m.runAgent()()
	if _, ok := msg.(agentDoneMsg); !ok {
		t.Fatalf("expected agentDoneMsg, got %T", msg)
	}
	if m.err == nil || !strings.Contains(m.err.Error(), "no more responses") {
		t.Errorf("expected provider error recorded, got %v", m.err)
	}
	if !strings.Contains(m.output.String(), "Error:") {
		t.Errorf("expected inline error line, got:\n%s", m.output.String())
	}
}

// TestSendMessageStartsRun wires the full Enter-to-run flow: typing a
// message and pressing Enter must append the user message, flip to
// stateRunning, and hand back a cmd that drives the agent to completion.
func TestSendMessageStartsRun(t *testing.T) {
	t.Parallel()

	p := &scriptedProvider{responses: [][]provider.Event{
		concatEvents(textEvents("hello back"), []provider.Event{{Type: provider.EventTypeDone}}),
	}}
	a := agent.New(p)
	store := newFullFakeStore()
	m := New(a, store)

	updatedModel, _ := m.Update(tea.WindowSizeMsg{Width: 100, Height: 40})
	m = updatedModel.(*Model)
	m = typeString(t, m, "hi there")

	updatedModel, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updatedModel.(*Model)
	if m.state != stateRunning {
		t.Fatalf("expected stateRunning after Enter, got %v", m.state)
	}
	if len(m.session.Messages) != 1 || m.session.Messages[0].TextContent() != "hi there" {
		t.Fatalf("expected the typed message in the session, got %+v", m.session.Messages)
	}
	if cmd == nil {
		t.Fatal("expected run cmd from Enter")
	}
	// The run cmd (batched with UI cmds) eventually yields agentDoneMsg.
	var done bool
	for _, msg := range collectMsgs(cmd) {
		if _, ok := msg.(agentDoneMsg); ok {
			done = true
		}
	}
	if !done {
		t.Fatal("expected agentDoneMsg from the run cmd")
	}
	if len(m.session.Messages) != 2 || m.session.Messages[1].TextContent() != "hello back" {
		t.Errorf("expected assistant reply appended, got %+v", m.session.Messages)
	}
}
