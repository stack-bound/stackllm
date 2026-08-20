package agent

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/provider"
	"github.com/stack-bound/stackllm/tools"
)

// mockProvider returns preconfigured events for testing.
type mockProvider struct {
	responses [][]provider.Event // one response per call
	callIndex int
}

func (m *mockProvider) Complete(_ context.Context, _ provider.Request) (<-chan provider.Event, error) {
	ch := make(chan provider.Event, 64)
	if m.callIndex >= len(m.responses) {
		close(ch)
		return ch, fmt.Errorf("mock: no more responses")
	}
	events := m.responses[m.callIndex]
	m.callIndex++
	go func() {
		defer close(ch)
		for _, ev := range events {
			ch <- ev
		}
	}()
	return ch, nil
}

func (m *mockProvider) Models(_ context.Context) ([]provider.ModelMeta, error) {
	return nil, nil
}

// textBlockEvents returns the canonical block event triplet for a
// single text block with the given content.
func textBlockEvents(text string) []provider.Event {
	blk := conversation.Block{Type: conversation.BlockText, Text: text}
	return []provider.Event{
		{Type: provider.EventTypeBlockStart, BlockType: conversation.BlockText},
		{Type: provider.EventTypeBlockDelta, BlockType: conversation.BlockText, Content: text},
		{Type: provider.EventTypeBlockEnd, BlockType: conversation.BlockText, Block: &blk},
	}
}

// toolUseBlockEvents returns the canonical block event triplet plus
// the convenience EventTypeToolCall for a closed tool_use block.
func toolUseBlockEvents(id, name, args string) []provider.Event {
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

func thinkingBlockEvents(text string) []provider.Event {
	blk := conversation.Block{Type: conversation.BlockThinking, Text: text}
	return []provider.Event{
		{Type: provider.EventTypeBlockStart, BlockType: conversation.BlockThinking},
		{Type: provider.EventTypeBlockDelta, BlockType: conversation.BlockThinking, Content: text},
		{Type: provider.EventTypeBlockEnd, BlockType: conversation.BlockThinking, Block: &blk},
	}
}

func userMessage(text string) conversation.Message {
	return conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: text}},
	}
}

func concat(groups ...[]provider.Event) []provider.Event {
	var out []provider.Event
	for _, g := range groups {
		out = append(out, g...)
	}
	return out
}

func TestStep_TextOnly(t *testing.T) {
	t.Parallel()

	events := concat(
		textBlockEvents("Hello world"),
		[]provider.Event{{Type: provider.EventTypeDone}},
	)
	p := &mockProvider{responses: [][]provider.Event{events}}

	a := New(p)
	msgs := []conversation.Message{userMessage("Hi")}

	msgs, result, err := a.Step(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Step error: %v", err)
	}
	if !result.Done {
		t.Error("expected Done=true for text-only response")
	}
	if got := result.AssistantMessage.TextContent(); got != "Hello world" {
		t.Errorf("text content = %q, want %q", got, "Hello world")
	}
	if len(result.ToolResults) != 0 {
		t.Errorf("expected 0 tool results, got %d", len(result.ToolResults))
	}
	// msgs should have user + assistant = 2
	if len(msgs) != 2 {
		t.Errorf("msgs len = %d, want 2", len(msgs))
	}
}

func TestStep_WithToolCall(t *testing.T) {
	t.Parallel()

	events := concat(
		toolUseBlockEvents("call_1", "echo", `{"text":"hello"}`),
		[]provider.Event{{Type: provider.EventTypeDone}},
	)
	p := &mockProvider{responses: [][]provider.Event{events}}

	type EchoArgs struct {
		Text string `json:"text"`
	}

	reg := tools.NewRegistry()
	reg.Register("echo", "echo text", func(ctx context.Context, args EchoArgs) (string, error) {
		return args.Text, nil
	})

	a := New(p, WithTools(reg))
	msgs := []conversation.Message{userMessage("echo hello")}

	msgs, result, err := a.Step(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Step error: %v", err)
	}
	if result.Done {
		t.Error("expected Done=false when tool calls present")
	}
	// A single tool-role message should carry one BlockToolResult.
	if len(result.ToolResults) != 1 {
		t.Fatalf("expected 1 tool message, got %d", len(result.ToolResults))
	}
	toolMsg := result.ToolResults[0]
	results := toolMsg.ToolResults()
	if len(results) != 1 {
		t.Fatalf("tool_result blocks = %d, want 1", len(results))
	}
	if results[0].Text != "hello" {
		t.Errorf("tool result text = %q, want %q", results[0].Text, "hello")
	}
	if results[0].ToolCallID != "call_1" {
		t.Errorf("tool_call_id = %q, want %q", results[0].ToolCallID, "call_1")
	}
	// msgs: user + assistant + tool = 3
	if len(msgs) != 3 {
		t.Errorf("msgs len = %d, want 3", len(msgs))
	}
}

func TestStep_MultipleToolCallsSingleToolMessage(t *testing.T) {
	t.Parallel()

	events := concat(
		toolUseBlockEvents("c1", "echo", `{"text":"a"}`),
		toolUseBlockEvents("c2", "echo", `{"text":"b"}`),
		[]provider.Event{{Type: provider.EventTypeDone}},
	)
	p := &mockProvider{responses: [][]provider.Event{events}}

	type EchoArgs struct {
		Text string `json:"text"`
	}
	reg := tools.NewRegistry()
	reg.Register("echo", "echo", func(ctx context.Context, args EchoArgs) (string, error) {
		return args.Text, nil
	})

	a := New(p, WithTools(reg))
	msgs, result, err := a.Step(context.Background(), []conversation.Message{userMessage("do it")})
	if err != nil {
		t.Fatalf("Step error: %v", err)
	}

	// One tool-role message with two tool_result blocks.
	if len(result.ToolResults) != 1 {
		t.Fatalf("tool messages = %d, want 1", len(result.ToolResults))
	}
	results := result.ToolResults[0].ToolResults()
	if len(results) != 2 {
		t.Fatalf("tool_result blocks = %d, want 2", len(results))
	}
	if results[0].Text != "a" || results[1].Text != "b" {
		t.Errorf("results = %q, %q", results[0].Text, results[1].Text)
	}
	if results[0].ToolCallID != "c1" || results[1].ToolCallID != "c2" {
		t.Errorf("call ids = %q, %q", results[0].ToolCallID, results[1].ToolCallID)
	}
	if len(msgs) != 3 {
		t.Errorf("msgs len = %d, want 3 (user, assistant, tool)", len(msgs))
	}
}

// TestStep_InterleavedBlocks is the Phase 1 end-to-end gate for the
// agent layer: a mock provider emits
//
//	thinking → text → tool_use → thinking → tool_use → thinking → text
//
// and the agent must capture all seven blocks in that exact order on
// the assistant message.
func TestStep_InterleavedBlocks(t *testing.T) {
	t.Parallel()

	events := concat(
		thinkingBlockEvents("planning"),
		textBlockEvents("Let me check."),
		toolUseBlockEvents("c1", "echo", `{"text":"a"}`),
		thinkingBlockEvents("found it"),
		toolUseBlockEvents("c2", "echo", `{"text":"b"}`),
		thinkingBlockEvents("analyzing"),
		textBlockEvents("The bug is X."),
		[]provider.Event{{Type: provider.EventTypeDone}},
	)
	p := &mockProvider{responses: [][]provider.Event{events}}

	type EchoArgs struct {
		Text string `json:"text"`
	}
	reg := tools.NewRegistry()
	reg.Register("echo", "echo", func(ctx context.Context, args EchoArgs) (string, error) {
		return args.Text, nil
	})

	var blockOrder []conversation.BlockType
	var thinkingStarts, textStarts, toolStarts int
	hooks := Hooks{
		OnBlockStart: func(_ context.Context, bt conversation.BlockType) {
			switch bt {
			case conversation.BlockThinking:
				thinkingStarts++
			case conversation.BlockText:
				textStarts++
			case conversation.BlockToolUse:
				toolStarts++
			}
		},
		OnBlockEnd: func(_ context.Context, blk conversation.Block) {
			blockOrder = append(blockOrder, blk.Type)
		},
	}

	a := New(p, WithTools(reg), WithHooks(hooks))
	_, result, err := a.Step(context.Background(), []conversation.Message{userMessage("go")})
	if err != nil {
		t.Fatalf("Step error: %v", err)
	}

	wantTypes := []conversation.BlockType{
		conversation.BlockThinking, conversation.BlockText, conversation.BlockToolUse,
		conversation.BlockThinking, conversation.BlockToolUse,
		conversation.BlockThinking, conversation.BlockText,
	}
	if len(result.AssistantMessage.Blocks) != len(wantTypes) {
		t.Fatalf("assistant blocks = %d, want %d", len(result.AssistantMessage.Blocks), len(wantTypes))
	}
	for i, want := range wantTypes {
		if result.AssistantMessage.Blocks[i].Type != want {
			t.Errorf("blocks[%d].Type = %q, want %q", i, result.AssistantMessage.Blocks[i].Type, want)
		}
	}
	if len(blockOrder) != len(wantTypes) {
		t.Fatalf("OnBlockEnd fired %d times, want %d", len(blockOrder), len(wantTypes))
	}
	for i, want := range wantTypes {
		if blockOrder[i] != want {
			t.Errorf("OnBlockEnd order[%d] = %q, want %q", i, blockOrder[i], want)
		}
	}
	if thinkingStarts != 3 || textStarts != 2 || toolStarts != 2 {
		t.Errorf("OnBlockStart counts: thinking=%d text=%d tool=%d (want 3/2/2)",
			thinkingStarts, textStarts, toolStarts)
	}
	if result.Done {
		t.Error("Done should be false (tool uses present)")
	}
	if got := result.AssistantMessage.TextContent(); got != "Let me check.The bug is X." {
		t.Errorf("TextContent() = %q", got)
	}
}

func TestStep_ToolError(t *testing.T) {
	t.Parallel()

	events := concat(
		toolUseBlockEvents("call_1", "fail", `{}`),
		[]provider.Event{{Type: provider.EventTypeDone}},
	)
	p := &mockProvider{responses: [][]provider.Event{events}}

	type FailArgs struct{}

	reg := tools.NewRegistry()
	reg.Register("fail", "always fails", func(ctx context.Context, args FailArgs) (string, error) {
		return "", fmt.Errorf("something broke")
	})

	a := New(p, WithTools(reg))
	msgs := []conversation.Message{userMessage("fail")}

	_, result, err := a.Step(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Step should not return error for tool errors, got: %v", err)
	}
	if len(result.ToolResults) != 1 {
		t.Fatalf("tool messages = %d, want 1", len(result.ToolResults))
	}
	results := result.ToolResults[0].ToolResults()
	if len(results) != 1 {
		t.Fatalf("tool_result blocks = %d, want 1", len(results))
	}
	if results[0].Text != "Error: something broke" {
		t.Errorf("tool result text = %q, want error message", results[0].Text)
	}
	if !results[0].ToolIsError {
		t.Error("ToolIsError should be true")
	}
}

func TestRun_ThreeSteps(t *testing.T) {
	t.Parallel()

	type EchoArgs struct {
		Text string `json:"text"`
	}

	reg := tools.NewRegistry()
	reg.Register("echo", "echo text", func(ctx context.Context, args EchoArgs) (string, error) {
		return args.Text, nil
	})

	p := &mockProvider{
		responses: [][]provider.Event{
			// Step 1: tool call
			concat(toolUseBlockEvents("c1", "echo", `{"text":"a"}`), []provider.Event{{Type: provider.EventTypeDone}}),
			// Step 2: another tool call
			concat(toolUseBlockEvents("c2", "echo", `{"text":"b"}`), []provider.Event{{Type: provider.EventTypeDone}}),
			// Step 3: final text response
			concat(textBlockEvents("done"), []provider.Event{{Type: provider.EventTypeDone}}),
		},
	}

	a := New(p, WithTools(reg), WithMaxSteps(10))
	events, err := a.Run(context.Background(), []conversation.Message{userMessage("do stuff")})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}

	var stepsDone int
	var completed bool
	for ev := range events {
		switch ev.Type {
		case EventStepDone:
			stepsDone++
		case EventComplete:
			completed = true
		case EventError:
			t.Fatalf("unexpected error: %v", ev.Err)
		}
	}

	if stepsDone != 3 {
		t.Errorf("steps = %d, want 3", stepsDone)
	}
	if !completed {
		t.Error("expected completion event")
	}
}

func TestRun_MaxSteps(t *testing.T) {
	t.Parallel()

	type NoArgs struct{}
	reg := tools.NewRegistry()
	reg.Register("noop", "noop", func(ctx context.Context, args NoArgs) (string, error) {
		return "ok", nil
	})

	// Always returns a tool call, never completes.
	responses := make([][]provider.Event, 5)
	for i := range responses {
		responses[i] = concat(
			toolUseBlockEvents(fmt.Sprintf("c%d", i), "noop", `{}`),
			[]provider.Event{{Type: provider.EventTypeDone}},
		)
	}

	p := &mockProvider{responses: responses}
	a := New(p, WithTools(reg), WithMaxSteps(3))

	events, err := a.Run(context.Background(), []conversation.Message{userMessage("loop forever")})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}

	var gotMaxStepsErr bool
	var errMessages []conversation.Message
	for ev := range events {
		if ev.Type == EventError && ev.Err == ErrMaxStepsReached {
			gotMaxStepsErr = true
			errMessages = ev.Messages
		}
	}
	if !gotMaxStepsErr {
		t.Error("expected ErrMaxStepsReached")
	}
	// EventError must carry the partial conversation so callers can persist it.
	if len(errMessages) == 0 {
		t.Fatal("expected Messages on ErrMaxStepsReached event")
	}
	// Should have: user + (assistant + tool) * 3 steps = 7
	if len(errMessages) != 7 {
		t.Errorf("errMessages len = %d, want 7 (user + 3 steps of assistant+tool)", len(errMessages))
	}
}

func TestRun_StepErrorCarriesMessages(t *testing.T) {
	t.Parallel()

	type EchoArgs struct {
		Text string `json:"text"`
	}
	reg := tools.NewRegistry()
	reg.Register("echo", "echo", func(ctx context.Context, args EchoArgs) (string, error) {
		return args.Text, nil
	})

	p := &mockProvider{
		responses: [][]provider.Event{
			// Step 1: successful tool call
			concat(toolUseBlockEvents("c1", "echo", `{"text":"a"}`), []provider.Event{{Type: provider.EventTypeDone}}),
			// Step 2: provider error
			{{Type: provider.EventTypeError, Err: fmt.Errorf("provider exploded")}},
		},
	}

	a := New(p, WithTools(reg), WithMaxSteps(10))
	events, err := a.Run(context.Background(), []conversation.Message{userMessage("go")})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}

	var gotError bool
	var errMessages []conversation.Message
	for ev := range events {
		if ev.Type == EventError {
			gotError = true
			errMessages = ev.Messages
		}
	}
	if !gotError {
		t.Fatal("expected error event")
	}
	// Should have: user + assistant + tool = 3 messages from the successful step.
	if len(errMessages) < 3 {
		t.Errorf("errMessages len = %d, want at least 3 (partial conversation from before error)", len(errMessages))
	}
}

func TestStep_HooksAreCalled(t *testing.T) {
	t.Parallel()

	var beforeCalled, tokenCalled bool

	events := concat(textBlockEvents("hi"), []provider.Event{{Type: provider.EventTypeDone}})
	p := &mockProvider{responses: [][]provider.Event{events}}

	hooks := Hooks{
		BeforeCall: func(ctx context.Context, msgs []conversation.Message) {
			beforeCalled = true
		},
		OnToken: func(ctx context.Context, delta string) {
			tokenCalled = true
		},
	}

	a := New(p, WithHooks(hooks))
	msgs := []conversation.Message{userMessage("test")}

	_, _, err := a.Step(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Step error: %v", err)
	}

	if !beforeCalled {
		t.Error("BeforeCall hook not called")
	}
	if !tokenCalled {
		t.Error("OnToken hook not called (should fire for BlockText deltas)")
	}
}

// TestAgent_Model_Getter verifies that the Model() getter reflects the
// name supplied to New and tracks SetModel. The TUI reads this field
// to render the status line, so a stale value would silently mislabel
// the displayed model.
func TestAgent_Model_Getter(t *testing.T) {
	t.Parallel()

	a := New(&mockProvider{}, WithModel("gpt-4o"))
	if got := a.Model(); got != "gpt-4o" {
		t.Errorf("Model() after New = %q, want gpt-4o", got)
	}
	a.SetModel("claude-3-5-sonnet")
	if got := a.Model(); got != "claude-3-5-sonnet" {
		t.Errorf("Model() after SetModel = %q, want claude-3-5-sonnet", got)
	}
}

// TestAgent_Run_ForwardsUsageEvent asserts that a provider
// EventTypeUsage fired during a Step is forwarded as an agent
// EventUsage on the channel returned by Run, carrying the same
// numbers. This is the data path that ultimately lets the TUI show
// actual token counts.
func TestAgent_Run_ForwardsUsageEvent(t *testing.T) {
	t.Parallel()

	usage := conversation.TokenUsage{PromptTokens: 11, CompletionTokens: 3, TotalTokens: 14}
	events := append([]provider.Event{}, textBlockEvents("hello")...)
	events = append(events,
		provider.Event{Type: provider.EventTypeUsage, Usage: &usage},
		provider.Event{Type: provider.EventTypeDone},
	)

	p := &mockProvider{responses: [][]provider.Event{events}}
	a := New(p, WithModel("test-model"), WithMaxSteps(2))

	ch, err := a.Run(context.Background(), []conversation.Message{userMessage("hi")})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	var gotUsage *conversation.TokenUsage
	var sawComplete bool
	for ev := range ch {
		switch ev.Type {
		case EventUsage:
			if ev.Usage == nil {
				t.Fatal("EventUsage had nil Usage")
			}
			u := *ev.Usage
			gotUsage = &u
		case EventComplete:
			sawComplete = true
		case EventError:
			t.Fatalf("unexpected error: %v", ev.Err)
		}
	}
	if !sawComplete {
		t.Error("never saw EventComplete")
	}
	if gotUsage == nil {
		t.Fatal("never saw EventUsage")
	}
	if *gotUsage != usage {
		t.Errorf("usage = %+v, want %+v", *gotUsage, usage)
	}
}

// routingProvider is a concurrency-safe fake provider for testing
// concurrent Run calls on one Agent. It is stateless per call: the
// script to play is selected by the first user message's text, and
// the position within that script by how many assistant messages the
// request already carries. That makes it safe for any number of
// interleaved Complete calls without locks.
type routingProvider struct {
	scripts map[string][][]provider.Event
}

func (r *routingProvider) Complete(_ context.Context, req provider.Request) (<-chan provider.Event, error) {
	var key string
	assistants := 0
	for _, m := range req.Messages {
		if key == "" && m.Role == conversation.RoleUser {
			key = m.TextContent()
		}
		if m.Role == conversation.RoleAssistant {
			assistants++
		}
	}
	script, ok := r.scripts[key]
	if !ok {
		return nil, fmt.Errorf("routingProvider: no script for %q", key)
	}
	if assistants >= len(script) {
		return nil, fmt.Errorf("routingProvider: script %q exhausted at step %d", key, assistants)
	}
	ch := make(chan provider.Event, 64)
	go func() {
		defer close(ch)
		for _, ev := range script[assistants] {
			ch <- ev
		}
	}()
	return ch, nil
}

func (r *routingProvider) Models(_ context.Context) ([]provider.ModelMeta, error) {
	return nil, nil
}

// TestRun_ConcurrentRunsAreIsolated is the regression test for the
// old Run implementation, which temporarily mutated a.opts.hooks on
// the shared Agent per step: two concurrent Run calls raced on those
// fields and delivered each other's events. Each run here drives a
// distinct tool call and final text; every event on a run's channel
// must belong to that run only, and the shared user hooks must still
// fire once per tool call. Run with -race.
func TestRun_ConcurrentRunsAreIsolated(t *testing.T) {
	t.Parallel()

	registry := tools.NewRegistry()
	type noArgs struct{}
	for _, name := range []string{"tool_A", "tool_B"} {
		name := name
		if err := registry.Register(name, "test tool "+name, func(_ context.Context, _ noArgs) (string, error) {
			return "result-" + name[len(name)-1:], nil
		}); err != nil {
			t.Fatalf("register %s: %v", name, err)
		}
	}

	scripts := map[string][][]provider.Event{}
	for _, suffix := range []string{"A", "B"} {
		scripts["run-"+suffix] = [][]provider.Event{
			concat(
				thinkingBlockEvents("thinking-"+suffix),
				toolUseBlockEvents("call-"+suffix, "tool_"+suffix, "{}"),
				[]provider.Event{{Type: provider.EventTypeDone}},
			),
			concat(
				textBlockEvents("done-"+suffix),
				[]provider.Event{{Type: provider.EventTypeDone}},
			),
		}
	}
	p := &routingProvider{scripts: scripts}

	// Shared user hooks: record which tool calls fired so we can
	// assert wrapping still chains to user hooks under concurrency.
	var mu sync.Mutex
	hookToolCalls := map[string]int{}
	a := New(p,
		WithTools(registry),
		WithMaxSteps(5),
		WithHooks(Hooks{
			OnToolCall: func(_ context.Context, call conversation.ToolCall) {
				mu.Lock()
				hookToolCalls[call.Name]++
				mu.Unlock()
			},
		}),
	)

	runAndCollect := func(suffix string) ([]Event, error) {
		msgs := []conversation.Message{userMessage("run-" + suffix)}
		ch, err := a.Run(context.Background(), msgs)
		if err != nil {
			return nil, err
		}
		var out []Event
		for ev := range ch {
			out = append(out, ev)
		}
		return out, nil
	}

	var wg sync.WaitGroup
	results := make(map[string][]Event, 2)
	errs := make(map[string]error, 2)
	for _, suffix := range []string{"A", "B"} {
		suffix := suffix
		wg.Add(1)
		go func() {
			defer wg.Done()
			evs, err := runAndCollect(suffix)
			mu.Lock()
			results[suffix] = evs
			errs[suffix] = err
			mu.Unlock()
		}()
	}
	wg.Wait()

	other := map[string]string{"A": "B", "B": "A"}
	for _, suffix := range []string{"A", "B"} {
		if errs[suffix] != nil {
			t.Fatalf("run %s: %v", suffix, errs[suffix])
		}
		evs := results[suffix]

		var (
			toolCalls   []conversation.ToolCall
			toolResults []string
			tokens      string
			complete    *Event
		)
		for i := range evs {
			ev := evs[i]
			switch ev.Type {
			case EventToolCall:
				toolCalls = append(toolCalls, *ev.ToolCall)
			case EventToolResult:
				toolResults = append(toolResults, ev.ToolResult)
			case EventToken:
				tokens += ev.Content
			case EventComplete:
				complete = &evs[i]
			case EventError:
				t.Fatalf("run %s: unexpected error event: %v", suffix, ev.Err)
			}

			// No event on this run's channel may carry the other
			// run's markers, in any field.
			for _, s := range []string{ev.Content, ev.ToolResult} {
				if strings.Contains(s, "-"+other[suffix]) {
					t.Errorf("run %s: cross-wired event content %q", suffix, s)
				}
			}
			if ev.ToolCall != nil && strings.Contains(ev.ToolCall.Name+ev.ToolCall.ID, "-"+other[suffix]) {
				t.Errorf("run %s: cross-wired tool call %+v", suffix, ev.ToolCall)
			}
			if ev.Block != nil && strings.Contains(ev.Block.Text+ev.Block.ToolName+ev.Block.ToolCallID, other[suffix]) {
				t.Errorf("run %s: cross-wired block %+v", suffix, ev.Block)
			}
		}

		if len(toolCalls) != 1 || toolCalls[0].Name != "tool_"+suffix || toolCalls[0].ID != "call-"+suffix {
			t.Errorf("run %s: tool calls = %+v, want exactly one tool_%s/call-%s", suffix, toolCalls, suffix, suffix)
		}
		if len(toolResults) != 1 || toolResults[0] != "result-"+suffix {
			t.Errorf("run %s: tool results = %q, want [result-%s]", suffix, toolResults, suffix)
		}
		if tokens != "done-"+suffix {
			t.Errorf("run %s: streamed text = %q, want %q", suffix, tokens, "done-"+suffix)
		}
		if complete == nil {
			t.Fatalf("run %s: no EventComplete", suffix)
		}

		// The completed conversation must round-trip this run's data:
		// user prompt, its own tool result payload, its own final text.
		final := complete.Messages
		if got := final[0].TextContent(); got != "run-"+suffix {
			t.Errorf("run %s: first message = %q", suffix, got)
		}
		last := final[len(final)-1]
		if got := last.TextContent(); got != "done-"+suffix {
			t.Errorf("run %s: final message = %q, want done-%s", suffix, got, suffix)
		}
		foundResult := false
		for _, m := range final {
			for _, tr := range m.ToolResults() {
				if tr.Text == "result-"+suffix {
					foundResult = true
				}
				if strings.Contains(tr.Text, "result-"+other[suffix]) {
					t.Errorf("run %s: conversation contains other run's tool result %q", suffix, tr.Text)
				}
			}
		}
		if !foundResult {
			t.Errorf("run %s: persisted conversation missing tool result result-%s", suffix, suffix)
		}
	}

	// Shared user hooks fired exactly once per run's tool call.
	mu.Lock()
	defer mu.Unlock()
	if hookToolCalls["tool_A"] != 1 || hookToolCalls["tool_B"] != 1 {
		t.Errorf("user OnToolCall hook counts = %v, want tool_A:1 tool_B:1", hookToolCalls)
	}
}

// TestRun_AllHooksChainDuringRun drives a full Run (tool step + final
// text) with every user hook set and asserts each hook received the
// actual payloads — block order, delta text, tool call/result values,
// usage numbers, and the final evolved conversation in AfterComplete.
// This pins the emittingHooks chaining contract: Run's event emission
// must never swallow or reorder the user's own hooks.
func TestRun_AllHooksChainDuringRun(t *testing.T) {
	t.Parallel()

	registry := tools.NewRegistry()
	type noArgs struct{}
	if err := registry.Register("echo", "echo tool", func(_ context.Context, _ noArgs) (string, error) {
		return "echo-result", nil
	}); err != nil {
		t.Fatalf("register: %v", err)
	}

	usage := conversation.TokenUsage{PromptTokens: 7, CompletionTokens: 3, TotalTokens: 10}
	p := &mockProvider{responses: [][]provider.Event{
		concat(
			thinkingBlockEvents("hmm"),
			toolUseBlockEvents("call-1", "echo", "{}"),
			[]provider.Event{{Type: provider.EventTypeUsage, Usage: &usage}, {Type: provider.EventTypeDone}},
		),
		concat(
			textBlockEvents("final"),
			[]provider.Event{{Type: provider.EventTypeDone}},
		),
	}}

	// All writes below happen in Run's goroutine; the channel close
	// after the deferred AfterComplete gives the happens-before edge
	// that makes reading them after the drain race-free.
	var (
		beforeCalls int
		blockStarts []conversation.BlockType
		blockEnds   []conversation.BlockType
		deltas      = map[conversation.BlockType]string{}
		tokens      string
		toolCalls   []string
		toolResults []string
		usages      []conversation.TokenUsage
		afterMsgs   []conversation.Message
	)
	a := New(p,
		WithTools(registry),
		WithMaxSteps(5),
		WithHooks(Hooks{
			BeforeCall: func(_ context.Context, msgs []conversation.Message) { beforeCalls++ },
			OnToken:    func(_ context.Context, delta string) { tokens += delta },
			OnBlockStart: func(_ context.Context, bt conversation.BlockType) {
				blockStarts = append(blockStarts, bt)
			},
			OnBlockDelta: func(_ context.Context, bt conversation.BlockType, delta string) {
				deltas[bt] += delta
			},
			OnBlockEnd: func(_ context.Context, blk conversation.Block) {
				blockEnds = append(blockEnds, blk.Type)
			},
			OnToolCall: func(_ context.Context, call conversation.ToolCall) {
				toolCalls = append(toolCalls, call.Name)
			},
			OnToolResult: func(_ context.Context, call conversation.ToolCall, result string, err error) {
				toolResults = append(toolResults, result)
			},
			OnUsage: func(_ context.Context, u conversation.TokenUsage) {
				usages = append(usages, u)
			},
			AfterComplete: func(_ context.Context, msgs []conversation.Message) {
				afterMsgs = append([]conversation.Message(nil), msgs...)
			},
		}),
	)

	ch, err := a.Run(context.Background(), []conversation.Message{userMessage("hi")})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	for ev := range ch {
		if ev.Type == EventError {
			t.Fatalf("unexpected error event: %v", ev.Err)
		}
	}

	if beforeCalls != 2 {
		t.Errorf("BeforeCall fired %d times, want 2 (one per step)", beforeCalls)
	}
	wantOrder := []conversation.BlockType{conversation.BlockThinking, conversation.BlockToolUse, conversation.BlockText}
	if fmt.Sprint(blockStarts) != fmt.Sprint(wantOrder) {
		t.Errorf("OnBlockStart order = %v, want %v", blockStarts, wantOrder)
	}
	if fmt.Sprint(blockEnds) != fmt.Sprint(wantOrder) {
		t.Errorf("OnBlockEnd order = %v, want %v", blockEnds, wantOrder)
	}
	if deltas[conversation.BlockThinking] != "hmm" || deltas[conversation.BlockText] != "final" {
		t.Errorf("OnBlockDelta accumulated = %v, want thinking:hmm text:final", deltas)
	}
	if tokens != "final" {
		t.Errorf("OnToken accumulated = %q, want %q (text deltas only)", tokens, "final")
	}
	if len(toolCalls) != 1 || toolCalls[0] != "echo" {
		t.Errorf("OnToolCall = %v, want [echo]", toolCalls)
	}
	if len(toolResults) != 1 || toolResults[0] != "echo-result" {
		t.Errorf("OnToolResult = %v, want [echo-result]", toolResults)
	}
	if len(usages) != 1 || usages[0] != usage {
		t.Errorf("OnUsage = %v, want [%v]", usages, usage)
	}
	if afterMsgs == nil {
		t.Fatal("AfterComplete never fired")
	}
	if got := afterMsgs[len(afterMsgs)-1].TextContent(); got != "final" {
		t.Errorf("AfterComplete final message = %q, want %q", got, "final")
	}
	foundResult := false
	for _, m := range afterMsgs {
		for _, tr := range m.ToolResults() {
			if tr.Text == "echo-result" {
				foundResult = true
			}
		}
	}
	if !foundResult {
		t.Error("AfterComplete conversation missing tool result echo-result")
	}
}

// TestAgent_SetProviderAndSetModel pins the (single-goroutine) mutator
// contract: SetProvider swaps which backend the next Step hits, and
// SetModel changes both the Model() getter and the model stamped on
// the next assistant message.
func TestAgent_SetProviderAndSetModel(t *testing.T) {
	t.Parallel()

	pA := &mockProvider{responses: [][]provider.Event{concat(
		textBlockEvents("from-A"), []provider.Event{{Type: provider.EventTypeDone}},
	)}}
	pB := &mockProvider{responses: [][]provider.Event{concat(
		textBlockEvents("from-B"), []provider.Event{{Type: provider.EventTypeDone}},
	)}}

	a := New(pA)
	// No WithModel and mockProvider exposes no Model() method, so the
	// getter's final fallback must report empty.
	if got := a.Model(); got != "" {
		t.Errorf("Model() before SetModel = %q, want empty", got)
	}

	_, res, err := a.Step(context.Background(), []conversation.Message{userMessage("hi")})
	if err != nil {
		t.Fatalf("Step on provider A: %v", err)
	}
	if got := res.AssistantMessage.TextContent(); got != "from-A" {
		t.Errorf("provider A response = %q, want from-A", got)
	}

	a.SetProvider(pB)
	a.SetModel("model-b")
	if got := a.Model(); got != "model-b" {
		t.Errorf("Model() after SetModel = %q, want model-b", got)
	}

	_, res, err = a.Step(context.Background(), []conversation.Message{userMessage("hi")})
	if err != nil {
		t.Fatalf("Step on provider B: %v", err)
	}
	if got := res.AssistantMessage.TextContent(); got != "from-B" {
		t.Errorf("provider B response = %q, want from-B", got)
	}
	if got := res.AssistantMessage.Model; got != "model-b" {
		t.Errorf("assistant message model = %q, want model-b", got)
	}
}

// TestStep_ProviderCompleteError covers the path where the provider's
// Complete call itself fails (before any stream exists): Step must
// wrap the error and leave the conversation untouched.
func TestStep_ProviderCompleteError(t *testing.T) {
	t.Parallel()

	p := &mockProvider{} // zero responses → Complete returns an error
	a := New(p)

	in := []conversation.Message{userMessage("hi")}
	out, res, err := a.Step(context.Background(), in)
	if err == nil {
		t.Fatal("Step returned nil error, want provider failure")
	}
	if !strings.Contains(err.Error(), "agent: complete:") {
		t.Errorf("error = %q, want it wrapped with %q", err, "agent: complete:")
	}
	if len(out) != 1 || out[0].TextContent() != "hi" {
		t.Errorf("conversation after failed Step = %v, want the original single message", out)
	}
	if res.Done || len(res.ToolResults) != 0 {
		t.Errorf("StepResult after failure = %+v, want zero value", res)
	}
}
