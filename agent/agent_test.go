package agent

import (
	"context"
	"fmt"
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
