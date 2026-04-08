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

func (m *mockProvider) Models(_ context.Context) ([]string, error) {
	return nil, nil
}

func TestStep_TextOnly(t *testing.T) {
	t.Parallel()

	p := &mockProvider{
		responses: [][]provider.Event{
			{
				{Type: provider.EventTypeToken, Content: "Hello "},
				{Type: provider.EventTypeToken, Content: "world"},
				{Type: provider.EventTypeDone},
			},
		},
	}

	a := New(p)
	msgs := []conversation.Message{
		{Role: conversation.RoleUser, Content: "Hi"},
	}

	msgs, result, err := a.Step(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Step error: %v", err)
	}
	if !result.Done {
		t.Error("expected Done=true for text-only response")
	}
	if result.AssistantMessage.Content != "Hello world" {
		t.Errorf("content = %q, want %q", result.AssistantMessage.Content, "Hello world")
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

	p := &mockProvider{
		responses: [][]provider.Event{
			{
				{Type: provider.EventTypeToolCall, Call: &provider.ToolCall{
					ID: "call_1", Name: "echo", Arguments: `{"text":"hello"}`,
				}},
				{Type: provider.EventTypeDone},
			},
		},
	}

	type EchoArgs struct {
		Text string `json:"text"`
	}

	reg := tools.NewRegistry()
	reg.Register("echo", "echo text", func(ctx context.Context, args EchoArgs) (string, error) {
		return args.Text, nil
	})

	a := New(p, WithTools(reg))
	msgs := []conversation.Message{
		{Role: conversation.RoleUser, Content: "echo hello"},
	}

	msgs, result, err := a.Step(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Step error: %v", err)
	}
	if result.Done {
		t.Error("expected Done=false when tool calls present")
	}
	if len(result.ToolResults) != 1 {
		t.Fatalf("expected 1 tool result, got %d", len(result.ToolResults))
	}
	if result.ToolResults[0].Content != "hello" {
		t.Errorf("tool result = %q, want %q", result.ToolResults[0].Content, "hello")
	}
	if result.ToolResults[0].ToolCallID != "call_1" {
		t.Errorf("tool call id = %q, want %q", result.ToolResults[0].ToolCallID, "call_1")
	}
	// msgs: user + assistant + tool = 3
	if len(msgs) != 3 {
		t.Errorf("msgs len = %d, want 3", len(msgs))
	}
}

func TestStep_ToolError(t *testing.T) {
	t.Parallel()

	p := &mockProvider{
		responses: [][]provider.Event{
			{
				{Type: provider.EventTypeToolCall, Call: &provider.ToolCall{
					ID: "call_1", Name: "fail", Arguments: `{}`,
				}},
				{Type: provider.EventTypeDone},
			},
		},
	}

	type FailArgs struct{}

	reg := tools.NewRegistry()
	reg.Register("fail", "always fails", func(ctx context.Context, args FailArgs) (string, error) {
		return "", fmt.Errorf("something broke")
	})

	a := New(p, WithTools(reg))
	msgs := []conversation.Message{
		{Role: conversation.RoleUser, Content: "fail"},
	}

	msgs, result, err := a.Step(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Step should not return error for tool errors, got: %v", err)
	}
	if result.ToolResults[0].Content != "Error: something broke" {
		t.Errorf("tool result = %q, want error message", result.ToolResults[0].Content)
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
			{
				{Type: provider.EventTypeToolCall, Call: &provider.ToolCall{
					ID: "c1", Name: "echo", Arguments: `{"text":"a"}`,
				}},
				{Type: provider.EventTypeDone},
			},
			// Step 2: another tool call
			{
				{Type: provider.EventTypeToolCall, Call: &provider.ToolCall{
					ID: "c2", Name: "echo", Arguments: `{"text":"b"}`,
				}},
				{Type: provider.EventTypeDone},
			},
			// Step 3: final text response
			{
				{Type: provider.EventTypeToken, Content: "done"},
				{Type: provider.EventTypeDone},
			},
		},
	}

	a := New(p, WithTools(reg), WithMaxSteps(10))
	events, err := a.Run(context.Background(), []conversation.Message{
		{Role: conversation.RoleUser, Content: "do stuff"},
	})
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
		responses[i] = []provider.Event{
			{Type: provider.EventTypeToolCall, Call: &provider.ToolCall{
				ID: fmt.Sprintf("c%d", i), Name: "noop", Arguments: `{}`,
			}},
			{Type: provider.EventTypeDone},
		}
	}

	p := &mockProvider{responses: responses}
	a := New(p, WithTools(reg), WithMaxSteps(3))

	events, err := a.Run(context.Background(), []conversation.Message{
		{Role: conversation.RoleUser, Content: "loop forever"},
	})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}

	var gotMaxStepsErr bool
	for ev := range events {
		if ev.Type == EventError && ev.Err == ErrMaxStepsReached {
			gotMaxStepsErr = true
		}
	}
	if !gotMaxStepsErr {
		t.Error("expected ErrMaxStepsReached")
	}
}

func TestStep_HooksAreCalled(t *testing.T) {
	t.Parallel()

	var beforeCalled, tokenCalled, afterCalled bool

	p := &mockProvider{
		responses: [][]provider.Event{
			{
				{Type: provider.EventTypeToken, Content: "hi"},
				{Type: provider.EventTypeDone},
			},
		},
	}

	hooks := Hooks{
		BeforeCall: func(ctx context.Context, msgs []conversation.Message) {
			beforeCalled = true
		},
		OnToken: func(ctx context.Context, delta string) {
			tokenCalled = true
		},
	}

	a := New(p, WithHooks(hooks))
	msgs := []conversation.Message{{Role: conversation.RoleUser, Content: "test"}}

	_, _, err := a.Step(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Step error: %v", err)
	}

	if !beforeCalled {
		t.Error("BeforeCall hook not called")
	}
	if !tokenCalled {
		t.Error("OnToken hook not called")
	}

	_ = afterCalled
}
