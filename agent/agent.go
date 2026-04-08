package agent

import (
	"context"
	"fmt"

	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/provider"
)

// ErrMaxStepsReached is returned when the agent loop hits MaxSteps.
var ErrMaxStepsReached = fmt.Errorf("agent: max steps reached")

// StepResult describes the outcome of a single Step.
type StepResult struct {
	// AssistantMessage is the message to append to the conversation.
	AssistantMessage conversation.Message

	// ToolResults are ready to append after AssistantMessage.
	// Empty if the model produced no tool calls.
	ToolResults []conversation.Message

	// Done is true if the model produced a final response (no tool calls).
	Done bool
}

// Agent drives the ReAct loop: call LLM, dispatch tools, repeat.
type Agent struct {
	provider provider.Provider
	opts     options
}

// New creates a new Agent with the given provider and options.
func New(p provider.Provider, opts ...Option) *Agent {
	o := defaultOptions()
	for _, opt := range opts {
		opt(&o)
	}
	return &Agent{provider: p, opts: o}
}

// Step executes one complete LLM round-trip plus tool dispatch.
//
// It takes the current conversation, calls the provider, collects the
// response (assembling streaming tokens into a complete message), dispatches
// any tool calls, and returns the updated conversation plus a StepResult.
func (a *Agent) Step(ctx context.Context, msgs []conversation.Message) ([]conversation.Message, StepResult, error) {
	// Hook: before call.
	if a.opts.hooks.BeforeCall != nil {
		a.opts.hooks.BeforeCall(ctx, msgs)
	}

	req := provider.Request{
		Model:       a.opts.model,
		Messages:    msgs,
		Tools:       a.opts.registry.Definitions(),
		MaxTokens:   a.opts.maxTokens,
		Temperature: a.opts.temperature,
		Stream:      true,
	}

	events, err := a.provider.Complete(ctx, req)
	if err != nil {
		return msgs, StepResult{}, fmt.Errorf("agent: complete: %w", err)
	}

	// Collect the response.
	var content string
	var toolCalls []conversation.ToolCall

	for ev := range events {
		switch ev.Type {
		case provider.EventTypeToken:
			content += ev.Content
			if a.opts.hooks.OnToken != nil {
				a.opts.hooks.OnToken(ctx, ev.Content)
			}
		case provider.EventTypeToolCall:
			toolCalls = append(toolCalls, *ev.Call)
		case provider.EventTypeError:
			return msgs, StepResult{}, fmt.Errorf("agent: provider error: %w", ev.Err)
		case provider.EventTypeDone:
			// Stream complete.
		}
	}

	// Build assistant message.
	assistantMsg := conversation.Message{
		Role:      conversation.RoleAssistant,
		Content:   content,
		ToolCalls: toolCalls,
	}

	result := StepResult{
		AssistantMessage: assistantMsg,
		Done:             len(toolCalls) == 0,
	}

	// Append assistant message.
	msgs = append(msgs, assistantMsg)

	// Dispatch tool calls.
	for _, tc := range toolCalls {
		if a.opts.hooks.OnToolCall != nil {
			a.opts.hooks.OnToolCall(ctx, tc)
		}

		toolResult, toolErr := a.opts.registry.Dispatch(ctx, tc.Name, tc.Arguments)

		if a.opts.hooks.OnToolResult != nil {
			a.opts.hooks.OnToolResult(ctx, tc, toolResult, toolErr)
		}

		resultContent := toolResult
		if toolErr != nil {
			resultContent = fmt.Sprintf("Error: %v", toolErr)
		}

		toolMsg := conversation.Message{
			Role:       conversation.RoleTool,
			Content:    resultContent,
			ToolCallID: tc.ID,
		}
		result.ToolResults = append(result.ToolResults, toolMsg)
		msgs = append(msgs, toolMsg)
	}

	return msgs, result, nil
}

// Run drives the ReAct loop until one of:
//   - The model returns a final text response with no tool calls
//   - MaxSteps is reached (returns ErrMaxStepsReached)
//   - ctx is cancelled
//
// Events are emitted to the returned channel as the loop progresses.
func (a *Agent) Run(ctx context.Context, msgs []conversation.Message) (<-chan Event, error) {
	events := make(chan Event, 64)

	go func() {
		defer close(events)
		defer func() {
			if a.opts.hooks.AfterComplete != nil {
				a.opts.hooks.AfterComplete(ctx, msgs)
			}
		}()

		for step := 0; step < a.opts.maxSteps; step++ {
			// Hook wrappers that also emit events.
			origOnToken := a.opts.hooks.OnToken
			a.opts.hooks.OnToken = func(ctx context.Context, delta string) {
				events <- Event{Type: EventToken, Content: delta, Step: step}
				if origOnToken != nil {
					origOnToken(ctx, delta)
				}
			}

			origOnToolCall := a.opts.hooks.OnToolCall
			a.opts.hooks.OnToolCall = func(ctx context.Context, call conversation.ToolCall) {
				events <- Event{Type: EventToolCall, ToolCall: &call, Step: step}
				if origOnToolCall != nil {
					origOnToolCall(ctx, call)
				}
			}

			origOnToolResult := a.opts.hooks.OnToolResult
			a.opts.hooks.OnToolResult = func(ctx context.Context, call conversation.ToolCall, result string, err error) {
				events <- Event{Type: EventToolResult, ToolResult: result, Step: step}
				if origOnToolResult != nil {
					origOnToolResult(ctx, call, result, err)
				}
			}

			var result StepResult
			var err error
			msgs, result, err = a.Step(ctx, msgs)

			// Restore original hooks.
			a.opts.hooks.OnToken = origOnToken
			a.opts.hooks.OnToolCall = origOnToolCall
			a.opts.hooks.OnToolResult = origOnToolResult

			if err != nil {
				events <- Event{Type: EventError, Err: err, Step: step, Messages: msgs}
				return
			}

			events <- Event{Type: EventStepDone, Step: step}

			if result.Done {
				events <- Event{Type: EventComplete, Step: step, Messages: msgs}
				return
			}
		}

		events <- Event{Type: EventError, Err: ErrMaxStepsReached, Step: a.opts.maxSteps, Messages: msgs}
	}()

	return events, nil
}
