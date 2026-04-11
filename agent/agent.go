package agent

import (
	"context"
	"fmt"
	"time"

	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/provider"
)

// ErrMaxStepsReached is returned when the agent loop hits MaxSteps.
var ErrMaxStepsReached = fmt.Errorf("agent: max steps reached")

// StepResult describes the outcome of a single Step.
type StepResult struct {
	// AssistantMessage is the message to append to the conversation.
	AssistantMessage conversation.Message

	// ToolResults holds the tool-role message appended after the
	// assistant message. At most one message, carrying one BlockToolResult
	// per BlockToolUse in the assistant message, matching the
	// Anthropic shape. The field is a slice for historical reasons;
	// new callers should treat len() ∈ {0, 1}.
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

// SetProvider swaps the underlying provider at runtime. The caller is
// responsible for ensuring no concurrent Run is in progress.
func (a *Agent) SetProvider(p provider.Provider) { a.provider = p }

// SetModel overrides the model used on the next Step/Run. The caller is
// responsible for ensuring no concurrent Run is in progress.
func (a *Agent) SetModel(model string) { a.opts.model = model }

// Model returns the model name the agent will use on its next
// Step/Run. It mirrors whatever the most recent New / SetModel call
// set, with a fallback to the provider's own configured model when
// the agent was built without WithModel — that's the common pattern
// for single-model embedders, and without the fallback the TUI
// status line would show nothing.
func (a *Agent) Model() string {
	if a.opts.model != "" {
		return a.opts.model
	}
	type modeled interface{ Model() string }
	if p, ok := a.provider.(modeled); ok {
		return p.Model()
	}
	return ""
}

// Step executes one complete LLM round-trip plus tool dispatch.
//
// It takes the current conversation, calls the provider, collects the
// response (assembling streaming block events into ordered blocks),
// dispatches any tool calls, and returns the updated conversation plus
// a StepResult.
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

	start := time.Now()
	events, err := a.provider.Complete(ctx, req)
	if err != nil {
		return msgs, StepResult{}, fmt.Errorf("agent: complete: %w", err)
	}

	// Accumulate blocks in the order the provider closes them. This
	// preserves interleaved thinking/text/tool_use ordering faithfully.
	var blocks []conversation.Block

	for ev := range events {
		switch ev.Type {
		case provider.EventTypeBlockStart:
			if a.opts.hooks.OnBlockStart != nil {
				a.opts.hooks.OnBlockStart(ctx, ev.BlockType)
			}
		case provider.EventTypeBlockDelta:
			if a.opts.hooks.OnBlockDelta != nil {
				a.opts.hooks.OnBlockDelta(ctx, ev.BlockType, ev.Content)
			}
			if ev.BlockType == conversation.BlockText && a.opts.hooks.OnToken != nil {
				a.opts.hooks.OnToken(ctx, ev.Content)
			}
		case provider.EventTypeBlockEnd:
			if ev.Block != nil {
				blocks = append(blocks, *ev.Block)
				if a.opts.hooks.OnBlockEnd != nil {
					a.opts.hooks.OnBlockEnd(ctx, *ev.Block)
				}
			}
		case provider.EventTypeToolCall:
			// Fired after the matching BlockEnd for a BlockToolUse —
			// used only for the convenience OnToolCall hook, the
			// block itself has already been appended via BlockEnd.
			if ev.Call != nil && a.opts.hooks.OnToolCall != nil {
				a.opts.hooks.OnToolCall(ctx, *ev.Call)
			}
		case provider.EventTypeUsage:
			if ev.Usage != nil && a.opts.hooks.OnUsage != nil {
				a.opts.hooks.OnUsage(ctx, *ev.Usage)
			}
		case provider.EventTypeError:
			return msgs, StepResult{}, fmt.Errorf("agent: provider error: %w", ev.Err)
		case provider.EventTypeDone:
			// Stream complete.
		}
	}

	elapsed := time.Since(start)

	// Build assistant message.
	assistantMsg := conversation.Message{
		Role:      conversation.RoleAssistant,
		Blocks:    blocks,
		Model:     a.opts.model,
		CreatedAt: start,
		Duration:  elapsed,
	}
	conversation.EnsureMessageIDs(&assistantMsg)

	toolUses := assistantMsg.ToolUses()

	result := StepResult{
		AssistantMessage: assistantMsg,
		Done:             len(toolUses) == 0,
	}

	// Append assistant message.
	msgs = append(msgs, assistantMsg)

	// Dispatch tool calls. Build a single tool-role message carrying
	// one tool_result block per tool_use, matching the Anthropic shape
	// and keeping the turn atomic for replay.
	if len(toolUses) > 0 {
		toolMsg := conversation.Message{
			Role:      conversation.RoleTool,
			CreatedAt: time.Now(),
		}
		for _, tu := range toolUses {
			tc := conversation.ToolCall{
				ID:        tu.ToolCallID,
				Name:      tu.ToolName,
				Arguments: tu.ToolArgsJSON,
			}

			toolResult, toolErr := a.opts.registry.Dispatch(ctx, tu.ToolName, tu.ToolArgsJSON)

			if a.opts.hooks.OnToolResult != nil {
				a.opts.hooks.OnToolResult(ctx, tc, toolResult, toolErr)
			}

			resultContent := toolResult
			isErr := false
			if toolErr != nil {
				resultContent = fmt.Sprintf("Error: %v", toolErr)
				isErr = true
			}

			toolMsg.Blocks = append(toolMsg.Blocks, conversation.Block{
				Type:        conversation.BlockToolResult,
				ToolCallID:  tu.ToolCallID,
				Text:        resultContent,
				ToolIsError: isErr,
			})
		}
		conversation.EnsureMessageIDs(&toolMsg)
		result.ToolResults = []conversation.Message{toolMsg}
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
			origOnBlockStart := a.opts.hooks.OnBlockStart
			a.opts.hooks.OnBlockStart = func(ctx context.Context, bt conversation.BlockType) {
				events <- Event{Type: EventBlockStart, BlockType: bt, Step: step}
				if origOnBlockStart != nil {
					origOnBlockStart(ctx, bt)
				}
			}

			origOnBlockDelta := a.opts.hooks.OnBlockDelta
			a.opts.hooks.OnBlockDelta = func(ctx context.Context, bt conversation.BlockType, delta string) {
				events <- Event{Type: EventBlockDelta, BlockType: bt, Content: delta, Step: step}
				if bt == conversation.BlockText {
					events <- Event{Type: EventToken, Content: delta, Step: step}
				}
				if origOnBlockDelta != nil {
					origOnBlockDelta(ctx, bt, delta)
				}
			}

			origOnBlockEnd := a.opts.hooks.OnBlockEnd
			a.opts.hooks.OnBlockEnd = func(ctx context.Context, blk conversation.Block) {
				blkCopy := blk
				events <- Event{Type: EventBlockEnd, BlockType: blk.Type, Block: &blkCopy, Step: step}
				if origOnBlockEnd != nil {
					origOnBlockEnd(ctx, blk)
				}
			}

			origOnToken := a.opts.hooks.OnToken
			a.opts.hooks.OnToken = func(ctx context.Context, delta string) {
				if origOnToken != nil {
					origOnToken(ctx, delta)
				}
			}

			origOnToolCall := a.opts.hooks.OnToolCall
			a.opts.hooks.OnToolCall = func(ctx context.Context, call conversation.ToolCall) {
				callCopy := call
				events <- Event{Type: EventToolCall, ToolCall: &callCopy, Step: step}
				if origOnToolCall != nil {
					origOnToolCall(ctx, call)
				}
			}

			origOnToolResult := a.opts.hooks.OnToolResult
			a.opts.hooks.OnToolResult = func(ctx context.Context, call conversation.ToolCall, result string, err error) {
				callCopy := call
				events <- Event{Type: EventToolResult, ToolCall: &callCopy, ToolResult: result, Step: step}
				if origOnToolResult != nil {
					origOnToolResult(ctx, call, result, err)
				}
			}

			origOnUsage := a.opts.hooks.OnUsage
			a.opts.hooks.OnUsage = func(ctx context.Context, usage conversation.TokenUsage) {
				usageCopy := usage
				events <- Event{Type: EventUsage, Usage: &usageCopy, Step: step}
				if origOnUsage != nil {
					origOnUsage(ctx, usage)
				}
			}

			var result StepResult
			var err error
			msgs, result, err = a.Step(ctx, msgs)

			// Restore original hooks.
			a.opts.hooks.OnBlockStart = origOnBlockStart
			a.opts.hooks.OnBlockDelta = origOnBlockDelta
			a.opts.hooks.OnBlockEnd = origOnBlockEnd
			a.opts.hooks.OnToken = origOnToken
			a.opts.hooks.OnToolCall = origOnToolCall
			a.opts.hooks.OnToolResult = origOnToolResult
			a.opts.hooks.OnUsage = origOnUsage

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
