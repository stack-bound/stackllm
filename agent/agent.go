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
//
// Concurrency: an Agent is safe for concurrent use. Multiple Run
// and/or Step calls may be in flight on the same Agent at once — each
// Run works on a private copy of the agent's options, so events and
// hooks never cross-wire between concurrent runs. The only exceptions
// are SetProvider and SetModel, which mutate the Agent and must not be
// called while any Run or Step is in progress.
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

// SetProvider swaps the underlying provider at runtime.
//
// SetProvider mutates the Agent and is NOT safe to call concurrently
// with Run or Step: the caller must ensure no Run or Step is in
// progress (and none is started concurrently) when calling it.
func (a *Agent) SetProvider(p provider.Provider) { a.provider = p }

// SetModel overrides the model used on the next Step/Run.
//
// SetModel mutates the Agent and is NOT safe to call concurrently
// with Run or Step: the caller must ensure no Run or Step is in
// progress (and none is started concurrently) when calling it.
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
	return a.step(ctx, msgs, &a.opts)
}

// step is the option-parameterised core of Step. The exported Step
// passes the agent's own options; Run passes a per-run copy whose
// hooks additionally emit to that run's event channel. Taking the
// options as a parameter (instead of reading a.opts) is what makes
// concurrent Run calls on one Agent safe: nothing in the loop ever
// writes to the shared Agent.
func (a *Agent) step(ctx context.Context, msgs []conversation.Message, o *options) ([]conversation.Message, StepResult, error) {
	// Hook: before call.
	if o.hooks.BeforeCall != nil {
		o.hooks.BeforeCall(ctx, msgs)
	}

	req := provider.Request{
		Model:       o.model,
		Messages:    msgs,
		Tools:       o.registry.Definitions(),
		MaxTokens:   o.maxTokens,
		Temperature: o.temperature,
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
			if o.hooks.OnBlockStart != nil {
				o.hooks.OnBlockStart(ctx, ev.BlockType)
			}
		case provider.EventTypeBlockDelta:
			if o.hooks.OnBlockDelta != nil {
				o.hooks.OnBlockDelta(ctx, ev.BlockType, ev.Content)
			}
			if ev.BlockType == conversation.BlockText && o.hooks.OnToken != nil {
				o.hooks.OnToken(ctx, ev.Content)
			}
		case provider.EventTypeBlockEnd:
			if ev.Block != nil {
				blocks = append(blocks, *ev.Block)
				if o.hooks.OnBlockEnd != nil {
					o.hooks.OnBlockEnd(ctx, *ev.Block)
				}
			}
		case provider.EventTypeToolCall:
			// Fired after the matching BlockEnd for a BlockToolUse —
			// used only for the convenience OnToolCall hook, the
			// block itself has already been appended via BlockEnd.
			if ev.Call != nil && o.hooks.OnToolCall != nil {
				o.hooks.OnToolCall(ctx, *ev.Call)
			}
		case provider.EventTypeUsage:
			if ev.Usage != nil && o.hooks.OnUsage != nil {
				o.hooks.OnUsage(ctx, *ev.Usage)
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
		Model:     o.model,
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

			toolResult, toolErr := o.registry.Dispatch(ctx, tu.ToolName, tu.ToolArgsJSON)

			if o.hooks.OnToolResult != nil {
				o.hooks.OnToolResult(ctx, tc, toolResult, toolErr)
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
//
// Run is safe for concurrent use: multiple Run calls may be in flight
// on the same Agent at once. Each call snapshots the agent's options
// and wraps its hooks in a per-run copy, so every returned channel
// receives only its own run's events and user-supplied hooks fire with
// that run's data — nothing on the shared Agent is mutated.
func (a *Agent) Run(ctx context.Context, msgs []conversation.Message) (<-chan Event, error) {
	events := make(chan Event, 64)

	// Snapshot the options once. The loop below only ever reads this
	// copy (and per-step derived copies), never the Agent itself.
	base := a.opts

	go func() {
		defer close(events)
		defer func() {
			if base.hooks.AfterComplete != nil {
				base.hooks.AfterComplete(ctx, msgs)
			}
		}()

		for step := 0; step < base.maxSteps; step++ {
			// Per-step options copy whose hooks emit this run's
			// events and then chain to the user-supplied hooks.
			stepOpts := base
			stepOpts.hooks = emittingHooks(base.hooks, events, step)

			var result StepResult
			var err error
			msgs, result, err = a.step(ctx, msgs, &stepOpts)

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

		events <- Event{Type: EventError, Err: ErrMaxStepsReached, Step: base.maxSteps, Messages: msgs}
	}()

	return events, nil
}

// emittingHooks returns a Hooks value whose callbacks emit Run events
// for the given step to the given channel and then chain to the
// corresponding user-supplied hook from orig. Hooks with no event to
// emit (BeforeCall, OnToken, AfterComplete) pass through unchanged via
// the initial copy.
func emittingHooks(orig Hooks, events chan<- Event, step int) Hooks {
	h := orig

	h.OnBlockStart = func(ctx context.Context, bt conversation.BlockType) {
		events <- Event{Type: EventBlockStart, BlockType: bt, Step: step}
		if orig.OnBlockStart != nil {
			orig.OnBlockStart(ctx, bt)
		}
	}

	h.OnBlockDelta = func(ctx context.Context, bt conversation.BlockType, delta string) {
		events <- Event{Type: EventBlockDelta, BlockType: bt, Content: delta, Step: step}
		if bt == conversation.BlockText {
			events <- Event{Type: EventToken, Content: delta, Step: step}
		}
		if orig.OnBlockDelta != nil {
			orig.OnBlockDelta(ctx, bt, delta)
		}
	}

	h.OnBlockEnd = func(ctx context.Context, blk conversation.Block) {
		blkCopy := blk
		events <- Event{Type: EventBlockEnd, BlockType: blk.Type, Block: &blkCopy, Step: step}
		if orig.OnBlockEnd != nil {
			orig.OnBlockEnd(ctx, blk)
		}
	}

	h.OnToolCall = func(ctx context.Context, call conversation.ToolCall) {
		callCopy := call
		events <- Event{Type: EventToolCall, ToolCall: &callCopy, Step: step}
		if orig.OnToolCall != nil {
			orig.OnToolCall(ctx, call)
		}
	}

	h.OnToolResult = func(ctx context.Context, call conversation.ToolCall, result string, err error) {
		callCopy := call
		events <- Event{Type: EventToolResult, ToolCall: &callCopy, ToolResult: result, Step: step}
		if orig.OnToolResult != nil {
			orig.OnToolResult(ctx, call, result, err)
		}
	}

	h.OnUsage = func(ctx context.Context, usage conversation.TokenUsage) {
		usageCopy := usage
		events <- Event{Type: EventUsage, Usage: &usageCopy, Step: step}
		if orig.OnUsage != nil {
			orig.OnUsage(ctx, usage)
		}
	}

	return h
}
