package agent

import (
	"context"

	"github.com/stack-bound/stackllm/conversation"
)

// Hooks are called at each stage of the agent loop.
// All fields are optional — nil hooks are silently skipped.
type Hooks struct {
	// Called before each LLM request with the current message slice.
	BeforeCall func(ctx context.Context, msgs []conversation.Message)

	// Called for each streaming token as it arrives.
	OnToken func(ctx context.Context, delta string)

	// Called when the LLM emits a complete tool call (before dispatch).
	OnToolCall func(ctx context.Context, call conversation.ToolCall)

	// Called after a tool returns (before appending to conversation).
	OnToolResult func(ctx context.Context, call conversation.ToolCall, result string, err error)

	// Called when the agent loop completes (naturally or via MaxSteps).
	AfterComplete func(ctx context.Context, msgs []conversation.Message)
}
