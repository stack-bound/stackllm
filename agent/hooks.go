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

	// Called for each text-block streaming delta (convenience wrapper
	// around OnBlockDelta filtered to BlockText).
	OnToken func(ctx context.Context, delta string)

	// Called when a new block opens in the provider stream.
	OnBlockStart func(ctx context.Context, blockType conversation.BlockType)

	// Called for each streaming delta on the currently open block.
	// For text/thinking blocks the delta is the partial text; for
	// tool_use blocks it is the partial arguments JSON.
	OnBlockDelta func(ctx context.Context, blockType conversation.BlockType, delta string)

	// Called when a block closes. The Block argument is the fully
	// accumulated block (text, tool_use id/name/args, etc.).
	OnBlockEnd func(ctx context.Context, block conversation.Block)

	// Called when the LLM emits a complete tool call (before dispatch).
	OnToolCall func(ctx context.Context, call conversation.ToolCall)

	// Called after a tool returns (before appending to conversation).
	OnToolResult func(ctx context.Context, call conversation.ToolCall, result string, err error)

	// Called when the agent loop completes (naturally or via MaxSteps).
	AfterComplete func(ctx context.Context, msgs []conversation.Message)
}
