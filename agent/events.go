package agent

import "github.com/stack-bound/stackllm/conversation"

// EventType identifies the kind of agent event.
type EventType int

const (
	EventBlockStart EventType = iota // a new block opened in the provider stream
	EventBlockDelta                  // streaming delta for the currently open block
	EventBlockEnd                    // the currently open block closed
	EventToken                       // convenience alias: BlockDelta filtered to BlockText
	EventToolCall                    // tool call dispatched
	EventToolResult                  // tool call completed
	EventUsage                       // token usage reported for the completed step
	EventStepDone                    // single step completed
	EventComplete                    // agent loop finished
	EventError                       // terminal error
)

// Event is emitted by the agent loop during Run.
//
// Field population by Type:
//
//	EventBlockStart: BlockType
//	EventBlockDelta: BlockType, Content
//	EventBlockEnd:   BlockType, Block
//	EventToken:      Content (convenience for BlockText deltas)
//	EventToolCall:   ToolCall
//	EventToolResult: ToolCall, ToolResult
//	EventUsage:      Usage
//	EventStepDone:   Step
//	EventComplete:   Messages
//	EventError:      Err, Messages
type Event struct {
	Type       EventType
	BlockType  conversation.BlockType
	Block      *conversation.Block
	Content    string                  // set for EventToken and EventBlockDelta
	ToolCall   *conversation.ToolCall  // set for EventToolCall
	ToolResult string                  // set for EventToolResult
	Usage      *conversation.TokenUsage // set for EventUsage
	Err        error                   // set for EventError
	Step       int                     // current step number
	Messages   []conversation.Message  // set for EventComplete and EventError
}
