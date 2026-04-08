package agent

import "github.com/stack-bound/stackllm/conversation"

// EventType identifies the kind of agent event.
type EventType int

const (
	EventToken      EventType = iota // streaming text delta
	EventToolCall                    // tool call dispatched
	EventToolResult                  // tool call completed
	EventStepDone                    // single step completed
	EventComplete                    // agent loop finished
	EventError                       // terminal error
)

// Event is emitted by the agent loop during Run.
type Event struct {
	Type       EventType
	Content    string             // set for EventToken
	ToolCall   *conversation.ToolCall // set for EventToolCall
	ToolResult string             // set for EventToolResult
	Err        error              // set for EventError
	Step       int                // current step number
	Messages   []conversation.Message // set for EventComplete
}
