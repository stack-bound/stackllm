package provider

import (
	"context"

	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/tools"
)

// EventType identifies the kind of streaming event.
type EventType int

const (
	EventTypeToken    EventType = iota // streaming text delta
	EventTypeToolCall                  // complete tool call ready for dispatch
	EventTypeDone                      // stream finished, no error
	EventTypeError                     // terminal error
)

// ToolCall is an alias for conversation.ToolCall.
type ToolCall = conversation.ToolCall

// Event is a single item in the streaming response.
type Event struct {
	Type    EventType
	Content string    // set for EventTypeToken
	Call    *ToolCall // set for EventTypeToolCall
	Err     error     // set for EventTypeError
}

// Request is the input to a provider call.
type Request struct {
	Model       string
	Messages    []conversation.Message
	Tools       []tools.Definition
	MaxTokens   int
	Temperature *float64 // nil means use provider default
	Stream      bool     // always true in practice; kept for testing
}

// Provider makes LLM calls and returns a stream of events.
type Provider interface {
	Complete(ctx context.Context, req Request) (<-chan Event, error)
	Models(ctx context.Context) ([]string, error)
}
