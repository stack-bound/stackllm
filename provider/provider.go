package provider

import (
	"context"

	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/tools"
)

// EventType identifies the kind of streaming event.
//
// Providers emit an ordered sequence of block events that lets
// downstream consumers reconstruct the interleaved timeline a model
// produced (thinking → text → tool_use → thinking → text, etc.):
//
//   - EventTypeBlockStart fires once when a new block opens.
//   - EventTypeBlockDelta fires repeatedly while the block's content
//     streams in. For text/thinking blocks the delta is the partial
//     text; for tool_use blocks the delta is the partial arguments
//     JSON.
//   - EventTypeBlockEnd fires once when the block closes and carries
//     the fully accumulated Block in ev.Block.
//
// EventTypeToolCall is a convenience alias fired immediately after the
// matching EventTypeBlockEnd for BlockToolUse blocks so callers that
// only care about dispatched tool calls can listen on a single event
// type. It is retained for backward compatibility and ease of use —
// the ordered source of truth is the block event stream.
//
// EventTypeDone marks the end of the stream. EventTypeError is
// terminal and carries the underlying error.
type EventType int

const (
	EventTypeBlockStart EventType = iota // a new block has opened
	EventTypeBlockDelta                  // incremental content for the currently open block
	EventTypeBlockEnd                    // the currently open block has closed
	EventTypeToolCall                    // convenience: fired after BlockEnd for a BlockToolUse
	EventTypeUsage                       // token usage report for this turn; fires once, before EventTypeDone
	EventTypeDone                        // stream finished, no error
	EventTypeError                       // terminal error
)

// Endpoint selects which OpenAI-compatible API path the provider uses.
// The empty string means the legacy /chat/completions path; this is the
// default for every provider config helper.
//
// Some Copilot models (e.g. gpt-5.4-mini, gpt-5.x-codex) are only
// reachable via /responses and return unsupported_api_for_model when
// called via /chat/completions. Set Config.Endpoint = EndpointResponses
// for those models.
const (
	EndpointChatCompletions = ""
	EndpointResponses       = "/responses"
)

// ToolCall is an alias for conversation.ToolCall.
type ToolCall = conversation.ToolCall

// TokenUsage is an alias for conversation.TokenUsage so callers in the
// provider package can refer to it without an extra import.
type TokenUsage = conversation.TokenUsage

// Event is a single item in the streaming response.
//
// Field population by Type:
//
//	EventTypeBlockStart: BlockType (the type of the newly opened block)
//	EventTypeBlockDelta: BlockType, Content (the partial content string)
//	EventTypeBlockEnd:   BlockType, Block (the fully accumulated block)
//	EventTypeToolCall:   Call (convenience copy of the closed tool_use block)
//	EventTypeUsage:      Usage (token usage for this turn)
//	EventTypeDone:       (no fields)
//	EventTypeError:      Err
type Event struct {
	Type      EventType
	BlockType conversation.BlockType
	Content   string              // set for EventTypeBlockDelta
	Block     *conversation.Block // set for EventTypeBlockEnd
	Call      *ToolCall           // set for EventTypeToolCall
	Usage     *TokenUsage         // set for EventTypeUsage
	Err       error               // set for EventTypeError
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

// ModelMeta describes one model returned by a provider's /models endpoint.
//
// SupportedEndpoints mirrors the field of the same name on Copilot's
// /models response. It is nil when the upstream API doesn't expose
// per-model endpoint metadata (OpenAI, Gemini, Ollama). Callers should
// treat a nil/empty SupportedEndpoints as "compatible with whatever the
// provider's default endpoint is".
//
// Type mirrors capabilities.type on Copilot ("chat", "embeddings", ...).
// It is empty when not provided.
//
// ModelPickerEnabled mirrors model_picker_enabled on Copilot's /models
// response. It is the same field GitHub's own UI uses to decide which
// models to surface in its model picker; entries with this set to
// false are routers, pinned legacy versions, internal load-test
// models, etc., and should not appear in user-facing pickers. The
// field is a pointer so callers can distinguish "explicitly disabled"
// (false) from "not set by upstream" (nil) — only Copilot exposes it.
//
// ContextWindow is the maximum prompt length in tokens for this model,
// populated from capabilities.limits.max_prompt_tokens on Copilot. It
// is zero for providers that do not expose this field; callers that
// need a value for unknown models should fall back to
// provider.ContextWindow(ID).
type ModelMeta struct {
	ID                 string
	SupportedEndpoints []string
	Type               string
	ModelPickerEnabled *bool
	ContextWindow      int
}

// Provider makes LLM calls and returns a stream of events.
type Provider interface {
	Complete(ctx context.Context, req Request) (<-chan Event, error)
	Models(ctx context.Context) ([]ModelMeta, error)
}
