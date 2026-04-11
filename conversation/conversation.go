package conversation

import (
	"time"

	"github.com/google/uuid"
)

// Role represents the role of a message sender in a conversation.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// BlockType enumerates the kinds of content blocks a message can carry.
type BlockType string

const (
	BlockText             BlockType = "text"              // visible text
	BlockThinking         BlockType = "thinking"          // model reasoning / chain-of-thought
	BlockRedactedThinking BlockType = "redacted_thinking" // opaque encrypted reasoning (Anthropic)
	BlockImage            BlockType = "image"             // image input (and, in future, output)
	BlockToolUse          BlockType = "tool_use"          // model requesting a tool invocation
	BlockToolResult       BlockType = "tool_result"       // output from a tool invocation
)

// ToolCall is a lightweight view of a BlockToolUse used by hook
// signatures and provider events. It mirrors the id/name/arguments
// fields of BlockToolUse.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string
}

// TokenUsage reports how many tokens a single provider turn consumed.
// PromptTokens is the size of the request context (the figure users
// care about when tracking context window headroom); CompletionTokens
// is what the model generated. TotalTokens mirrors the upstream field
// where it's present and will equal the sum of the two otherwise.
type TokenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ArtifactRef points at a row in an external artifacts store. The
// referenced payload may be the full content of a tool_result block,
// image bytes, or redacted thinking data. When ArtifactRef is non-nil
// on a loaded Block, the inline fields (Text, ImageData, etc.) may hold
// only a preview; call the store's Hydrate helper to fetch the full
// payload on demand.
type ArtifactRef struct {
	ID       string `json:"id"` // opaque store-assigned ID
	MimeType string `json:"mime_type"`
	ByteSize int64  `json:"byte_size"`
	SHA256   string `json:"sha256"`
}

// Block is one content element within a Message. Messages hold an ordered
// slice of Blocks — the order is the replay timeline.
//
// Each block type populates a distinct subset of fields. Helpers on
// Message (TextContent, ToolUses, etc.) give typed access without
// forcing callers to switch on Type directly.
type Block struct {
	ID   string    `json:"id"` // stable identifier, assigned eagerly at construction
	Type BlockType `json:"type"`

	// Text / thinking content. Also holds small tool_result output and
	// the short preview for large tool results that were offloaded to
	// an artifact.
	Text string `json:"text,omitempty"`

	// Tool use (model → tool) fields.
	ToolCallID   string `json:"tool_call_id,omitempty"` // unique id for this tool call (matches provider's call_id)
	ToolName     string `json:"tool_name,omitempty"`
	ToolArgsJSON string `json:"tool_args_json,omitempty"`

	// Tool result (tool → model) fields. ToolCallID refers back to the
	// originating tool_use block.
	ToolIsError bool `json:"tool_is_error,omitempty"`

	// Image fields. Exactly one of ImageURL / ImageData is set when
	// Type == BlockImage.
	MimeType  string `json:"mime_type,omitempty"`
	ImageURL  string `json:"image_url,omitempty"`
	ImageData []byte `json:"image_data,omitempty"`

	// Redacted thinking opaque payload (Anthropic only).
	RedactedData []byte `json:"redacted_data,omitempty"`

	// ArtifactRef is set by the store when Load offloads a large payload
	// to the artifacts table instead of hydrating it in-memory.
	ArtifactRef *ArtifactRef `json:"artifact_ref,omitempty"`
}

// Message is one turn (or partial turn) in a conversation.
//
// For assistant messages, Blocks captures the ordered interleaving of
// thinking → text → tool_use the model produced. For tool messages,
// Blocks holds one or more tool_result blocks linked to the tool_use
// blocks in the preceding assistant message.
type Message struct {
	ID        string        `json:"id"`
	Role      Role          `json:"role"`
	Blocks    []Block       `json:"blocks"`
	Model     string        `json:"model,omitempty"` // which model produced this message (assistant only)
	CreatedAt time.Time     `json:"created_at,omitempty"`
	Duration  time.Duration `json:"duration,omitempty"` // wall-clock time to produce this message (assistant only)
}

// EnsureMessageIDs assigns stable IDs to the message and each of its
// blocks if they do not already have one.
func EnsureMessageIDs(m *Message) {
	if m.ID == "" {
		m.ID = NewID()
	}
	for i := range m.Blocks {
		if m.Blocks[i].ID == "" {
			m.Blocks[i].ID = NewID()
		}
	}
}

// NewID returns a fresh UUIDv7 string suitable for use as a
// Message.ID, Block.ID, Session.ID, artifact ID, or any other eager
// identifier within the library. UUIDv7 is time-ordered (48 bits of
// Unix-millis prefix followed by randomness), so sequential inserts
// cluster at the tail of a B-tree index — a small but real win for
// SQLite write locality as sessions grow.
//
// Producers of Messages and Blocks call this directly when they need
// to mint an ID at construction time (e.g. provider stream parsers
// minting Block.ID before emitting BlockEnd events).
func NewID() string {
	id, err := uuid.NewV7()
	if err != nil {
		// uuid.NewV7 only errors if the underlying rand.Read fails,
		// which would also break crypto anywhere else in the binary.
		panic("conversation: uuid.NewV7 failed: " + err.Error())
	}
	return id.String()
}

// IsSystem returns true if the message has the system role.
func (m Message) IsSystem() bool {
	return m.Role == RoleSystem
}

// TextContent returns the concatenation of all BlockText blocks in
// order. It is the convenience accessor for "give me the visible text"
// — callers that need block-level fidelity should walk Blocks directly.
func (m Message) TextContent() string {
	var out string
	for _, b := range m.Blocks {
		if b.Type == BlockText {
			out += b.Text
		}
	}
	return out
}

// ThinkingText returns the concatenation of all BlockThinking blocks in
// order. Useful for surfacing model reasoning in a UI.
func (m Message) ThinkingText() string {
	var out string
	for _, b := range m.Blocks {
		if b.Type == BlockThinking {
			out += b.Text
		}
	}
	return out
}

// ToolUses returns all BlockToolUse blocks in order.
func (m Message) ToolUses() []Block {
	var out []Block
	for _, b := range m.Blocks {
		if b.Type == BlockToolUse {
			out = append(out, b)
		}
	}
	return out
}

// ToolResults returns all BlockToolResult blocks in order.
func (m Message) ToolResults() []Block {
	var out []Block
	for _, b := range m.Blocks {
		if b.Type == BlockToolResult {
			out = append(out, b)
		}
	}
	return out
}

// HasToolUses reports whether the message contains any tool_use blocks.
func (m Message) HasToolUses() bool {
	for _, b := range m.Blocks {
		if b.Type == BlockToolUse {
			return true
		}
	}
	return false
}
