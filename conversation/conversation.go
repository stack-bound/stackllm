package conversation

// Role represents the role of a message sender in a conversation.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ToolCall represents a tool invocation requested by the model.
type ToolCall struct {
	ID        string // unique identifier for this tool call
	Name      string // tool name to invoke
	Arguments string // raw JSON arguments
}

// Message is a single message in a conversation.
type Message struct {
	Role       Role
	Content    string
	ToolCallID string     // set when Role == RoleTool
	ToolCalls  []ToolCall // set when Role == RoleAssistant with tool calls
}

// IsSystem returns true if the message has the system role.
func (m Message) IsSystem() bool {
	return m.Role == RoleSystem
}

// HasToolCalls returns true if the message contains tool calls.
func (m Message) HasToolCalls() bool {
	return len(m.ToolCalls) > 0
}
