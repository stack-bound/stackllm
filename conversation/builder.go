package conversation

// Builder provides a fluent API for constructing message slices.
type Builder struct {
	msgs []Message
}

// NewBuilder creates a new conversation builder.
func NewBuilder() *Builder {
	return &Builder{}
}

// System appends a system message.
func (b *Builder) System(content string) *Builder {
	b.msgs = append(b.msgs, Message{Role: RoleSystem, Content: content})
	return b
}

// User appends a user message.
func (b *Builder) User(content string) *Builder {
	b.msgs = append(b.msgs, Message{Role: RoleUser, Content: content})
	return b
}

// Assistant appends an assistant message with text content.
func (b *Builder) Assistant(content string) *Builder {
	b.msgs = append(b.msgs, Message{Role: RoleAssistant, Content: content})
	return b
}

// AssistantWithToolCalls appends an assistant message that contains tool calls.
func (b *Builder) AssistantWithToolCalls(content string, calls []ToolCall) *Builder {
	b.msgs = append(b.msgs, Message{
		Role:      RoleAssistant,
		Content:   content,
		ToolCalls: calls,
	})
	return b
}

// ToolResult appends a tool result message.
func (b *Builder) ToolResult(callID, content string) *Builder {
	b.msgs = append(b.msgs, Message{
		Role:       RoleTool,
		Content:    content,
		ToolCallID: callID,
	})
	return b
}

// Add appends an arbitrary message.
func (b *Builder) Add(msg Message) *Builder {
	b.msgs = append(b.msgs, msg)
	return b
}

// Build returns the constructed message slice.
func (b *Builder) Build() []Message {
	out := make([]Message, len(b.msgs))
	copy(out, b.msgs)
	return out
}

// Len returns the current number of messages.
func (b *Builder) Len() int {
	return len(b.msgs)
}
