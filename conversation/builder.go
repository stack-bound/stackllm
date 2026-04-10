package conversation

// Builder provides a fluent API for constructing message slices. It
// works at two granularities:
//
//   - Message-level: System, User, Assistant, ToolResult start a new
//     message with a single text block (the common ergonomic case).
//   - Block-level: Text, Thinking, ToolUse, Image, ImageURL append an
//     additional block to the most recently added message, letting
//     callers compose interleaved assistant turns that faithfully
//     represent what the model produced.
type Builder struct {
	msgs []Message
}

// NewBuilder creates a new conversation builder.
func NewBuilder() *Builder {
	return &Builder{}
}

// System appends a system message with a single text block.
func (b *Builder) System(content string) *Builder {
	msg := Message{
		Role:   RoleSystem,
		Blocks: []Block{{Type: BlockText, Text: content}},
	}
	EnsureMessageIDs(&msg)
	b.msgs = append(b.msgs, msg)
	return b
}

// User appends a user message with a single text block.
func (b *Builder) User(content string) *Builder {
	msg := Message{
		Role:   RoleUser,
		Blocks: []Block{{Type: BlockText, Text: content}},
	}
	EnsureMessageIDs(&msg)
	b.msgs = append(b.msgs, msg)
	return b
}

// Assistant appends an assistant message with a single text block.
func (b *Builder) Assistant(content string) *Builder {
	msg := Message{
		Role:   RoleAssistant,
		Blocks: []Block{{Type: BlockText, Text: content}},
	}
	EnsureMessageIDs(&msg)
	b.msgs = append(b.msgs, msg)
	return b
}

// ToolResult appends a tool-role message with a single tool_result
// block linked to callID. The isError flag marks execution failures.
func (b *Builder) ToolResult(callID, content string) *Builder {
	msg := Message{
		Role: RoleTool,
		Blocks: []Block{{
			Type:       BlockToolResult,
			ToolCallID: callID,
			Text:       content,
		}},
	}
	EnsureMessageIDs(&msg)
	b.msgs = append(b.msgs, msg)
	return b
}

// Text appends a text block to the most recently added message.
// Panics if no message has been started yet — callers must start a
// message first via System / User / Assistant / ToolResult or Add.
func (b *Builder) Text(s string) *Builder {
	b.appendBlock(Block{Type: BlockText, Text: s})
	return b
}

// Thinking appends a thinking block to the most recently added
// message. Used to represent model reasoning / chain-of-thought.
func (b *Builder) Thinking(s string) *Builder {
	b.appendBlock(Block{Type: BlockThinking, Text: s})
	return b
}

// ToolUse appends a tool_use block (model → tool invocation request)
// to the most recently added message.
func (b *Builder) ToolUse(callID, name, argsJSON string) *Builder {
	b.appendBlock(Block{
		Type:         BlockToolUse,
		ToolCallID:   callID,
		ToolName:     name,
		ToolArgsJSON: argsJSON,
	})
	return b
}

// ToolResultBlock appends a tool_result block to the most recently
// added (tool-role) message. Use this when a single tool-role message
// carries multiple results for sibling tool_use blocks.
func (b *Builder) ToolResultBlock(callID, content string, isErr bool) *Builder {
	b.appendBlock(Block{
		Type:        BlockToolResult,
		ToolCallID:  callID,
		Text:        content,
		ToolIsError: isErr,
	})
	return b
}

// Image appends an inline image block (raw bytes) to the most recently
// added message.
func (b *Builder) Image(mime string, data []byte) *Builder {
	b.appendBlock(Block{
		Type:      BlockImage,
		MimeType:  mime,
		ImageData: data,
	})
	return b
}

// ImageURL appends an image block that references an external URL.
func (b *Builder) ImageURL(mime, url string) *Builder {
	b.appendBlock(Block{
		Type:     BlockImage,
		MimeType: mime,
		ImageURL: url,
	})
	return b
}

// Add appends an arbitrary message.
func (b *Builder) Add(msg Message) *Builder {
	EnsureMessageIDs(&msg)
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

func (b *Builder) appendBlock(blk Block) {
	if len(b.msgs) == 0 {
		panic("conversation: Builder block appended before any message was started")
	}
	last := &b.msgs[len(b.msgs)-1]
	if blk.ID == "" {
		blk.ID = NewID()
	}
	last.Blocks = append(last.Blocks, blk)
}
