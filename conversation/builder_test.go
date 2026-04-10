package conversation

import "testing"

func TestBuilder_BasicConversation(t *testing.T) {
	t.Parallel()

	msgs := NewBuilder().
		System("You are a helpful assistant.").
		User("Hello").
		Assistant("Hi there!").
		Build()

	if len(msgs) != 3 {
		t.Fatalf("got %d messages, want 3", len(msgs))
	}

	tests := []struct {
		idx     int
		role    Role
		content string
	}{
		{0, RoleSystem, "You are a helpful assistant."},
		{1, RoleUser, "Hello"},
		{2, RoleAssistant, "Hi there!"},
	}
	for _, tt := range tests {
		if msgs[tt.idx].Role != tt.role {
			t.Errorf("msgs[%d].Role = %v, want %v", tt.idx, msgs[tt.idx].Role, tt.role)
		}
		if got := msgs[tt.idx].TextContent(); got != tt.content {
			t.Errorf("msgs[%d].TextContent() = %q, want %q", tt.idx, got, tt.content)
		}
	}
}

func TestBuilder_ToolUseAndResult(t *testing.T) {
	t.Parallel()

	msgs := NewBuilder().
		System("system").
		User("read /tmp/test").
		Add(Message{Role: RoleAssistant}).
		Text("let me read that").
		ToolUse("call_1", "read_file", `{"path":"/tmp/test"}`).
		ToolResult("call_1", "file contents here").
		Build()

	if len(msgs) != 4 {
		t.Fatalf("got %d messages, want 4", len(msgs))
	}

	// Assistant message now has text + tool_use blocks in order.
	if !msgs[2].HasToolUses() {
		t.Error("msgs[2] should have tool uses")
	}
	if got, want := msgs[2].TextContent(), "let me read that"; got != want {
		t.Errorf("assistant text = %q, want %q", got, want)
	}
	uses := msgs[2].ToolUses()
	if len(uses) != 1 {
		t.Fatalf("tool uses len = %d, want 1", len(uses))
	}
	if uses[0].ToolName != "read_file" {
		t.Errorf("tool use name = %q, want %q", uses[0].ToolName, "read_file")
	}
	if uses[0].ToolCallID != "call_1" {
		t.Errorf("tool use call id = %q, want %q", uses[0].ToolCallID, "call_1")
	}

	// Tool result message.
	if msgs[3].Role != RoleTool {
		t.Errorf("msgs[3].Role = %v, want %v", msgs[3].Role, RoleTool)
	}
	results := msgs[3].ToolResults()
	if len(results) != 1 {
		t.Fatalf("tool results len = %d, want 1", len(results))
	}
	if results[0].ToolCallID != "call_1" {
		t.Errorf("tool result call id = %q, want %q", results[0].ToolCallID, "call_1")
	}
	if results[0].Text != "file contents here" {
		t.Errorf("tool result text = %q, want %q", results[0].Text, "file contents here")
	}
}

func TestBuilder_InterleavedAssistantTurn(t *testing.T) {
	t.Parallel()

	// Build an assistant turn containing the full interleaved sequence
	// the plan calls out as the Phase 1 goal:
	//   thinking → text → tool_use → thinking → tool_use → thinking → text
	msgs := NewBuilder().
		User("do it").
		Add(Message{Role: RoleAssistant}).
		Thinking("planning").
		Text("Let me check.").
		ToolUse("c1", "read_file", `{"p":"a"}`).
		Thinking("found it").
		ToolUse("c2", "grep", `{"q":"b"}`).
		Thinking("analyzing").
		Text("The bug is X.").
		Build()

	if len(msgs) != 2 {
		t.Fatalf("got %d messages, want 2", len(msgs))
	}
	assistant := msgs[1]

	wantTypes := []BlockType{
		BlockThinking, BlockText, BlockToolUse,
		BlockThinking, BlockToolUse, BlockThinking, BlockText,
	}
	if len(assistant.Blocks) != len(wantTypes) {
		t.Fatalf("assistant blocks len = %d, want %d", len(assistant.Blocks), len(wantTypes))
	}
	for i, want := range wantTypes {
		if assistant.Blocks[i].Type != want {
			t.Errorf("Blocks[%d].Type = %q, want %q", i, assistant.Blocks[i].Type, want)
		}
	}

	if got, want := assistant.TextContent(), "Let me check.The bug is X."; got != want {
		t.Errorf("TextContent() = %q, want %q", got, want)
	}
}

func TestBuilder_ImageBlocks(t *testing.T) {
	t.Parallel()

	msgs := NewBuilder().
		User("what is this?").
		Image("image/png", []byte{0x89, 0x50, 0x4E, 0x47}).
		Build()

	if len(msgs) != 1 {
		t.Fatalf("got %d messages, want 1", len(msgs))
	}
	user := msgs[0]
	if len(user.Blocks) != 2 {
		t.Fatalf("user blocks = %d, want 2 (text + image)", len(user.Blocks))
	}
	if user.Blocks[0].Type != BlockText {
		t.Errorf("blocks[0].Type = %v, want BlockText", user.Blocks[0].Type)
	}
	if user.Blocks[1].Type != BlockImage {
		t.Errorf("blocks[1].Type = %v, want BlockImage", user.Blocks[1].Type)
	}
	if user.Blocks[1].MimeType != "image/png" {
		t.Errorf("mime = %q", user.Blocks[1].MimeType)
	}
	if len(user.Blocks[1].ImageData) != 4 {
		t.Errorf("image data len = %d", len(user.Blocks[1].ImageData))
	}
}

func TestBuilder_BuildReturnsCopy(t *testing.T) {
	t.Parallel()

	b := NewBuilder().User("hello")
	msgs1 := b.Build()
	b.User("world")
	msgs2 := b.Build()

	if len(msgs1) != 1 {
		t.Errorf("msgs1 should have 1 message, got %d", len(msgs1))
	}
	if len(msgs2) != 2 {
		t.Errorf("msgs2 should have 2 messages, got %d", len(msgs2))
	}
}

func TestBuilder_Len(t *testing.T) {
	t.Parallel()

	b := NewBuilder()
	if b.Len() != 0 {
		t.Errorf("empty builder Len() = %d, want 0", b.Len())
	}
	b.User("a").User("b")
	if b.Len() != 2 {
		t.Errorf("Len() = %d, want 2", b.Len())
	}
}

func TestBuilder_Add(t *testing.T) {
	t.Parallel()

	msg := Message{Role: RoleUser, Blocks: []Block{{Type: BlockText, Text: "custom"}}}
	msgs := NewBuilder().Add(msg).Build()

	if len(msgs) != 1 {
		t.Fatalf("got %d messages, want 1", len(msgs))
	}
	if got := msgs[0].TextContent(); got != "custom" {
		t.Errorf("TextContent() = %q, want %q", got, "custom")
	}
}
