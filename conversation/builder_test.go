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
		if msgs[tt.idx].Content != tt.content {
			t.Errorf("msgs[%d].Content = %q, want %q", tt.idx, msgs[tt.idx].Content, tt.content)
		}
	}
}

func TestBuilder_ToolCalls(t *testing.T) {
	t.Parallel()

	calls := []ToolCall{
		{ID: "call_1", Name: "read_file", Arguments: `{"path":"/tmp/test"}`},
	}

	msgs := NewBuilder().
		System("system").
		User("read /tmp/test").
		AssistantWithToolCalls("", calls).
		ToolResult("call_1", "file contents here").
		Build()

	if len(msgs) != 4 {
		t.Fatalf("got %d messages, want 4", len(msgs))
	}

	// Assistant message with tool calls.
	if !msgs[2].HasToolCalls() {
		t.Error("msgs[2] should have tool calls")
	}
	if msgs[2].ToolCalls[0].Name != "read_file" {
		t.Errorf("tool call name = %q, want %q", msgs[2].ToolCalls[0].Name, "read_file")
	}

	// Tool result message.
	if msgs[3].Role != RoleTool {
		t.Errorf("msgs[3].Role = %v, want %v", msgs[3].Role, RoleTool)
	}
	if msgs[3].ToolCallID != "call_1" {
		t.Errorf("msgs[3].ToolCallID = %q, want %q", msgs[3].ToolCallID, "call_1")
	}
	if msgs[3].Content != "file contents here" {
		t.Errorf("msgs[3].Content = %q, want %q", msgs[3].Content, "file contents here")
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

	msg := Message{Role: RoleUser, Content: "custom"}
	msgs := NewBuilder().Add(msg).Build()

	if len(msgs) != 1 {
		t.Fatalf("got %d messages, want 1", len(msgs))
	}
	if msgs[0].Content != "custom" {
		t.Errorf("Content = %q, want %q", msgs[0].Content, "custom")
	}
}
