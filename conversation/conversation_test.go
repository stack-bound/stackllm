package conversation

import "testing"

func TestRoleConstants(t *testing.T) {
	t.Parallel()

	tests := []struct {
		role Role
		want string
	}{
		{RoleSystem, "system"},
		{RoleUser, "user"},
		{RoleAssistant, "assistant"},
		{RoleTool, "tool"},
	}
	for _, tt := range tests {
		if string(tt.role) != tt.want {
			t.Errorf("Role %v = %q, want %q", tt.role, string(tt.role), tt.want)
		}
	}
}

func TestMessage_IsSystem(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		msg  Message
		want bool
	}{
		{"system message", Message{Role: RoleSystem}, true},
		{"user message", Message{Role: RoleUser}, false},
		{"assistant message", Message{Role: RoleAssistant}, false},
		{"tool message", Message{Role: RoleTool}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := tt.msg.IsSystem(); got != tt.want {
				t.Errorf("IsSystem() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMessage_HasToolUses(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		msg  Message
		want bool
	}{
		{
			name: "no blocks",
			msg:  Message{Role: RoleAssistant},
			want: false,
		},
		{
			name: "text only",
			msg: Message{
				Role:   RoleAssistant,
				Blocks: []Block{{Type: BlockText, Text: "hello"}},
			},
			want: false,
		},
		{
			name: "with tool use",
			msg: Message{
				Role: RoleAssistant,
				Blocks: []Block{
					{Type: BlockText, Text: "let me check"},
					{Type: BlockToolUse, ToolCallID: "1", ToolName: "read_file", ToolArgsJSON: `{"path":"/tmp/foo"}`},
				},
			},
			want: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := tt.msg.HasToolUses(); got != tt.want {
				t.Errorf("HasToolUses() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestMessage_InterleavedBlocks asserts the Blocks helpers return the
// expected subsets of an assistant message whose turn interleaves
// thinking, text, and tool_use blocks — the critical Phase 1 goal.
func TestMessage_InterleavedBlocks(t *testing.T) {
	t.Parallel()

	msg := Message{
		Role: RoleAssistant,
		Blocks: []Block{
			{Type: BlockThinking, Text: "planning"},
			{Type: BlockText, Text: "Let me check."},
			{Type: BlockToolUse, ToolCallID: "c1", ToolName: "read_file", ToolArgsJSON: `{"p":"a"}`},
			{Type: BlockThinking, Text: "found it"},
			{Type: BlockToolUse, ToolCallID: "c2", ToolName: "grep", ToolArgsJSON: `{"q":"b"}`},
			{Type: BlockThinking, Text: "analyzing"},
			{Type: BlockText, Text: "The bug is X."},
		},
	}

	if got, want := msg.TextContent(), "Let me check.The bug is X."; got != want {
		t.Errorf("TextContent() = %q, want %q", got, want)
	}
	if got, want := msg.ThinkingText(), "planningfound itanalyzing"; got != want {
		t.Errorf("ThinkingText() = %q, want %q", got, want)
	}

	uses := msg.ToolUses()
	if len(uses) != 2 {
		t.Fatalf("ToolUses() len = %d, want 2", len(uses))
	}
	if uses[0].ToolName != "read_file" || uses[1].ToolName != "grep" {
		t.Errorf("ToolUses order wrong: %q, %q", uses[0].ToolName, uses[1].ToolName)
	}
	if !msg.HasToolUses() {
		t.Error("HasToolUses() should be true")
	}

	// Block order must be preserved verbatim.
	wantTypes := []BlockType{
		BlockThinking, BlockText, BlockToolUse,
		BlockThinking, BlockToolUse, BlockThinking, BlockText,
	}
	if len(msg.Blocks) != len(wantTypes) {
		t.Fatalf("blocks len = %d, want %d", len(msg.Blocks), len(wantTypes))
	}
	for i, want := range wantTypes {
		if msg.Blocks[i].Type != want {
			t.Errorf("Blocks[%d].Type = %q, want %q", i, msg.Blocks[i].Type, want)
		}
	}
}
