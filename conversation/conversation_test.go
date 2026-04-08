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

func TestMessage_HasToolCalls(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		msg  Message
		want bool
	}{
		{"no tool calls", Message{Role: RoleAssistant, Content: "hello"}, false},
		{"empty tool calls", Message{Role: RoleAssistant, ToolCalls: []ToolCall{}}, false},
		{"with tool calls", Message{
			Role: RoleAssistant,
			ToolCalls: []ToolCall{
				{ID: "1", Name: "read_file", Arguments: `{"path":"/tmp/foo"}`},
			},
		}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := tt.msg.HasToolCalls(); got != tt.want {
				t.Errorf("HasToolCalls() = %v, want %v", got, tt.want)
			}
		})
	}
}
