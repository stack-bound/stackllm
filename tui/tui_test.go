package tui

import (
	"strings"
	"testing"

	"github.com/stack-bound/stackllm/conversation"
)

func TestRenderMessage_User(t *testing.T) {
	t.Parallel()
	msg := conversation.Message{Role: conversation.RoleUser, Content: "hello"}
	out := RenderMessage(msg)
	if !strings.Contains(out, "hello") {
		t.Errorf("expected output to contain 'hello', got %q", out)
	}
}

func TestRenderMessage_Assistant(t *testing.T) {
	t.Parallel()
	msg := conversation.Message{Role: conversation.RoleAssistant, Content: "response"}
	out := RenderMessage(msg)
	if !strings.Contains(out, "response") {
		t.Errorf("expected output to contain 'response', got %q", out)
	}
}

func TestRenderMessage_AssistantWithToolCalls(t *testing.T) {
	t.Parallel()
	msg := conversation.Message{
		Role: conversation.RoleAssistant,
		ToolCalls: []conversation.ToolCall{
			{ID: "1", Name: "read_file", Arguments: `{"path":"/tmp"}`},
		},
	}
	out := RenderMessage(msg)
	if !strings.Contains(out, "read_file") {
		t.Errorf("expected output to contain 'read_file', got %q", out)
	}
}

func TestRenderMessage_Tool(t *testing.T) {
	t.Parallel()
	msg := conversation.Message{Role: conversation.RoleTool, Content: "result", ToolCallID: "call_1"}
	out := RenderMessage(msg)
	if !strings.Contains(out, "result") {
		t.Errorf("expected output to contain 'result', got %q", out)
	}
	if !strings.Contains(out, "call_1") {
		t.Errorf("expected output to contain 'call_1', got %q", out)
	}
}

func TestRenderConversation(t *testing.T) {
	t.Parallel()
	msgs := []conversation.Message{
		{Role: conversation.RoleSystem, Content: "sys"},
		{Role: conversation.RoleUser, Content: "hi"},
		{Role: conversation.RoleAssistant, Content: "hello"},
	}
	out := RenderConversation(msgs)
	if !strings.Contains(out, "hi") || !strings.Contains(out, "hello") {
		t.Errorf("expected conversation content in output, got %q", out)
	}
}

func TestDeviceCodePrompt(t *testing.T) {
	t.Parallel()
	out := DeviceCodePrompt("ABCD-1234", "https://example.com/device")
	if !strings.Contains(out, "ABCD-1234") {
		t.Errorf("expected code in output, got %q", out)
	}
	if !strings.Contains(out, "example.com") {
		t.Errorf("expected URL in output, got %q", out)
	}
}

func TestWebFlowPrompt(t *testing.T) {
	t.Parallel()
	out := WebFlowPrompt("https://auth.example.com/login")
	if !strings.Contains(out, "auth.example.com") {
		t.Errorf("expected URL in output, got %q", out)
	}
}

func TestAuthHooks(t *testing.T) {
	t.Parallel()
	hooks := AuthHooks()
	if hooks.OnToken == nil {
		t.Error("expected OnToken to be set")
	}
	if hooks.OnToolCall == nil {
		t.Error("expected OnToolCall to be set")
	}
	if hooks.OnToolResult == nil {
		t.Error("expected OnToolResult to be set")
	}
	if hooks.AfterComplete == nil {
		t.Error("expected AfterComplete to be set")
	}
}

func TestTruncate(t *testing.T) {
	t.Parallel()

	tests := []struct {
		input string
		max   int
		want  string
	}{
		{"short", 10, "short"},
		{"this is longer", 5, "this ..."},
		{"exact", 5, "exact"},
	}
	for _, tt := range tests {
		got := truncate(tt.input, tt.max)
		if got != tt.want {
			t.Errorf("truncate(%q, %d) = %q, want %q", tt.input, tt.max, got, tt.want)
		}
	}
}
