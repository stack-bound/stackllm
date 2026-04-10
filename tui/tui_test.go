package tui

import (
	"strings"
	"testing"

	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/profile"
)

func TestRenderMessage_User(t *testing.T) {
	t.Parallel()
	msg := conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hello"}},
	}
	out := RenderMessage(msg)
	if !strings.Contains(out, "hello") {
		t.Errorf("expected output to contain 'hello', got %q", out)
	}
}

func TestRenderMessage_Assistant(t *testing.T) {
	t.Parallel()
	msg := conversation.Message{
		Role:   conversation.RoleAssistant,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "response"}},
	}
	out := RenderMessage(msg)
	if !strings.Contains(out, "response") {
		t.Errorf("expected output to contain 'response', got %q", out)
	}
}

func TestRenderMessage_AssistantWithToolUse(t *testing.T) {
	t.Parallel()
	msg := conversation.Message{
		Role: conversation.RoleAssistant,
		Blocks: []conversation.Block{
			{Type: conversation.BlockToolUse, ToolCallID: "1", ToolName: "read_file", ToolArgsJSON: `{"path":"/tmp"}`},
		},
	}
	out := RenderMessage(msg)
	if !strings.Contains(out, "read_file") {
		t.Errorf("expected output to contain 'read_file', got %q", out)
	}
}

// TestRenderMessage_InterleavedThinkingAndText asserts that a rendered
// assistant turn containing thinking blocks interleaved with text
// renders each block in order — i.e. the thinking preview appears
// between the surrounding text segments, not lumped at the top or
// bottom of the message.
func TestRenderMessage_InterleavedThinkingAndText(t *testing.T) {
	t.Parallel()

	msg := conversation.Message{
		Role: conversation.RoleAssistant,
		Blocks: []conversation.Block{
			{Type: conversation.BlockText, Text: "Before."},
			{Type: conversation.BlockThinking, Text: "pondering hard"},
			{Type: conversation.BlockText, Text: "After."},
		},
	}

	out := RenderMessage(msg)
	beforeIdx := strings.Index(out, "Before.")
	thinkIdx := strings.Index(out, "pondering hard")
	afterIdx := strings.Index(out, "After.")

	if beforeIdx < 0 || thinkIdx < 0 || afterIdx < 0 {
		t.Fatalf("all three markers must be present; got %q", out)
	}
	if !(beforeIdx < thinkIdx && thinkIdx < afterIdx) {
		t.Errorf("block order lost: before=%d think=%d after=%d\n%s", beforeIdx, thinkIdx, afterIdx, out)
	}
}

func TestRenderMessage_Tool(t *testing.T) {
	t.Parallel()
	msg := conversation.Message{
		Role: conversation.RoleTool,
		Blocks: []conversation.Block{
			{Type: conversation.BlockToolResult, ToolCallID: "call_1", Text: "result"},
		},
	}
	out := RenderMessage(msg)
	if !strings.Contains(out, "result") {
		t.Errorf("expected output to contain 'result', got %q", out)
	}
	if !strings.Contains(out, "call_1") {
		t.Errorf("expected output to contain 'call_1', got %q", out)
	}
}

func TestRenderMessage_ImagePlaceholder(t *testing.T) {
	t.Parallel()
	msg := conversation.Message{
		Role: conversation.RoleUser,
		Blocks: []conversation.Block{
			{Type: conversation.BlockText, Text: "look"},
			{Type: conversation.BlockImage, MimeType: "image/jpeg", ImageData: []byte{0x01, 0x02, 0x03}},
		},
	}
	out := RenderMessage(msg)
	if !strings.Contains(out, "image/jpeg") {
		t.Errorf("expected mime in image placeholder, got %q", out)
	}
}

func TestRenderConversation(t *testing.T) {
	t.Parallel()
	msgs := []conversation.Message{
		{Role: conversation.RoleSystem, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "sys"}}},
		{Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hi"}}},
		{Role: conversation.RoleAssistant, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hello"}}},
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

// TestRenderModelPicker_DividerBetweenRecentAndAll asserts that the
// picker renders a "── all models ──" divider line at the boundary
// between the recent-used section and the full catalogue when both
// sections are visible. With no recents, no divider should appear.
func TestRenderModelPicker_DividerBetweenRecentAndAll(t *testing.T) {
	t.Parallel()

	m := newTestModel(t)
	m.state = stateModelPicker
	m.models = []profile.ModelInfo{
		// Recents at the top.
		{Provider: "copilot", Model: "claude-opus-4.6"},
		{Provider: "openai", Model: "gpt-4o"},
		// Full list below.
		{Provider: "copilot", Model: "claude-haiku-4.5"},
		{Provider: "copilot", Model: "claude-sonnet-4.6"},
		{Provider: "openai", Model: "gpt-5.4"},
	}
	m.modelRecentCount = 2
	m.modelCursor = 0

	out := m.renderModelPicker()
	if !strings.Contains(out, "── all models ──") {
		t.Errorf("expected divider in picker output, got:\n%s", out)
	}
	if !strings.Contains(out, "copilot/claude-opus-4.6") {
		t.Errorf("expected first recent in output, got:\n%s", out)
	}
	if !strings.Contains(out, "openai/gpt-5.4") {
		t.Errorf("expected last full-list entry in output, got:\n%s", out)
	}

	// Verify the divider sits between the recents and the full list,
	// not before the first recent or after the last entry.
	dividerIdx := strings.Index(out, "── all models ──")
	gpt4oIdx := strings.Index(out, "openai/gpt-4o")
	haikuIdx := strings.Index(out, "copilot/claude-haiku-4.5")
	if dividerIdx < gpt4oIdx {
		t.Errorf("divider before last recent: divider=%d, gpt-4o=%d", dividerIdx, gpt4oIdx)
	}
	if dividerIdx > haikuIdx {
		t.Errorf("divider after first non-recent: divider=%d, haiku=%d", dividerIdx, haikuIdx)
	}

	// With no recents the divider must not appear.
	m.modelRecentCount = 0
	out = m.renderModelPicker()
	if strings.Contains(out, "── all models ──") {
		t.Errorf("divider should not appear when there are no recents, got:\n%s", out)
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
