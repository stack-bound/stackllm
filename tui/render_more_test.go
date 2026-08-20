package tui

import (
	"strings"
	"testing"
	"time"

	"github.com/stack-bound/stackllm/conversation"
)

func futureTime() time.Time { return time.Now().Add(30 * time.Second) }

func TestRenderMessage_AssistantRedactedThinkingAndImage(t *testing.T) {
	t.Parallel()

	msg := conversation.Message{
		Role: conversation.RoleAssistant,
		Blocks: []conversation.Block{
			{Type: conversation.BlockThinking, Text: "hidden plan"},
			{Type: conversation.BlockRedactedThinking, RedactedData: []byte{1, 2, 3, 4, 5}},
			{Type: conversation.BlockText, Text: "visible text"},
			{Type: conversation.BlockImage, MimeType: "image/png", ImageData: []byte{9, 9}},
		},
	}
	out := RenderMessage(msg)
	if !strings.Contains(out, "thinking: hidden plan") {
		t.Errorf("expected thinking preview, got:\n%s", out)
	}
	if !strings.Contains(out, "[redacted thinking, 5 bytes]") {
		t.Errorf("expected redacted placeholder with byte count, got:\n%s", out)
	}
	if !strings.Contains(out, "visible text") {
		t.Errorf("expected text block, got:\n%s", out)
	}
	if !strings.Contains(out, "[image: image/png, 2 bytes]") {
		t.Errorf("expected image placeholder, got:\n%s", out)
	}
	// Blocks after the first are newline-separated.
	if !strings.Contains(out, "\nvisible text") {
		t.Errorf("expected newline before later block, got:\n%s", out)
	}
}

func TestRenderImagePlaceholder_Variants(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		blk  conversation.Block
		want string
	}{
		{
			"url image",
			conversation.Block{Type: conversation.BlockImage, MimeType: "image/jpeg", ImageURL: "https://example.com/x.jpg"},
			"[image: image/jpeg @ https://example.com/x.jpg]",
		},
		{
			"inline bytes",
			conversation.Block{Type: conversation.BlockImage, MimeType: "image/png", ImageData: []byte{1, 2, 3}},
			"[image: image/png, 3 bytes]",
		},
		{
			"missing mime falls back",
			conversation.Block{Type: conversation.BlockImage, ImageData: []byte{1}},
			"[image: image, 1 bytes]",
		},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := renderImagePlaceholder(tc.blk); got != tc.want {
				t.Errorf("renderImagePlaceholder() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestTruncateArgs(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		in   string
		max  int
		want string
	}{
		{"short passes through", `{"a":1}`, 100, `{"a":1}`},
		{"newlines flattened", "{\n\"a\": 1\n}", 100, `{ "a": 1 }`},
		{"long truncated with ellipsis", strings.Repeat("a", 20), 10, strings.Repeat("a", 10) + "..."},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := truncateArgs(tc.in, tc.max); got != tc.want {
				t.Errorf("truncateArgs(%q, %d) = %q, want %q", tc.in, tc.max, got, tc.want)
			}
		})
	}
}

func TestRenderUserInputPreview(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		blocks []conversation.Block
		want   string
	}{
		{
			"empty text blocks skipped",
			[]conversation.Block{
				{Type: conversation.BlockText, Text: ""},
				{Type: conversation.BlockText, Text: "hello"},
			},
			"hello",
		},
		{
			"image then text",
			[]conversation.Block{
				{Type: conversation.BlockImage, MimeType: "image/png", ImageData: []byte{1}},
				{Type: conversation.BlockText, Text: "caption"},
			},
			"[image: image/png, 1 bytes] caption",
		},
		{
			"text then image",
			[]conversation.Block{
				{Type: conversation.BlockText, Text: "look at"},
				{Type: conversation.BlockImage, MimeType: "image/png", ImageData: []byte{1, 2}},
			},
			"look at [image: image/png, 2 bytes]",
		},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := renderUserInputPreview(tc.blocks); got != tc.want {
				t.Errorf("renderUserInputPreview() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestPadBetween_EdgeCases(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		left, right string
		width       int
		want        string
	}{
		{"empty right returns left", "status", "", 80, "status"},
		{"empty left returns right", "", "model", 80, "model"},
		{"no room drops right entirely", "0123456789", "model-name", 11, "0123456789"},
		{"tight budget truncates right with ellipsis", "left", "abcdefghij", 10, "left abcd…"},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := padBetween(tc.left, tc.right, tc.width); got != tc.want {
				t.Errorf("padBetween(%q, %q, %d) = %q, want %q", tc.left, tc.right, tc.width, got, tc.want)
			}
		})
	}
}

func TestFormatRelative_FutureClampsToJustNow(t *testing.T) {
	t.Parallel()
	// A slightly-future timestamp (clock skew) must clamp to "just now",
	// not render a negative duration.
	got := formatRelative(futureTime())
	if got != "just now" {
		t.Errorf("formatRelative(future) = %q, want %q", got, "just now")
	}
}
