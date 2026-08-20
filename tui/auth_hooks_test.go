package tui

import (
	"context"
	"errors"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/stack-bound/stackllm/conversation"
)

// captureStdout redirects os.Stdout while fn runs and returns whatever
// was printed. AuthHooks writes directly to stdout, so this is the only
// way to observe its behaviour. Tests using it must not run in parallel
// because os.Stdout is process-global.
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	os.Stdout = w
	defer func() { os.Stdout = old }()

	fn()

	if err := w.Close(); err != nil {
		t.Fatalf("close pipe: %v", err)
	}
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("read pipe: %v", err)
	}
	return string(out)
}

// Intentionally not parallel: captureStdout swaps the process-global
// os.Stdout.
func TestAuthHooks_Behaviour(t *testing.T) {
	hooks := AuthHooks()
	ctx := context.Background()

	t.Run("OnToken prints the delta", func(t *testing.T) {
		out := captureStdout(t, func() {
			hooks.OnToken(ctx, "streamed-delta")
		})
		if out != "streamed-delta" {
			t.Errorf("OnToken output = %q, want %q", out, "streamed-delta")
		}
	})

	t.Run("OnToolCall prints the tool name", func(t *testing.T) {
		out := captureStdout(t, func() {
			hooks.OnToolCall(ctx, conversation.ToolCall{ID: "c1", Name: "read_file"})
		})
		if !strings.Contains(out, "⚡ read_file") {
			t.Errorf("OnToolCall output = %q, want it to contain %q", out, "⚡ read_file")
		}
	})

	t.Run("OnToolResult success prints the result", func(t *testing.T) {
		out := captureStdout(t, func() {
			hooks.OnToolResult(ctx, conversation.ToolCall{ID: "c1", Name: "read_file"}, "file body", nil)
		})
		if !strings.Contains(out, "→ file body") {
			t.Errorf("OnToolResult output = %q, want it to contain %q", out, "→ file body")
		}
	})

	t.Run("OnToolResult error prints the error", func(t *testing.T) {
		out := captureStdout(t, func() {
			hooks.OnToolResult(ctx, conversation.ToolCall{ID: "c1", Name: "read_file"}, "", errors.New("permission denied"))
		})
		if !strings.Contains(out, "✗ permission denied") {
			t.Errorf("OnToolResult error output = %q, want it to contain %q", out, "✗ permission denied")
		}
	})

	t.Run("AfterComplete prints a separator newline", func(t *testing.T) {
		out := captureStdout(t, func() {
			hooks.AfterComplete(ctx, nil)
		})
		if out != "\n" {
			t.Errorf("AfterComplete output = %q, want a single newline", out)
		}
	})

	t.Run("long tool results are truncated", func(t *testing.T) {
		long := strings.Repeat("x", 300)
		out := captureStdout(t, func() {
			hooks.OnToolResult(ctx, conversation.ToolCall{ID: "c1", Name: "read_file"}, long, nil)
		})
		if strings.Contains(out, long) {
			t.Error("expected 300-char result to be truncated to 200")
		}
		if !strings.Contains(out, strings.Repeat("x", 200)+"...") {
			t.Errorf("expected 200-char prefix with ellipsis, got %q", out)
		}
	})
}
