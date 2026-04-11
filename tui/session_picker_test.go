package tui

import (
	"strings"
	"testing"
	"time"

	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/session"
)

func TestFormatRelative(t *testing.T) {
	t.Parallel()
	now := time.Now()
	tests := []struct {
		name  string
		input time.Time
		want  string
	}{
		{"zero", time.Time{}, "—"},
		{"just now", now.Add(-5 * time.Second), "just now"},
		{"minutes", now.Add(-2 * time.Minute), "2m ago"},
		{"hours", now.Add(-3 * time.Hour), "3h ago"},
		{"yesterday", now.Add(-30 * time.Hour), "yesterday"},
		{"days", now.Add(-4 * 24 * time.Hour), "4d ago"},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := formatRelative(tt.input)
			if got != tt.want {
				t.Errorf("formatRelative(%v) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestFormatRelative_OldFallsBackToDate(t *testing.T) {
	t.Parallel()
	old := time.Now().Add(-30 * 24 * time.Hour)
	got := formatRelative(old)
	// Must not be a relative string; expect "Jan 2"-style format.
	if strings.HasSuffix(got, "ago") || got == "just now" || got == "yesterday" {
		t.Errorf("expected date fallback for old time, got %q", got)
	}
}

func TestTruncateLine(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name  string
		input string
		max   int
		want  string
	}{
		{"short", "hello", 20, "hello"},
		{"exact", "hello", 5, "hello"},
		{"truncate", "this is a long line of text that will not fit", 10, "this is a…"},
		{"newlines collapsed", "one\ntwo\nthree", 20, "one two three"},
		{"runs collapsed", "a   b   c", 20, "a b c"},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := truncateLine(tt.input, tt.max)
			if got != tt.want {
				t.Errorf("truncateLine(%q, %d) = %q, want %q", tt.input, tt.max, got, tt.want)
			}
		})
	}
}

func TestRenderSessionPicker_Basic(t *testing.T) {
	t.Parallel()
	m := newTestModel(t)
	m.width = 120
	m.height = 40
	m.state = stateSessionPicker

	now := time.Now()
	currentID := m.session.ID
	m.sessions = []*session.Session{
		{
			ID:      currentID,
			Name:    "My debugging run",
			Updated: now.Add(-2 * time.Minute),
			Messages: []conversation.Message{
				{Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hi"}}},
				{Role: conversation.RoleAssistant, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hello"}}},
			},
		},
		{
			ID:      "other-id",
			Name:    "second session",
			Updated: now.Add(-1 * time.Hour),
			Messages: []conversation.Message{
				{Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "q"}}},
			},
		},
	}
	m.sessionCursor = 0

	out := m.renderSessionPicker()
	if !strings.Contains(out, "Sessions (2)") {
		t.Errorf("expected header with session count, got:\n%s", out)
	}
	if !strings.Contains(out, "My debugging run") {
		t.Errorf("expected first session name, got:\n%s", out)
	}
	if !strings.Contains(out, "second session") {
		t.Errorf("expected second session name, got:\n%s", out)
	}
	if !strings.Contains(out, "2 msgs") {
		t.Errorf("expected message count for first session, got:\n%s", out)
	}
	if !strings.Contains(out, "1 msgs") {
		t.Errorf("expected message count for second session, got:\n%s", out)
	}
	if !strings.Contains(out, "●") {
		t.Errorf("expected current-session marker ●, got:\n%s", out)
	}
	if !strings.Contains(out, "↑↓ move") {
		t.Errorf("expected footer hint, got:\n%s", out)
	}
}

func TestRenderSessionPicker_Empty(t *testing.T) {
	t.Parallel()
	m := newTestModel(t)
	m.sessions = nil
	out := m.renderSessionPicker()
	if !strings.Contains(out, "no sessions") {
		t.Errorf("expected 'no sessions' placeholder, got:\n%s", out)
	}
}

func TestRenderForkPicker_Basic(t *testing.T) {
	t.Parallel()
	m := newTestModel(t)
	m.width = 120
	m.height = 40
	m.state = stateForkPicker
	m.session.Name = "Untitled"
	m.session.Messages = []conversation.Message{
		{ID: "m1", Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "first question"}}},
		{ID: "m2", Role: conversation.RoleAssistant, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "first answer"}}},
		{ID: "m3", Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "second question"}}},
	}
	m.forkCursor = 2

	out := m.renderForkPicker()
	if !strings.Contains(out, "Fork from which message") {
		t.Errorf("expected header, got:\n%s", out)
	}
	if !strings.Contains(out, "Untitled") {
		t.Errorf("expected session name in header, got:\n%s", out)
	}
	if !strings.Contains(out, "[1]") || !strings.Contains(out, "[2]") || !strings.Contains(out, "[3]") {
		t.Errorf("expected index markers, got:\n%s", out)
	}
	if !strings.Contains(out, "first question") {
		t.Errorf("expected first message preview, got:\n%s", out)
	}
	if !strings.Contains(out, "(leaf)") {
		t.Errorf("expected leaf marker on last message, got:\n%s", out)
	}
	if !strings.Contains(out, "↑↓ move") {
		t.Errorf("expected footer hint, got:\n%s", out)
	}
}

func TestDisplaySessionName(t *testing.T) {
	t.Parallel()
	s := &session.Session{ID: "abcdef0123456789", Name: "Hello"}
	if got := displaySessionName(s); got != "Hello" {
		t.Errorf("want Hello, got %q", got)
	}
	s.Name = ""
	if got := displaySessionName(s); got != "abcdef01" {
		t.Errorf("want first 8 of id, got %q", got)
	}
	short := &session.Session{ID: "abc", Name: ""}
	if got := displaySessionName(short); got != "abc" {
		t.Errorf("want abc, got %q", got)
	}
	if got := displaySessionName(nil); got != "" {
		t.Errorf("want empty, got %q", got)
	}
}

func TestWindowBounds(t *testing.T) {
	t.Parallel()
	tests := []struct {
		cursor, total, visible int
		wantStart, wantEnd     int
	}{
		{0, 5, 10, 0, 5},    // fits, no scroll
		{0, 20, 10, 0, 10},  // cursor at top
		{19, 20, 10, 10, 20}, // cursor at bottom
		{10, 20, 10, 5, 15},  // cursor in middle
	}
	for _, tt := range tests {
		gotStart, gotEnd := windowBounds(tt.cursor, tt.total, tt.visible)
		if gotStart != tt.wantStart || gotEnd != tt.wantEnd {
			t.Errorf("windowBounds(%d, %d, %d) = (%d, %d), want (%d, %d)",
				tt.cursor, tt.total, tt.visible, gotStart, gotEnd, tt.wantStart, tt.wantEnd)
		}
	}
}
