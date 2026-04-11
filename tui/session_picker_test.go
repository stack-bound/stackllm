package tui

import (
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

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

// seedSessionPicker puts the model into stateSessionPicker with two
// sessions: the currently-loaded one and a second "other" row. The
// cursor is parked on the "other" row so 'd' targets that entry. Both
// sessions are written to the store so a successful confirm-then-delete
// can observe the store change.
func seedSessionPicker(t *testing.T, store *fullFakeStore) (*Model, *session.Session) {
	t.Helper()
	m := testModel(t, store)
	m.session.Name = "current"
	if err := store.Save(nil, m.session); err != nil {
		t.Fatalf("save current: %v", err)
	}
	other := &session.Session{ID: "other-id", Name: "target row"}
	if err := store.Save(nil, other); err != nil {
		t.Fatalf("save other: %v", err)
	}
	m.state = stateSessionPicker
	m.sessions = []*session.Session{m.session, other}
	m.sessionCursor = 1
	return m, other
}

func TestSessionPicker_PressDOpensConfirmModal(t *testing.T) {
	t.Parallel()
	store := newFullFakeStore()
	m, target := seedSessionPicker(t, store)

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'d'}})
	m = updated.(*Model)

	if m.state != stateConfirmModal {
		t.Fatalf("expected stateConfirmModal, got %v", m.state)
	}
	if !strings.Contains(m.confirmTitle, "Delete") {
		t.Errorf("expected 'Delete' in title, got %q", m.confirmTitle)
	}
	if !strings.Contains(m.confirmPrompt, target.Name) {
		t.Errorf("expected target session name %q in prompt, got %q", target.Name, m.confirmPrompt)
	}
	if m.confirmReturnState != stateSessionPicker {
		t.Errorf("expected return state stateSessionPicker, got %v", m.confirmReturnState)
	}
	// Pre-confirm, the session must still exist in the store.
	if _, err := store.Load(nil, target.ID); err != nil {
		t.Errorf("target session should still exist before confirm, got err %v", err)
	}
}

func TestSessionPicker_CancelReturnsToPicker(t *testing.T) {
	t.Parallel()
	store := newFullFakeStore()
	m, target := seedSessionPicker(t, store)

	// Open the confirm modal.
	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'d'}})
	m = updated.(*Model)

	// 'n' should cancel and return to the picker, not drop to idle.
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'n'}})
	m = updated.(*Model)

	if m.state != stateSessionPicker {
		t.Errorf("expected stateSessionPicker after cancel, got %v", m.state)
	}
	if _, err := store.Load(nil, target.ID); err != nil {
		t.Errorf("target session should still exist after cancel, got err %v", err)
	}
	if len(m.sessions) != 2 {
		t.Errorf("expected picker to still show both sessions after cancel, got %d", len(m.sessions))
	}
}

func TestSessionPicker_EscCancelReturnsToPicker(t *testing.T) {
	t.Parallel()
	store := newFullFakeStore()
	m, target := seedSessionPicker(t, store)

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'d'}})
	m = updated.(*Model)

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = updated.(*Model)

	if m.state != stateSessionPicker {
		t.Errorf("expected stateSessionPicker after Esc, got %v", m.state)
	}
	if _, err := store.Load(nil, target.ID); err != nil {
		t.Errorf("target session should still exist after Esc, got err %v", err)
	}
}

func TestSessionPicker_ConfirmDeletesTargetAndStaysInPicker(t *testing.T) {
	t.Parallel()
	store := newFullFakeStore()
	m, target := seedSessionPicker(t, store)

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'d'}})
	m = updated.(*Model)

	if m.confirmAction == nil {
		t.Fatal("expected confirmAction to be set after opening confirm modal")
	}

	// Invoke the stored action directly so we can synchronously drain
	// the resulting delete Cmd, then dispatch its message through
	// Update exactly as the tea runtime would.
	actionCmd := m.confirmAction()
	if actionCmd == nil {
		t.Fatal("expected confirm action to return a tea.Cmd for the delete")
	}
	msg := actionCmd()
	if _, ok := msg.(sessionDeletedMsg); !ok {
		t.Fatalf("expected sessionDeletedMsg, got %T", msg)
	}
	// Close the confirm modal now — mirrors what confirmYes() would
	// do after dispatching the action.
	m.closeConfirmModal()
	if m.state != stateSessionPicker {
		t.Errorf("expected to return to stateSessionPicker after confirm close, got %v", m.state)
	}

	updated, _ = m.Update(msg)
	m = updated.(*Model)

	if _, err := store.Load(nil, target.ID); err == nil {
		t.Error("expected target session to be deleted from store after confirm")
	}
	if len(m.sessions) != 1 {
		t.Fatalf("expected 1 session remaining in picker, got %d", len(m.sessions))
	}
	if m.sessions[0].ID != m.session.ID {
		t.Errorf("expected only the current session remaining, got %q", m.sessions[0].ID)
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
