package tui

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/stack-bound/stackllm/agent"
	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/provider"
	"github.com/stack-bound/stackllm/session"

	tea "github.com/charmbracelet/bubbletea"
)

func TestFirstTextPreview(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		msg  conversation.Message
		want string
	}{
		{
			"text block wins",
			conversation.Message{Blocks: []conversation.Block{
				{Type: conversation.BlockText, Text: "plain answer"},
			}},
			"plain answer",
		},
		{
			"thinking-only turn falls back to thinking",
			conversation.Message{Blocks: []conversation.Block{
				{Type: conversation.BlockThinking, Text: "internal reasoning"},
			}},
			"internal reasoning",
		},
		{
			"text preferred over thinking",
			conversation.Message{Blocks: []conversation.Block{
				{Type: conversation.BlockThinking, Text: "internal"},
				{Type: conversation.BlockText, Text: "visible"},
			}},
			"visible",
		},
		{
			"tool use renders compact call",
			conversation.Message{Blocks: []conversation.Block{
				{Type: conversation.BlockToolUse, ToolName: "read_file", ToolArgsJSON: `{"path":"/tmp"}`},
			}},
			`⚡ read_file({"path":"/tmp"})`,
		},
		{
			"tool result renders its text",
			conversation.Message{Blocks: []conversation.Block{
				{Type: conversation.BlockToolResult, ToolCallID: "c1", Text: "result body"},
			}},
			"result body",
		},
		{
			"image renders placeholder",
			conversation.Message{Blocks: []conversation.Block{
				{Type: conversation.BlockImage, MimeType: "image/png", ImageData: []byte{1, 2}},
			}},
			"[image: image/png]",
		},
		{
			"empty message",
			conversation.Message{},
			"",
		},
		{
			"whitespace-only text falls through",
			conversation.Message{Blocks: []conversation.Block{
				{Type: conversation.BlockText, Text: "   "},
				{Type: conversation.BlockToolResult, ToolCallID: "c1", Text: "tool output"},
			}},
			"tool output",
		},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := firstTextPreview(tc.msg); got != tc.want {
				t.Errorf("firstTextPreview() = %q, want %q", got, tc.want)
			}
		})
	}
}

// metaListStore returns metadata-only rows from List (Messages
// stripped), matching how SessionStore.List behaves for real stores and
// forcing loadSessions' hydration pass through Load.
type metaListStore struct{ *fullFakeStore }

func (s *metaListStore) List(ctx context.Context) ([]*session.Session, error) {
	list, err := s.fullFakeStore.List(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]*session.Session, 0, len(list))
	for _, sess := range list {
		meta := *sess
		meta.Messages = nil
		out = append(out, &meta)
	}
	return out, nil
}

func TestLoadSessions_FlushesAndHydrates(t *testing.T) {
	t.Parallel()

	inner := newFullFakeStore()
	store := &metaListStore{fullFakeStore: inner}
	p := provider.New(provider.OllamaConfig("http://localhost", "test"))
	m := New(agent.New(p), store)

	// A previously saved session with two messages.
	other := session.New()
	other.Name = "older"
	other.AppendMessage(conversation.Message{Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "q"}}})
	other.AppendMessage(conversation.Message{Role: conversation.RoleAssistant, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "a"}}})
	if err := inner.Save(context.Background(), other); err != nil {
		t.Fatalf("save other: %v", err)
	}

	// The current in-memory session has one message and has never been
	// saved — loadSessions must flush it before listing.
	m.session.AppendMessage(conversation.Message{Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "fresh"}}})

	msg := m.loadSessions()()
	loaded, ok := msg.(sessionsLoadedMsg)
	if !ok {
		t.Fatalf("expected sessionsLoadedMsg, got %T", msg)
	}
	if loaded.err != nil {
		t.Fatalf("unexpected err: %v", loaded.err)
	}
	if len(loaded.sessions) != 2 {
		t.Fatalf("sessions = %d, want 2 (current flushed + saved)", len(loaded.sessions))
	}
	counts := map[string]int{}
	for _, s := range loaded.sessions {
		counts[s.ID] = len(s.Messages)
	}
	if counts[other.ID] != 2 {
		t.Errorf("saved session should be hydrated to 2 msgs, got %d", counts[other.ID])
	}
	if counts[m.session.ID] != 1 {
		t.Errorf("current session should be flushed and hydrated to 1 msg, got %d", counts[m.session.ID])
	}
}

// failingLoadStore lists rows but errors on Load, exercising the
// hydration pass's best-effort continue branch.
type failingLoadStore struct{ *metaListStore }

func (f *failingLoadStore) Load(_ context.Context, id string) (*session.Session, error) {
	return nil, fmt.Errorf("row corrupt: %s", id)
}

func TestLoadSessions_SaveFailurePropagates(t *testing.T) {
	t.Parallel()
	m := testModel(t, &failingSaveStore{fullFakeStore: newFullFakeStore()})
	msg := m.loadSessions()()
	loaded := msg.(sessionsLoadedMsg)
	if loaded.err == nil || !strings.Contains(loaded.err.Error(), "disk full") {
		t.Errorf("expected flush error, got %v", loaded.err)
	}
}

func TestLoadSessions_HydrationFailureIsBestEffort(t *testing.T) {
	t.Parallel()
	inner := newFullFakeStore()
	store := &failingLoadStore{metaListStore: &metaListStore{fullFakeStore: inner}}
	p := provider.New(provider.OllamaConfig("http://localhost", "test"))
	m := New(agent.New(p), store)

	msg := m.loadSessions()()
	loaded := msg.(sessionsLoadedMsg)
	if loaded.err != nil {
		t.Fatalf("hydration failure must not fail the list: %v", loaded.err)
	}
	// The flushed current session is still listed, just with no
	// hydrated messages.
	if len(loaded.sessions) != 1 || loaded.sessions[0].ID != m.session.ID {
		t.Fatalf("expected the flushed session listed, got %+v", loaded.sessions)
	}
	if len(loaded.sessions[0].Messages) != 0 {
		t.Errorf("expected zero messages when hydration fails, got %d", len(loaded.sessions[0].Messages))
	}
}

func TestOpenForkPicker_EmptySessionFailsSoftly(t *testing.T) {
	t.Parallel()
	m := testModel(t, newFullFakeStore())
	if cmd := m.openForkPicker(); cmd != nil {
		t.Error("expected nil cmd for empty session")
	}
	if m.state != stateIdle {
		t.Errorf("expected stateIdle, got %v", m.state)
	}
	if !strings.Contains(m.output.String(), "no messages yet") {
		t.Errorf("expected empty-session error, got:\n%s", m.output.String())
	}
}

func TestLoadSessions_NilStore(t *testing.T) {
	t.Parallel()
	p := provider.New(provider.OllamaConfig("http://localhost", "test"))
	m := New(agent.New(p), nil)
	msg := m.loadSessions()()
	loaded := msg.(sessionsLoadedMsg)
	if loaded.err == nil || !strings.Contains(loaded.err.Error(), "no session store configured") {
		t.Errorf("expected no-store error, got %v", loaded.err)
	}
}

func TestOpenSessionPicker(t *testing.T) {
	t.Parallel()

	t.Run("nil store fails inline", func(t *testing.T) {
		t.Parallel()
		p := provider.New(provider.OllamaConfig("http://localhost", "test"))
		m := New(agent.New(p), nil)
		if cmd := m.openSessionPicker(); cmd != nil {
			t.Error("expected nil cmd with no store")
		}
		if !strings.Contains(m.output.String(), "no session store configured") {
			t.Errorf("expected inline error, got:\n%s", m.output.String())
		}
		if m.state != stateIdle {
			t.Errorf("expected stateIdle, got %v", m.state)
		}
	})

	t.Run("with store starts loading", func(t *testing.T) {
		t.Parallel()
		m := testModel(t, newFullFakeStore())
		cmd := m.openSessionPicker()
		if cmd == nil {
			t.Fatal("expected load cmd")
		}
		if m.state != stateSessionLoading {
			t.Errorf("expected stateSessionLoading, got %v", m.state)
		}
		if _, ok := cmd().(sessionsLoadedMsg); !ok {
			t.Error("expected the cmd to produce a sessionsLoadedMsg")
		}
	})
}

func TestLoadSession(t *testing.T) {
	t.Parallel()

	t.Run("loads existing session", func(t *testing.T) {
		t.Parallel()
		store := newFullFakeStore()
		m := testModel(t, store)
		other := session.New()
		other.Name = "wanted"
		other.AppendMessage(conversation.Message{Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hello"}}})
		store.Save(context.Background(), other)

		msg := m.loadSession(other.ID)()
		loaded, ok := msg.(sessionLoadedMsg)
		if !ok {
			t.Fatalf("expected sessionLoadedMsg, got %T", msg)
		}
		if loaded.err != nil {
			t.Fatalf("unexpected err: %v", loaded.err)
		}
		if loaded.session.ID != other.ID || len(loaded.session.Messages) != 1 {
			t.Errorf("loaded wrong session: %+v", loaded.session)
		}
	})

	t.Run("missing id errors", func(t *testing.T) {
		t.Parallel()
		m := testModel(t, newFullFakeStore())
		msg := m.loadSession("nope")()
		loaded := msg.(sessionLoadedMsg)
		if loaded.err == nil {
			t.Error("expected error for unknown session id")
		}
	})

	t.Run("nil store errors", func(t *testing.T) {
		t.Parallel()
		p := provider.New(provider.OllamaConfig("http://localhost", "test"))
		m := New(agent.New(p), nil)
		msg := m.loadSession("any")()
		loaded := msg.(sessionLoadedMsg)
		if loaded.err == nil || !strings.Contains(loaded.err.Error(), "no session store configured") {
			t.Errorf("expected no-store error, got %v", loaded.err)
		}
	})
}

func TestForkAt(t *testing.T) {
	t.Parallel()

	t.Run("forks and reports index", func(t *testing.T) {
		t.Parallel()
		store := newFullFakeStore()
		m := testModel(t, store)
		m.session.AppendMessage(conversation.Message{Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "one"}}})
		m.session.AppendMessage(conversation.Message{Role: conversation.RoleAssistant, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "two"}}})
		atID := m.session.Messages[0].ID

		msg := m.forkAt(atID, 1)()
		forked, ok := msg.(sessionForkedMsg)
		if !ok {
			t.Fatalf("expected sessionForkedMsg, got %T", msg)
		}
		if forked.err != nil {
			t.Fatalf("unexpected err: %v", forked.err)
		}
		if forked.atIndex != 1 {
			t.Errorf("atIndex = %d, want 1", forked.atIndex)
		}
		if store.forkSrcID != m.session.ID || store.forkMsgID != atID {
			t.Errorf("fork called with (%q, %q), want (%q, %q)", store.forkSrcID, store.forkMsgID, m.session.ID, atID)
		}
		if forked.session.ID == m.session.ID {
			t.Error("fork must produce a session with a fresh ID")
		}
		if len(forked.session.Messages) != 1 {
			t.Errorf("fork should copy 1 message, got %d", len(forked.session.Messages))
		}
		// The source session must have been flushed before forking.
		if _, err := store.Load(context.Background(), m.session.ID); err != nil {
			t.Errorf("source session should be saved before fork: %v", err)
		}
	})

	t.Run("no forker errors", func(t *testing.T) {
		t.Parallel()
		m := testModel(t, &minimalStore{inner: newFullFakeStore()})
		msg := m.forkAt("m1", 1)()
		forked := msg.(sessionForkedMsg)
		if forked.err == nil || !strings.Contains(forked.err.Error(), "does not support branching") {
			t.Errorf("expected capability error, got %v", forked.err)
		}
	})

	t.Run("save failure aborts fork", func(t *testing.T) {
		t.Parallel()
		m := testModel(t, &failingSaveStore{fullFakeStore: newFullFakeStore()})
		msg := m.forkAt("m1", 1)()
		forked := msg.(sessionForkedMsg)
		if forked.err == nil || !strings.Contains(forked.err.Error(), "disk full") {
			t.Errorf("expected save error, got %v", forked.err)
		}
	})
}

func TestDeleteSession_ErrorPaths(t *testing.T) {
	t.Parallel()

	t.Run("nil store", func(t *testing.T) {
		t.Parallel()
		p := provider.New(provider.OllamaConfig("http://localhost", "test"))
		m := New(agent.New(p), nil)
		msg := m.deleteSession("x")()
		del := msg.(sessionDeletedMsg)
		if del.err == nil || !strings.Contains(del.err.Error(), "no session store configured") {
			t.Errorf("expected no-store error, got %v", del.err)
		}
	})

	t.Run("store delete failure", func(t *testing.T) {
		t.Parallel()
		m := testModel(t, &failingDeleteStore{fullFakeStore: newFullFakeStore()})
		msg := m.deleteSession("x")()
		del := msg.(sessionDeletedMsg)
		if del.err == nil || !strings.Contains(del.err.Error(), "locked") {
			t.Errorf("expected delete error, got %v", del.err)
		}
	})
}

// failingDeleteStore errors on Delete to exercise the delete failure path.
type failingDeleteStore struct{ *fullFakeStore }

func (f *failingDeleteStore) Delete(_ context.Context, _ string) error {
	return fmt.Errorf("database locked")
}

func TestUpdate_SessionsLoadedMsg(t *testing.T) {
	t.Parallel()

	t.Run("cursor lands on current session", func(t *testing.T) {
		t.Parallel()
		m := testModel(t, newFullFakeStore())
		m.state = stateSessionLoading
		other1 := session.New()
		other2 := session.New()
		updated, _ := m.Update(sessionsLoadedMsg{sessions: []*session.Session{other1, m.session, other2}})
		m = updated.(*Model)
		if m.state != stateSessionPicker {
			t.Errorf("expected stateSessionPicker, got %v", m.state)
		}
		if m.sessionCursor != 1 {
			t.Errorf("cursor = %d, want 1 (current session's row)", m.sessionCursor)
		}
	})

	t.Run("empty list returns to idle", func(t *testing.T) {
		t.Parallel()
		m := testModel(t, newFullFakeStore())
		m.state = stateSessionLoading
		updated, _ := m.Update(sessionsLoadedMsg{})
		m = updated.(*Model)
		if m.state != stateIdle {
			t.Errorf("expected stateIdle, got %v", m.state)
		}
		if !strings.Contains(m.output.String(), "No saved sessions yet.") {
			t.Errorf("expected empty hint, got:\n%s", m.output.String())
		}
	})

	t.Run("error surfaces inline", func(t *testing.T) {
		t.Parallel()
		m := testModel(t, newFullFakeStore())
		m.state = stateSessionLoading
		updated, _ := m.Update(sessionsLoadedMsg{err: fmt.Errorf("boom")})
		m = updated.(*Model)
		if m.state != stateIdle {
			t.Errorf("expected stateIdle, got %v", m.state)
		}
		if !strings.Contains(m.output.String(), "✗ /sessions: boom") {
			t.Errorf("expected error line, got:\n%s", m.output.String())
		}
	})
}

func TestUpdate_SessionPickerEnter(t *testing.T) {
	t.Parallel()

	t.Run("enter on current session just closes", func(t *testing.T) {
		t.Parallel()
		m := testModel(t, newFullFakeStore())
		m.state = stateSessionPicker
		m.sessions = []*session.Session{m.session}
		m.sessionCursor = 0
		updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
		m = updated.(*Model)
		if m.state != stateIdle {
			t.Errorf("expected stateIdle, got %v", m.state)
		}
		if m.sessions != nil {
			t.Errorf("expected sessions cleared, got %d", len(m.sessions))
		}
	})

	t.Run("enter on other session loads it", func(t *testing.T) {
		t.Parallel()
		store := newFullFakeStore()
		m := testModel(t, store)
		other := session.New()
		other.Name = "target"
		other.AppendMessage(conversation.Message{Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "loaded text"}}})
		store.Save(context.Background(), other)

		m.state = stateSessionPicker
		m.sessions = []*session.Session{m.session, other}
		m.sessionCursor = 1

		updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
		m = updated.(*Model)
		if m.state != stateSessionLoading {
			t.Fatalf("expected stateSessionLoading, got %v", m.state)
		}
		if cmd == nil {
			t.Fatal("expected load cmd")
		}

		var loaded *sessionLoadedMsg
		for _, msg := range collectMsgs(cmd) {
			if lm, ok := msg.(sessionLoadedMsg); ok {
				loaded = &lm
			}
		}
		if loaded == nil {
			t.Fatal("expected sessionLoadedMsg from Enter cmd")
		}
		updated, _ = m.Update(*loaded)
		m = updated.(*Model)
		if m.session.ID != other.ID {
			t.Errorf("expected the target session swapped in, got %q", m.session.ID)
		}
		if m.state != stateIdle {
			t.Errorf("expected stateIdle after load, got %v", m.state)
		}
		out := m.output.String()
		if !strings.Contains(out, `loaded session "target"`) {
			t.Errorf("expected load feedback, got:\n%s", out)
		}
		if !strings.Contains(out, "loaded text") {
			t.Errorf("expected conversation re-rendered into viewport, got:\n%s", out)
		}
	})
}

func TestUpdate_SessionLoadedMsgError(t *testing.T) {
	t.Parallel()
	m := testModel(t, newFullFakeStore())
	m.state = stateSessionLoading
	updated, _ := m.Update(sessionLoadedMsg{err: fmt.Errorf("gone")})
	m = updated.(*Model)
	if m.state != stateIdle {
		t.Errorf("expected stateIdle, got %v", m.state)
	}
	if !strings.Contains(m.output.String(), "✗ /sessions: gone") {
		t.Errorf("expected error line, got:\n%s", m.output.String())
	}
}

func TestUpdate_ForkPickerEnterForksSession(t *testing.T) {
	t.Parallel()
	store := newFullFakeStore()
	m := testModel(t, store)
	m.session.AppendMessage(conversation.Message{Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "one"}}})
	m.session.AppendMessage(conversation.Message{Role: conversation.RoleAssistant, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "two"}}})
	srcID := m.session.ID

	m.state = stateForkPicker
	m.forkCursor = 0 // fork at the first message

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(*Model)
	if m.state != stateSessionLoading {
		t.Fatalf("expected stateSessionLoading, got %v", m.state)
	}
	if cmd == nil {
		t.Fatal("expected fork cmd")
	}

	var forked *sessionForkedMsg
	for _, msg := range collectMsgs(cmd) {
		if fm, ok := msg.(sessionForkedMsg); ok {
			forked = &fm
		}
	}
	if forked == nil {
		t.Fatal("expected sessionForkedMsg from Enter cmd")
	}
	if store.forkSrcID != srcID {
		t.Errorf("fork src = %q, want %q", store.forkSrcID, srcID)
	}

	updated, _ = m.Update(*forked)
	m = updated.(*Model)
	if m.session.ID == srcID {
		t.Error("expected the forked session swapped in")
	}
	if m.state != stateIdle {
		t.Errorf("expected stateIdle, got %v", m.state)
	}
	if !strings.Contains(m.output.String(), "forked from message [1]") {
		t.Errorf("expected fork feedback, got:\n%s", m.output.String())
	}
}

func TestUpdate_SessionForkedMsgError(t *testing.T) {
	t.Parallel()
	m := testModel(t, newFullFakeStore())
	m.state = stateSessionLoading
	updated, _ := m.Update(sessionForkedMsg{err: fmt.Errorf("cannot fork")})
	m = updated.(*Model)
	if m.state != stateIdle {
		t.Errorf("expected stateIdle, got %v", m.state)
	}
	if !strings.Contains(m.output.String(), "✗ /fork: cannot fork") {
		t.Errorf("expected error line, got:\n%s", m.output.String())
	}
}

func TestUpdate_SessionDeletedMsg(t *testing.T) {
	t.Parallel()

	t.Run("error surfaces inline", func(t *testing.T) {
		t.Parallel()
		m := testModel(t, newFullFakeStore())
		updated, _ := m.Update(sessionDeletedMsg{deletedID: "x", err: fmt.Errorf("locked")})
		m = updated.(*Model)
		if !strings.Contains(m.output.String(), "✗ /sessions: locked") {
			t.Errorf("expected error line, got:\n%s", m.output.String())
		}
	})

	t.Run("deleting the loaded session resets it", func(t *testing.T) {
		t.Parallel()
		m := testModel(t, newFullFakeStore())
		m.state = stateSessionPicker
		oldID := m.session.ID
		m.sessions = []*session.Session{m.session}
		m.sessionCursor = 0

		updated, _ := m.Update(sessionDeletedMsg{deletedID: oldID, deletedSelf: true})
		m = updated.(*Model)
		if m.session.ID == oldID {
			t.Error("expected a fresh session after deleting the loaded one")
		}
		if m.state != stateIdle {
			t.Errorf("expected stateIdle when the picker emptied, got %v", m.state)
		}
		if !strings.Contains(m.output.String(), "no saved sessions remain") {
			t.Errorf("expected empty-picker feedback, got:\n%s", m.output.String())
		}
	})

	t.Run("cursor clamps after deleting last row", func(t *testing.T) {
		t.Parallel()
		m := testModel(t, newFullFakeStore())
		m.state = stateSessionPicker
		other := session.New()
		m.sessions = []*session.Session{m.session, other}
		m.sessionCursor = 1

		updated, _ := m.Update(sessionDeletedMsg{deletedID: other.ID})
		m = updated.(*Model)
		if len(m.sessions) != 1 {
			t.Fatalf("expected 1 row after delete, got %d", len(m.sessions))
		}
		if m.sessionCursor != 0 {
			t.Errorf("cursor = %d, want 0", m.sessionCursor)
		}
		if !strings.Contains(m.output.String(), "✓ session deleted") {
			t.Errorf("expected delete feedback, got:\n%s", m.output.String())
		}
	})
}
