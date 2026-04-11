package tui

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/stack-bound/stackllm/agent"
	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/provider"
	"github.com/stack-bound/stackllm/session"
)

// fullFakeStore implements SessionStore + SessionForker + SessionExporter
// so the TUI caches both capability interfaces in New() and every new
// command path can be exercised.
type fullFakeStore struct {
	mu       sync.Mutex
	sessions map[string]*session.Session
	// forkCall / exportCall record the last invocation so tests can
	// assert the TUI passed through the right session ID.
	forkSrcID   string
	forkMsgID   string
	forkNewName string
	exportID    string
}

func newFullFakeStore() *fullFakeStore {
	return &fullFakeStore{sessions: make(map[string]*session.Session)}
}

func (f *fullFakeStore) Save(_ context.Context, s *session.Session) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.sessions[s.ID] = s
	return nil
}

func (f *fullFakeStore) Load(_ context.Context, id string) (*session.Session, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	s, ok := f.sessions[id]
	if !ok {
		return nil, fmt.Errorf("not found: %s", id)
	}
	return s, nil
}

func (f *fullFakeStore) Delete(_ context.Context, id string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	delete(f.sessions, id)
	return nil
}

func (f *fullFakeStore) List(_ context.Context) ([]*session.Session, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]*session.Session, 0, len(f.sessions))
	for _, s := range f.sessions {
		out = append(out, s)
	}
	return out, nil
}

func (f *fullFakeStore) Fork(_ context.Context, srcSessionID, atMessageID, newName string) (*session.Session, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.forkSrcID = srcSessionID
	f.forkMsgID = atMessageID
	f.forkNewName = newName
	src, ok := f.sessions[srcSessionID]
	if !ok {
		return nil, fmt.Errorf("not found: %s", srcSessionID)
	}
	// Copy the chain up to and including the requested message,
	// mirroring SQLiteStore.Fork's semantics well enough for the
	// test to observe a fork result with a fresh ID.
	fork := session.New()
	fork.Name = newName
	for _, msg := range src.Messages {
		fork.Messages = append(fork.Messages, msg)
		if msg.ID == atMessageID {
			break
		}
	}
	f.sessions[fork.ID] = fork
	return fork, nil
}

func (f *fullFakeStore) ExportJSONL(_ context.Context, sessionID string, w io.Writer) error {
	f.mu.Lock()
	f.exportID = sessionID
	_, ok := f.sessions[sessionID]
	f.mu.Unlock()
	if !ok {
		return fmt.Errorf("not found: %s", sessionID)
	}
	// Cheap JSONL stand-in: one line per message with the role.
	s := f.sessions[sessionID]
	for _, msg := range s.Messages {
		if _, err := fmt.Fprintf(w, "{\"role\":%q}\n", msg.Role); err != nil {
			return err
		}
	}
	return nil
}

// minimalStore implements only SessionStore — not SessionForker or
// SessionExporter — so the TUI's capability gating branches can be
// exercised when the underlying store lacks branching/export support.
type minimalStore struct{ inner *fullFakeStore }

func (s *minimalStore) Save(ctx context.Context, sess *session.Session) error {
	return s.inner.Save(ctx, sess)
}
func (s *minimalStore) Load(ctx context.Context, id string) (*session.Session, error) {
	return s.inner.Load(ctx, id)
}
func (s *minimalStore) Delete(ctx context.Context, id string) error {
	return s.inner.Delete(ctx, id)
}
func (s *minimalStore) List(ctx context.Context) ([]*session.Session, error) {
	return s.inner.List(ctx)
}

// testModel constructs a *Model backed by the given store. The agent
// is wired to a dummy Ollama provider so its HTTP surface is never hit.
func testModel(t *testing.T, store session.SessionStore) *Model {
	t.Helper()
	p := provider.New(provider.OllamaConfig("http://localhost", "test"))
	a := agent.New(p)
	m := New(a, store)
	// Give the view some width so picker and status bar math works.
	m.width = 120
	m.height = 40
	return m
}

func TestNew_CachesCapabilityInterfaces(t *testing.T) {
	t.Parallel()
	full := newFullFakeStore()
	m := testModel(t, full)
	if m.forker == nil {
		t.Error("expected forker to be set when store implements SessionForker")
	}
	if m.exporter == nil {
		t.Error("expected exporter to be set when store implements SessionExporter")
	}

	mini := &minimalStore{inner: newFullFakeStore()}
	m2 := testModel(t, mini)
	if m2.forker != nil {
		t.Error("expected forker to be nil when store does not implement SessionForker")
	}
	if m2.exporter != nil {
		t.Error("expected exporter to be nil when store does not implement SessionExporter")
	}
}

func TestExecuteHelp_RendersAllCommands(t *testing.T) {
	t.Parallel()
	m := testModel(t, newFullFakeStore())
	m.executeHelp()
	out := m.output.String()
	for _, c := range commands {
		if !strings.Contains(out, c.Name) {
			t.Errorf("help output missing command %q:\n%s", c.Name, out)
		}
		if !strings.Contains(out, c.Description) {
			t.Errorf("help output missing description %q:\n%s", c.Description, out)
		}
	}
}

func TestExecuteDelete_ResetsSession(t *testing.T) {
	t.Parallel()
	store := newFullFakeStore()
	m := testModel(t, store)

	// Stash the current (new) session in the store so Delete finds
	// something to remove.
	m.session.Name = "to delete"
	store.Save(context.Background(), m.session)
	oldID := m.session.ID

	m.executeDelete()

	if _, err := store.Load(context.Background(), oldID); err == nil {
		t.Error("expected old session to be gone from store")
	}
	if m.session.ID == oldID {
		t.Error("expected TUI to swap in a fresh session after delete")
	}
	if !strings.Contains(m.output.String(), "✓ deleted") {
		t.Errorf("expected success line in output, got:\n%s", m.output.String())
	}
}

func TestSubmitRename_PersistsName(t *testing.T) {
	t.Parallel()
	store := newFullFakeStore()
	m := testModel(t, store)

	m.submitRename("My new name")

	if m.session.Name != "My new name" {
		t.Errorf("expected session name to be updated, got %q", m.session.Name)
	}
	saved, err := store.Load(context.Background(), m.session.ID)
	if err != nil {
		t.Fatalf("store.Load: %v", err)
	}
	if saved.Name != "My new name" {
		t.Errorf("expected store to persist the new name, got %q", saved.Name)
	}
	if !strings.Contains(m.output.String(), "✓ renamed") {
		t.Errorf("expected success line in output, got:\n%s", m.output.String())
	}
}

// failingSaveStore wraps fullFakeStore so Save always errors, letting
// us verify that submitRename rolls the in-memory name back when the
// persistence step fails.
type failingSaveStore struct{ *fullFakeStore }

func (f *failingSaveStore) Save(_ context.Context, _ *session.Session) error {
	return fmt.Errorf("disk full")
}

func TestSubmitRename_SaveFailureRollsBack(t *testing.T) {
	t.Parallel()
	m := testModel(t, &failingSaveStore{fullFakeStore: newFullFakeStore()})
	m.session.Name = "original"
	m.submitRename("new name")
	if m.session.Name != "original" {
		t.Errorf("expected name to roll back to 'original' on Save failure, got %q", m.session.Name)
	}
	if !strings.Contains(m.output.String(), "✗ /rename") {
		t.Errorf("expected rename error in output, got:\n%s", m.output.String())
	}
}

func TestSubmitRename_RejectsEmpty(t *testing.T) {
	t.Parallel()
	m := testModel(t, newFullFakeStore())
	m.session.Name = "original"
	m.submitRename("   ")
	if m.session.Name != "original" {
		t.Errorf("expected name to remain 'original', got %q", m.session.Name)
	}
	if !strings.Contains(m.output.String(), "cannot be empty") {
		t.Errorf("expected empty-name error, got:\n%s", m.output.String())
	}
}

func TestSubmitExport_WritesFile(t *testing.T) {
	t.Parallel()
	store := newFullFakeStore()
	m := testModel(t, store)

	m.session.Messages = []conversation.Message{
		{Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hi"}}},
		{Role: conversation.RoleAssistant, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hello"}}},
	}
	store.Save(context.Background(), m.session)

	dir := t.TempDir()
	path := filepath.Join(dir, "out.jsonl")
	m.submitExport(path)

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read exported file: %v", err)
	}
	if !strings.Contains(string(data), `"role":"user"`) {
		t.Errorf("expected user message in export, got:\n%s", data)
	}
	if !strings.Contains(string(data), `"role":"assistant"`) {
		t.Errorf("expected assistant message in export, got:\n%s", data)
	}
	if store.exportID != m.session.ID {
		t.Errorf("expected ExportJSONL to be called with session id %s, got %s", m.session.ID, store.exportID)
	}
	if !strings.Contains(m.output.String(), "✓ exported 2 messages") {
		t.Errorf("expected success line, got:\n%s", m.output.String())
	}
}

func TestSubmitExport_NoExporterFailsSoftly(t *testing.T) {
	t.Parallel()
	m := testModel(t, &minimalStore{inner: newFullFakeStore()})
	m.submitExport("/tmp/should-not-be-created.jsonl")
	if !strings.Contains(m.output.String(), "does not support export") {
		t.Errorf("expected capability-gated error, got:\n%s", m.output.String())
	}
	if _, err := os.Stat("/tmp/should-not-be-created.jsonl"); err == nil {
		os.Remove("/tmp/should-not-be-created.jsonl")
		t.Error("export path should not have been created when capability missing")
	}
}

func TestOpenForkPicker_NoForkerFailsSoftly(t *testing.T) {
	t.Parallel()
	m := testModel(t, &minimalStore{inner: newFullFakeStore()})
	m.session.Messages = []conversation.Message{
		{ID: "m1", Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "q"}}},
	}
	cmd := m.openForkPicker()
	if cmd != nil {
		t.Errorf("expected nil cmd when forker missing")
	}
	if m.state != stateIdle {
		t.Errorf("expected stateIdle after missing-capability rejection, got %v", m.state)
	}
	if !strings.Contains(m.output.String(), "does not support branching") {
		t.Errorf("expected capability error, got:\n%s", m.output.String())
	}
}

func TestOpenForkPicker_SetsCursorToLeaf(t *testing.T) {
	t.Parallel()
	m := testModel(t, newFullFakeStore())
	m.session.Messages = []conversation.Message{
		{ID: "m1", Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "q"}}},
		{ID: "m2", Role: conversation.RoleAssistant, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "a"}}},
		{ID: "m3", Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "q2"}}},
	}
	cmd := m.openForkPicker()
	if cmd != nil {
		t.Errorf("expected nil cmd, got one")
	}
	if m.state != stateForkPicker {
		t.Errorf("expected stateForkPicker, got %v", m.state)
	}
	if m.forkCursor != 2 {
		t.Errorf("expected forkCursor=2 (leaf), got %d", m.forkCursor)
	}
}

func TestRenderSessionStatus(t *testing.T) {
	t.Parallel()
	m := testModel(t, newFullFakeStore())
	m.session.Name = "My run"
	m.session.Messages = []conversation.Message{
		{Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hi"}}},
	}
	got := m.renderSessionStatus()
	if !strings.Contains(got, "session: My run") {
		t.Errorf("expected session name, got %q", got)
	}
	if !strings.Contains(got, "1 msgs") {
		t.Errorf("expected message count, got %q", got)
	}
}

func TestExpandHome(t *testing.T) {
	t.Parallel()
	got, err := expandHome("plain/path.jsonl")
	if err != nil {
		t.Fatalf("expandHome: %v", err)
	}
	if got != "plain/path.jsonl" {
		t.Errorf("plain path should pass through, got %q", got)
	}

	got, err = expandHome("~/foo.jsonl")
	if err != nil {
		t.Fatalf("expandHome: %v", err)
	}
	home, _ := os.UserHomeDir()
	want := filepath.Join(home, "foo.jsonl")
	if got != want {
		t.Errorf("expected %q, got %q", want, got)
	}
}
