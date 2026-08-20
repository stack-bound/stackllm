package sqlitestore

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/session"
)

// TestStore_AppendBlocksToExistingTailMessage verifies that Save
// persists blocks appended in memory to an already-saved message (the
// streaming "assistant message grows" case), and that the new blocks
// round-trip through Load in order.
func TestStore_AppendBlocksToExistingTailMessage(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)
	sess := simpleSession(t, store, "grower")

	// Append a second block to the persisted message.
	msg := &sess.Messages[len(sess.Messages)-1]
	newBlock := conversation.Block{
		ID:   conversation.NewID(),
		Type: conversation.BlockText,
		Text: "appended later",
	}
	msg.Blocks = append(msg.Blocks, newBlock)

	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("re-Save: %v", err)
	}

	loaded, err := store.Load(ctx, sess.ID)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got := loaded.Messages[len(loaded.Messages)-1].Blocks
	if len(got) != 2 {
		t.Fatalf("blocks after append = %d, want 2", len(got))
	}
	if got[1].ID != newBlock.ID || got[1].Text != "appended later" {
		t.Errorf("appended block = %+v, want ID %s with text 'appended later'", got[1], newBlock.ID)
	}
}

// failingWriter fails after n successful writes.
type failingWriter struct {
	n int
}

func (w *failingWriter) Write(p []byte) (int, error) {
	if w.n <= 0 {
		return 0, fmt.Errorf("writer full")
	}
	w.n--
	return len(p), nil
}

func TestExportJSONL_WriterErrors(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)
	sess := simpleSession(t, store, "exporter")

	// Header write fails immediately.
	err := store.ExportJSONL(ctx, sess.ID, &failingWriter{n: 0})
	if err == nil || !strings.Contains(err.Error(), "header") {
		t.Errorf("error = %v, want header write error", err)
	}

	// Header succeeds (JSON + newline = 2 writes), message line fails.
	err = store.ExportJSONL(ctx, sess.ID, &failingWriter{n: 2})
	if err == nil || !strings.Contains(err.Error(), "message") {
		t.Errorf("error = %v, want message write error", err)
	}
}

func TestImportJSONL_HeaderOnlyAndGarbageMessage(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)

	// A header without a state field (marshalState's nil branch) plus a
	// single message imports cleanly under a fresh session ID.
	id, err := store.ImportJSONL(ctx, strings.NewReader(
		`{"kind":"session_header","id":"orig","name":"imported-min"}`+"\n"+
			`{"kind":"message","id":"m1","role":"user","blocks":[{"type":"text","text":"imported hello"}]}`+"\n",
	))
	if err != nil {
		t.Fatalf("ImportJSONL minimal: %v", err)
	}
	if id == "orig" {
		t.Error("import must allocate a fresh session ID")
	}
	loaded, err := store.Load(ctx, id)
	if err != nil {
		t.Fatalf("Load imported: %v", err)
	}
	if loaded.Name != "imported-min" {
		t.Errorf("Name = %q, want imported-min", loaded.Name)
	}
	if len(loaded.Messages) != 1 || loaded.Messages[0].TextContent() != "imported hello" {
		t.Errorf("messages = %+v, want single 'imported hello'", loaded.Messages)
	}

	// A valid header followed by a garbage line must fail.
	_, err = store.ImportJSONL(ctx, strings.NewReader(
		`{"kind":"session_header","id":"x"}`+"\n"+"garbage-not-json\n",
	))
	if err == nil {
		t.Error("expected error for garbage message line")
	}
}

func TestImportJSONL_ZeroMessagesLoadsBack(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)

	// A header-only export (no messages) must import to a session whose
	// current_leaf_id is NULL, not the empty string — otherwise Load
	// fails with "current_leaf_id not found".
	id, err := store.ImportJSONL(ctx, strings.NewReader(
		`{"kind":"session_header","id":"orig","name":"empty-import"}`+"\n",
	))
	if err != nil {
		t.Fatalf("ImportJSONL header-only: %v", err)
	}
	loaded, err := store.Load(ctx, id)
	if err != nil {
		t.Fatalf("Load imported empty session: %v", err)
	}
	if loaded.Name != "empty-import" {
		t.Errorf("Name = %q, want empty-import", loaded.Name)
	}
	if len(loaded.Messages) != 0 {
		t.Errorf("messages = %d, want 0", len(loaded.Messages))
	}
}

func TestLoad_NullStateJSONNormalisedToEmptyMap(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)
	sess := simpleSession(t, store, "null-state")

	if _, err := store.DB().Exec(
		`UPDATE stackllm_sessions SET state_json = 'null' WHERE id = ?`, sess.ID,
	); err != nil {
		t.Fatalf("null state_json: %v", err)
	}

	loaded, err := store.Load(ctx, sess.ID)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.State == nil {
		t.Error("State should be a non-nil empty map for null state_json")
	}
	if len(loaded.State) != 0 {
		t.Errorf("State = %v, want empty", loaded.State)
	}
}

// TestExportImport_RoundTripWithArtifact exercises the artifact
// hydration path on export and the artifact re-offload path on import
// for a payload above the offload threshold.
func TestExportImport_RoundTripWithArtifact(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)

	big := strings.Repeat("artifact payload line\n", (defaultArtifactThreshold/22)+10)
	sess := simpleSession(t, store, "artifacted")
	sess.AppendMessage(conversation.Message{
		Role: conversation.RoleTool,
		Blocks: []conversation.Block{{
			Type:       conversation.BlockToolResult,
			ToolCallID: "call_big",
			Text:       big,
		}},
	})
	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("Save: %v", err)
	}

	var buf bytes.Buffer
	if err := store.ExportJSONL(ctx, sess.ID, &buf); err != nil {
		t.Fatalf("ExportJSONL: %v", err)
	}
	// The export must contain the full payload, not just the preview.
	if !strings.Contains(buf.String(), "artifact payload line") {
		t.Fatal("export missing artifact text")
	}

	newID, err := store.ImportJSONL(ctx, bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("ImportJSONL: %v", err)
	}
	imported, err := store.Load(ctx, newID)
	if err != nil {
		t.Fatalf("Load imported: %v", err)
	}
	last := imported.Messages[len(imported.Messages)-1]
	blk := last.Blocks[0]
	if blk.ArtifactRef == nil {
		t.Fatal("imported big tool_result should be artifact-backed")
	}
	data, mime, err := store.HydrateArtifact(ctx, blk.ArtifactRef.ID)
	if err != nil {
		t.Fatalf("HydrateArtifact: %v", err)
	}
	if mime != "text/plain" {
		t.Errorf("mime = %q, want text/plain", mime)
	}
	if string(data) != big {
		t.Errorf("hydrated artifact length = %d, want %d (content must round-trip)", len(data), len(big))
	}
}

func TestListBranches_PreviewFromChildren(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)
	sess := simpleSession(t, store, "branch-previews")
	rootID := sess.Messages[0].ID

	// Create a divergent sibling by rewinding to the root and saving a
	// different continuation.
	sess.AppendMessage(conversation.Message{
		Role:   conversation.RoleAssistant,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "first branch reply"}},
	})
	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("Save branch 1: %v", err)
	}
	if err := store.Rewind(ctx, sess.ID, rootID); err != nil {
		t.Fatalf("Rewind: %v", err)
	}
	reloaded, err := store.Load(ctx, sess.ID)
	if err != nil {
		t.Fatalf("Load after rewind: %v", err)
	}
	reloaded.AppendMessage(conversation.Message{
		Role:   conversation.RoleAssistant,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "second branch reply"}},
	})
	if err := store.Save(ctx, reloaded); err != nil {
		t.Fatalf("Save branch 2: %v", err)
	}

	refs, err := store.ListBranches(ctx, sess.ID, rootID)
	if err != nil {
		t.Fatalf("ListBranches: %v", err)
	}
	if len(refs) != 2 {
		t.Fatalf("branches = %d, want 2", len(refs))
	}
	previews := map[string]bool{}
	for _, r := range refs {
		previews[r.Preview] = true
		if r.Role != conversation.RoleAssistant {
			t.Errorf("branch role = %q, want assistant", r.Role)
		}
	}
	if !previews["first branch reply"] || !previews["second branch reply"] {
		t.Errorf("previews = %v, want both branch texts", previews)
	}
}

// TestStore_SaveLoadEmptySession pins the NULL-leaf path: a session
// with no messages round-trips with empty Messages and no error.
func TestStore_SaveLoadEmptySession(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)

	sess := session.New()
	sess.Name = "empty"
	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := store.Load(ctx, sess.ID)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.Name != "empty" {
		t.Errorf("Name = %q, want empty", loaded.Name)
	}
	if len(loaded.Messages) != 0 {
		t.Errorf("messages = %d, want 0", len(loaded.Messages))
	}
}

// TestNew_EmptyVersionTableRunsAllMigrations covers the ErrNoRows
// branch of the version probe: the table exists but has no row, so
// every migration must run and record the latest version.
func TestNew_EmptyVersionTableRunsAllMigrations(t *testing.T) {
	t.Parallel()

	db := newSharedDB(t)
	if _, err := db.Exec(`CREATE TABLE stackllm_schema_version(version INTEGER NOT NULL)`); err != nil {
		t.Fatalf("create empty version table: %v", err)
	}

	store, err := New(db)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	var version int
	if err := store.DB().QueryRow(`SELECT version FROM stackllm_schema_version`).Scan(&version); err != nil {
		t.Fatalf("read version: %v", err)
	}
	if version != latestSchemaVersion {
		t.Errorf("version = %d, want %d", version, latestSchemaVersion)
	}

	// The migrated schema must actually work.
	sess := session.New()
	sess.AppendMessage(conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "post-migration"}},
	})
	if err := store.Save(context.Background(), sess); err != nil {
		t.Errorf("Save after migration: %v", err)
	}
}
