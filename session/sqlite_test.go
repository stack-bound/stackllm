package session

import (
	"bytes"
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stack-bound/stackllm/conversation"

	_ "modernc.org/sqlite"
)

// newFileStore opens a fresh SQLiteStore in a per-test temporary
// directory and registers Close on cleanup.
func newFileStore(t *testing.T) *SQLiteStore {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.db")
	store, err := OpenSQLiteStore(SQLiteConfig{Path: path})
	if err != nil {
		t.Fatalf("OpenSQLiteStore: %v", err)
	}
	t.Cleanup(func() { store.Close() })
	return store
}

// newSharedDB opens a *sql.DB the caller owns (for NewSQLiteStore
// tests) and pins it to a single connection so the test can exercise
// parent-app coexistence without pool races.
func newSharedDB(t *testing.T) *sql.DB {
	t.Helper()
	path := filepath.Join(t.TempDir(), "shared.db")
	db, err := sql.Open("sqlite", "file:"+path+"?_pragma=foreign_keys(1)&_pragma=journal_mode(wal)")
	if err != nil {
		t.Fatalf("sql.Open: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

// makeInterleavedSession returns a session whose messages exercise
// every block type in the representative combinations the store
// needs to handle: text/thinking interleave, multi-tool-use + tool
// results, an image with inline bytes, and redacted thinking.
//
// Returned alongside the session is the full 200 KB payload used by
// the tool_result offload test so dedup/artifact assertions can hash
// against it directly.
func makeInterleavedSession(t *testing.T) (*Session, string, []byte) {
	t.Helper()
	sess := New()
	sess.Name = "roundtrip"
	sess.ProjectPath = "/tmp/demo"
	sess.Model = "gpt-4o"
	sess.SetState("foo", "bar")

	// System message.
	sess.AppendMessage(conversation.Message{
		Role:   conversation.RoleSystem,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "You are a helpful assistant."}},
	})

	// User message with inline image (500 KB -> artifact).
	img := bytes.Repeat([]byte{0x89, 0x50, 0x4E, 0x47}, 128*1024)
	sess.AppendMessage(conversation.Message{
		Role: conversation.RoleUser,
		Blocks: []conversation.Block{
			{Type: conversation.BlockText, Text: "what's in this png?"},
			{Type: conversation.BlockImage, MimeType: "image/png", ImageData: img},
		},
	})

	// Assistant with interleaved thinking/text/tool_use x 7.
	sess.AppendMessage(conversation.Message{
		Role:     conversation.RoleAssistant,
		Model:    "gpt-4o",
		Duration: 2300 * time.Millisecond,
		Blocks: []conversation.Block{
			{Type: conversation.BlockThinking, Text: "planning"},
			{Type: conversation.BlockText, Text: "Let me check the file and search for the bug."},
			{Type: conversation.BlockToolUse, ToolCallID: "c1", ToolName: "read_file", ToolArgsJSON: `{"p":"a.go"}`},
			{Type: conversation.BlockThinking, Text: "found it"},
			{Type: conversation.BlockToolUse, ToolCallID: "c2", ToolName: "grep", ToolArgsJSON: `{"q":"Bug"}`},
			{Type: conversation.BlockThinking, Text: "analyzing match"},
			{Type: conversation.BlockText, Text: "auth"},
		},
	})

	// Tool-role message: one small result, one 200 KB result (offloads).
	small := "file contents here (auth)"
	big := strings.Repeat("big tool output line\n", 11_000) // ~220 KB
	sess.AppendMessage(conversation.Message{
		Role: conversation.RoleTool,
		Blocks: []conversation.Block{
			{Type: conversation.BlockToolResult, ToolCallID: "c1", Text: small},
			{Type: conversation.BlockToolResult, ToolCallID: "c2", Text: big},
		},
	})

	// Final assistant message with thinking + text.
	sess.AppendMessage(conversation.Message{
		Role:  conversation.RoleAssistant,
		Model: "gpt-4o",
		Blocks: []conversation.Block{
			{Type: conversation.BlockThinking, Text: "wrapping up"},
			{Type: conversation.BlockText, Text: "The bug is in the auth path."},
		},
	})

	// Also exercise redacted thinking by adding a trailing assistant
	// block set containing that opaque payload. Keeps the store
	// test exercising the artifact path for redacted content.
	sess.AppendMessage(conversation.Message{
		Role: conversation.RoleAssistant,
		Blocks: []conversation.Block{
			{Type: conversation.BlockRedactedThinking, RedactedData: []byte("super secret reasoning bytes")},
		},
	})

	return sess, big, img
}

// ---------------------------------------------------------------------
// Path resolution.
// ---------------------------------------------------------------------

func TestSQLiteConfig_ResolvePath_AppName(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_DATA_HOME", dir)

	store, err := OpenSQLiteStore(SQLiteConfig{AppName: "stackllm-test"})
	if err != nil {
		t.Fatalf("OpenSQLiteStore: %v", err)
	}
	defer store.Close()

	want := filepath.Join(dir, "stackllm-test", "state.db")
	if _, err := os.Stat(want); err != nil {
		t.Fatalf("expected db at %s, got: %v", want, err)
	}
}

func TestSQLiteConfig_ResolvePath_RequiresConfig(t *testing.T) {
	t.Parallel()
	_, err := OpenSQLiteStore(SQLiteConfig{})
	if err == nil {
		t.Fatal("expected error for empty config")
	}
	if !strings.Contains(err.Error(), "AppName or Path") {
		t.Errorf("error = %v, want 'AppName or Path'", err)
	}
}

func TestSQLiteConfig_ResolvePath_PathWins(t *testing.T) {
	path := filepath.Join(t.TempDir(), "explicit.db")
	store, err := OpenSQLiteStore(SQLiteConfig{AppName: "ignored", Path: path})
	if err != nil {
		t.Fatalf("OpenSQLiteStore: %v", err)
	}
	defer store.Close()
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("expected db at %s, got: %v", path, err)
	}
}

// ---------------------------------------------------------------------
// Parent-app sharing.
// ---------------------------------------------------------------------

func TestSQLiteStore_ParentAppSharing(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	db := newSharedDB(t)
	if _, err := db.ExecContext(ctx, `CREATE TABLE memories (id INTEGER PRIMARY KEY, text TEXT)`); err != nil {
		t.Fatalf("create parent table: %v", err)
	}
	if _, err := db.ExecContext(ctx, `INSERT INTO memories (text) VALUES ('remembered')`); err != nil {
		t.Fatalf("insert parent row: %v", err)
	}

	store, err := NewSQLiteStore(db)
	if err != nil {
		t.Fatalf("NewSQLiteStore: %v", err)
	}

	// Save a tiny stackllm session.
	sess := New()
	sess.AppendMessage(conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hello"}},
	})
	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Parent table row still there.
	var text string
	if err := db.QueryRowContext(ctx, `SELECT text FROM memories WHERE id = 1`).Scan(&text); err != nil {
		t.Fatalf("query parent: %v", err)
	}
	if text != "remembered" {
		t.Errorf("parent row text = %q, want remembered", text)
	}

	// stackllm session is loadable.
	loaded, err := store.Load(ctx, sess.ID)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := loaded.Messages[0].TextContent(); got != "hello" {
		t.Errorf("loaded text = %q", got)
	}

	// Close is a no-op for NewSQLiteStore; the caller's DB stays open.
	if err := store.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := db.PingContext(ctx); err != nil {
		t.Fatalf("caller DB closed unexpectedly: %v", err)
	}
}

// ---------------------------------------------------------------------
// Save/Load full round-trip.
// ---------------------------------------------------------------------

func TestSQLiteStore_SaveLoadRoundTrip(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)

	sess, bigPayload, imgPayload := makeInterleavedSession(t)

	// Snapshot IDs before Save so we can assert they round-trip.
	originalMsgIDs := make([]string, len(sess.Messages))
	originalBlockIDs := make([][]string, len(sess.Messages))
	for i, m := range sess.Messages {
		originalMsgIDs[i] = m.ID
		for _, b := range m.Blocks {
			originalBlockIDs[i] = append(originalBlockIDs[i], b.ID)
		}
	}

	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := store.Load(ctx, sess.ID)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if loaded.Name != "roundtrip" || loaded.ProjectPath != "/tmp/demo" || loaded.Model != "gpt-4o" {
		t.Errorf("session metadata not restored: %+v", loaded)
	}
	if v, _ := loaded.GetState("foo"); v != "bar" {
		t.Errorf("state foo = %v, want bar", v)
	}

	if len(loaded.Messages) != len(sess.Messages) {
		t.Fatalf("loaded messages = %d, want %d", len(loaded.Messages), len(sess.Messages))
	}

	// Every original message+block ID survived the round-trip.
	for i, m := range loaded.Messages {
		if m.ID != originalMsgIDs[i] {
			t.Errorf("messages[%d].ID = %s, want %s", i, m.ID, originalMsgIDs[i])
		}
		if len(m.Blocks) != len(originalBlockIDs[i]) {
			t.Fatalf("messages[%d] blocks = %d, want %d", i, len(m.Blocks), len(originalBlockIDs[i]))
		}
		for j, b := range m.Blocks {
			if b.ID != originalBlockIDs[i][j] {
				t.Errorf("messages[%d].blocks[%d].ID = %s, want %s", i, j, b.ID, originalBlockIDs[i][j])
			}
		}
	}

	// Assistant block order preserved.
	assistant := loaded.Messages[2]
	wantTypes := []conversation.BlockType{
		conversation.BlockThinking, conversation.BlockText, conversation.BlockToolUse,
		conversation.BlockThinking, conversation.BlockToolUse,
		conversation.BlockThinking, conversation.BlockText,
	}
	if len(assistant.Blocks) != len(wantTypes) {
		t.Fatalf("assistant blocks = %d, want %d", len(assistant.Blocks), len(wantTypes))
	}
	for i, wt := range wantTypes {
		if assistant.Blocks[i].Type != wt {
			t.Errorf("assistant.Blocks[%d].Type = %q, want %q", i, assistant.Blocks[i].Type, wt)
		}
	}
	// tool_use linkage
	if assistant.Blocks[2].ToolCallID != "c1" || assistant.Blocks[2].ToolName != "read_file" {
		t.Errorf("tool_use[0] lost metadata: %+v", assistant.Blocks[2])
	}
	if assistant.Blocks[4].ToolCallID != "c2" || assistant.Blocks[4].ToolName != "grep" {
		t.Errorf("tool_use[1] lost metadata: %+v", assistant.Blocks[4])
	}
	if assistant.Model != "gpt-4o" || assistant.Duration != 2300*time.Millisecond {
		t.Errorf("assistant metadata lost: model=%q duration=%v", assistant.Model, assistant.Duration)
	}

	// Tool message: small is inline, big is artifact-backed.
	tool := loaded.Messages[3]
	if len(tool.Blocks) != 2 {
		t.Fatalf("tool blocks = %d, want 2", len(tool.Blocks))
	}
	if tool.Blocks[0].ArtifactRef != nil {
		t.Errorf("small tool_result should be inline, got artifact %+v", tool.Blocks[0].ArtifactRef)
	}
	if tool.Blocks[0].Text != "file contents here (auth)" {
		t.Errorf("small tool_result text = %q", tool.Blocks[0].Text)
	}
	if tool.Blocks[1].ArtifactRef == nil {
		t.Fatal("big tool_result should have ArtifactRef populated")
	}
	if tool.Blocks[1].ArtifactRef.ByteSize != int64(len(bigPayload)) {
		t.Errorf("big tool_result ArtifactRef.ByteSize = %d, want %d", tool.Blocks[1].ArtifactRef.ByteSize, len(bigPayload))
	}
	if tool.Blocks[1].ArtifactRef.SHA256 != sha256Hex([]byte(bigPayload)) {
		t.Error("big tool_result SHA256 mismatch")
	}
	if tool.Blocks[1].Text == "" {
		t.Error("big tool_result should carry preview text after Load")
	}
	if len(tool.Blocks[1].Text) > artifactPreviewBytes+64 {
		t.Errorf("big tool_result preview is unreasonably large: %d bytes", len(tool.Blocks[1].Text))
	}

	// User image block should be artifact-backed.
	user := loaded.Messages[1]
	if user.Blocks[1].Type != conversation.BlockImage {
		t.Fatalf("user image block type = %q", user.Blocks[1].Type)
	}
	if user.Blocks[1].ArtifactRef == nil {
		t.Fatal("user image should be artifact-backed")
	}
	if user.Blocks[1].ArtifactRef.ByteSize != int64(len(imgPayload)) {
		t.Errorf("user image byte_size = %d, want %d", user.Blocks[1].ArtifactRef.ByteSize, len(imgPayload))
	}
	// Load should not hydrate raw bytes.
	if len(user.Blocks[1].ImageData) != 0 {
		t.Errorf("Load should not hydrate ImageData, got %d bytes", len(user.Blocks[1].ImageData))
	}

	// HydrateArtifact returns the full payload.
	data, mime, err := store.HydrateArtifact(ctx, tool.Blocks[1].ArtifactRef.ID)
	if err != nil {
		t.Fatalf("HydrateArtifact: %v", err)
	}
	if string(data) != bigPayload {
		t.Errorf("hydrated payload mismatch (%d bytes vs %d bytes)", len(data), len(bigPayload))
	}
	if mime != "text/plain" {
		t.Errorf("hydrated mime = %q, want text/plain", mime)
	}

	// Redacted thinking lives in the last assistant message.
	redacted := loaded.Messages[5]
	if len(redacted.Blocks) != 1 || redacted.Blocks[0].Type != conversation.BlockRedactedThinking {
		t.Fatalf("redacted thinking not loaded: %+v", redacted.Blocks)
	}
	if redacted.Blocks[0].ArtifactRef == nil {
		t.Fatal("redacted thinking must be artifact-backed")
	}
	bts, _, err := store.HydrateArtifact(ctx, redacted.Blocks[0].ArtifactRef.ID)
	if err != nil {
		t.Fatalf("HydrateArtifact redacted: %v", err)
	}
	if string(bts) != "super secret reasoning bytes" {
		t.Errorf("redacted bytes lost: %q", string(bts))
	}
}

// ---------------------------------------------------------------------
// Repeated Save is append-only.
// ---------------------------------------------------------------------

func TestSQLiteStore_RepeatSaveIsAppendOnly(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)

	sess := New()
	sess.AppendMessage(conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "first"}},
	})
	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("Save 1: %v", err)
	}

	countMessages := func() int {
		var n int
		if err := store.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM stackllm_messages WHERE session_id = ?`, sess.ID).Scan(&n); err != nil {
			t.Fatalf("count messages: %v", err)
		}
		return n
	}
	countBlocks := func() int {
		var n int
		if err := store.db.QueryRowContext(ctx, `
			SELECT COUNT(*) FROM stackllm_blocks b
			JOIN stackllm_messages m ON b.message_id = m.id
			WHERE m.session_id = ?`, sess.ID).Scan(&n); err != nil {
			t.Fatalf("count blocks: %v", err)
		}
		return n
	}

	if got := countMessages(); got != 1 {
		t.Fatalf("after first save messages = %d, want 1", got)
	}
	if got := countBlocks(); got != 1 {
		t.Fatalf("after first save blocks = %d, want 1", got)
	}

	// Add one message with 3 blocks, save again.
	sess.AppendMessage(conversation.Message{
		Role: conversation.RoleAssistant,
		Blocks: []conversation.Block{
			{Type: conversation.BlockThinking, Text: "let me think"},
			{Type: conversation.BlockText, Text: "here's my answer"},
			{Type: conversation.BlockToolUse, ToolCallID: "c1", ToolName: "noop", ToolArgsJSON: "{}"},
		},
	})
	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("Save 2: %v", err)
	}

	if got := countMessages(); got != 2 {
		t.Errorf("after second save messages = %d, want 2", got)
	}
	if got := countBlocks(); got != 4 {
		t.Errorf("after second save blocks = %d, want 4", got)
	}
}

// ---------------------------------------------------------------------
// In-memory mutation is silently dropped.
// ---------------------------------------------------------------------

func TestSQLiteStore_InMemoryMutationDropped(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)

	sess := New()
	sess.AppendMessage(conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "original"}},
	})
	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Mutate in memory and re-save. Because the block ID already
	// exists in the DB, Save ignores the mutation.
	sess.Messages[0].Blocks[0].Text = "mutated in memory"
	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("Save 2: %v", err)
	}

	loaded, err := store.Load(ctx, sess.ID)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := loaded.Messages[0].Blocks[0].Text; got != "original" {
		t.Errorf("block text = %q, want original (mutation should be dropped)", got)
	}
}

// ---------------------------------------------------------------------
// Parent chain divergence is rejected.
// ---------------------------------------------------------------------

func TestSQLiteStore_ParentChainDivergenceRejected(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)

	sess := New()
	for i := 0; i < 3; i++ {
		sess.AppendMessage(conversation.Message{
			Role:   conversation.RoleUser,
			Blocks: []conversation.Block{{Type: conversation.BlockText, Text: fmt.Sprintf("m%d", i)}},
		})
	}
	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Snapshot block count so we can prove the failed Save didn't
	// leak any rows.
	var beforeBlocks int
	store.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM stackllm_blocks`).Scan(&beforeBlocks)

	// Load and drop the middle message. The remaining chain is
	// [m0, m2] but m2's DB parent is m1 — divergence detected.
	loaded, err := store.Load(ctx, sess.ID)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	loaded.Messages = []conversation.Message{loaded.Messages[0], loaded.Messages[2]}

	err = store.Save(ctx, loaded)
	if err == nil {
		t.Fatal("expected error for divergent parent chain")
	}
	if !strings.Contains(err.Error(), "parent chain diverged") {
		t.Errorf("error = %v, want 'parent chain diverged'", err)
	}

	var afterBlocks int
	store.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM stackllm_blocks`).Scan(&afterBlocks)
	if afterBlocks != beforeBlocks {
		t.Errorf("blocks leaked on failed Save: before=%d after=%d", beforeBlocks, afterBlocks)
	}
}

// ---------------------------------------------------------------------
// Artifact dedupe across sessions.
// ---------------------------------------------------------------------

func TestSQLiteStore_ArtifactDedupe(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)

	big := strings.Repeat("dedupe me\n", 20_000) // ~200 KB

	for i := 0; i < 2; i++ {
		sess := New()
		sess.AppendMessage(conversation.Message{
			Role:   conversation.RoleUser,
			Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hi"}},
		})
		sess.AppendMessage(conversation.Message{
			Role: conversation.RoleTool,
			Blocks: []conversation.Block{
				{Type: conversation.BlockToolResult, ToolCallID: "c1", Text: big},
			},
		})
		if err := store.Save(ctx, sess); err != nil {
			t.Fatalf("Save %d: %v", i, err)
		}
	}

	var count int
	if err := store.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM stackllm_artifacts`).Scan(&count); err != nil {
		t.Fatalf("count artifacts: %v", err)
	}
	if count != 1 {
		t.Errorf("artifact rows = %d, want 1 (dedupe)", count)
	}
}

// ---------------------------------------------------------------------
// Fork.
// ---------------------------------------------------------------------

func TestSQLiteStore_Fork(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)

	// Five messages, middle one carries an offloaded tool result so
	// we can assert artifact reuse.
	sess := New()
	big := strings.Repeat("fork payload\n", 12_000)
	for i := 0; i < 5; i++ {
		if i == 2 {
			sess.AppendMessage(conversation.Message{
				Role: conversation.RoleTool,
				Blocks: []conversation.Block{
					{Type: conversation.BlockToolResult, ToolCallID: "c1", Text: big},
				},
			})
			continue
		}
		sess.AppendMessage(conversation.Message{
			Role:   conversation.RoleUser,
			Blocks: []conversation.Block{{Type: conversation.BlockText, Text: fmt.Sprintf("m%d", i)}},
		})
	}
	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("Save: %v", err)
	}

	forkAt := sess.Messages[2].ID
	forked, err := store.Fork(ctx, sess.ID, forkAt, "my-fork")
	if err != nil {
		t.Fatalf("Fork: %v", err)
	}

	// Fork contains 3 messages (through m2).
	if len(forked.Messages) != 3 {
		t.Fatalf("fork messages = %d, want 3", len(forked.Messages))
	}
	if forked.Name != "my-fork" {
		t.Errorf("fork name = %q", forked.Name)
	}
	for i, m := range forked.Messages {
		if m.ID == sess.Messages[i].ID {
			t.Errorf("fork msg %d reused original ID %s", i, m.ID)
		}
		for j, b := range m.Blocks {
			if b.ID == sess.Messages[i].Blocks[j].ID {
				t.Errorf("fork msg %d block %d reused original ID", i, j)
			}
		}
	}
	// Artifact reuse: source and fork tool_result point at the same
	// artifact_id.
	var srcArtifact, forkArtifact sql.NullString
	store.db.QueryRow(`SELECT artifact_id FROM stackllm_blocks WHERE message_id = ? AND type = 'tool_result'`, sess.Messages[2].ID).Scan(&srcArtifact)
	store.db.QueryRow(`SELECT artifact_id FROM stackllm_blocks WHERE message_id = ? AND type = 'tool_result'`, forked.Messages[2].ID).Scan(&forkArtifact)
	if !srcArtifact.Valid || !forkArtifact.Valid || srcArtifact.String != forkArtifact.String {
		t.Errorf("fork did not reuse artifact: src=%v fork=%v", srcArtifact, forkArtifact)
	}
	var artifactCount int
	store.db.QueryRow(`SELECT COUNT(*) FROM stackllm_artifacts`).Scan(&artifactCount)
	if artifactCount != 1 {
		t.Errorf("artifact rows = %d, want 1 (fork should reuse)", artifactCount)
	}

	// Original session untouched.
	original, err := store.Load(ctx, sess.ID)
	if err != nil {
		t.Fatalf("Load original: %v", err)
	}
	if len(original.Messages) != 5 {
		t.Errorf("original messages = %d, want 5", len(original.Messages))
	}
}

// ---------------------------------------------------------------------
// Rewind creates a sibling branch.
// ---------------------------------------------------------------------

func TestSQLiteStore_RewindAndListBranches(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)

	sess := New()
	for i := 0; i < 5; i++ {
		sess.AppendMessage(conversation.Message{
			Role:   conversation.RoleUser,
			Blocks: []conversation.Block{{Type: conversation.BlockText, Text: fmt.Sprintf("m%d", i)}},
		})
	}
	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("Save: %v", err)
	}

	m2ID := sess.Messages[2].ID
	origM3ID := sess.Messages[3].ID

	if err := store.Rewind(ctx, sess.ID, m2ID); err != nil {
		t.Fatalf("Rewind: %v", err)
	}

	// Loading after Rewind gives us [m0, m1, m2] only.
	rewound, err := store.Load(ctx, sess.ID)
	if err != nil {
		t.Fatalf("Load after rewind: %v", err)
	}
	if len(rewound.Messages) != 3 {
		t.Fatalf("after rewind messages = %d, want 3", len(rewound.Messages))
	}

	// Append two new messages and Save. The new messages are
	// parented on m2, creating a sibling branch to the original
	// m3/m4 which still exist in the DB.
	rewound.AppendMessage(conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "new3"}},
	})
	rewound.AppendMessage(conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "new4"}},
	})
	if err := store.Save(ctx, rewound); err != nil {
		t.Fatalf("Save after rewind: %v", err)
	}

	newM3ID := rewound.Messages[3].ID
	newM4ID := rewound.Messages[4].ID

	// Leaf is new4.
	var leaf sql.NullString
	store.db.QueryRow(`SELECT current_leaf_id FROM stackllm_sessions WHERE id = ?`, sess.ID).Scan(&leaf)
	if !leaf.Valid || leaf.String != newM4ID {
		t.Errorf("current_leaf_id = %v, want %s", leaf, newM4ID)
	}

	// Original messages 3 and 4 still exist in the DB.
	var originalM3Exists int
	store.db.QueryRow(`SELECT 1 FROM stackllm_messages WHERE id = ?`, origM3ID).Scan(&originalM3Exists)
	if originalM3Exists != 1 {
		t.Error("original m3 was deleted by Rewind")
	}

	// ListBranches returns both m3 (original) and m3 (new) as
	// children of m2.
	refs, err := store.ListBranches(ctx, sess.ID, m2ID)
	if err != nil {
		t.Fatalf("ListBranches: %v", err)
	}
	if len(refs) != 2 {
		t.Fatalf("branches = %d, want 2 — got %+v", len(refs), refs)
	}
	seen := map[string]bool{}
	for _, r := range refs {
		seen[r.MessageID] = true
	}
	if !seen[origM3ID] || !seen[newM3ID] {
		t.Errorf("branches missing entries: origM3=%v newM3=%v", seen[origM3ID], seen[newM3ID])
	}
}

// ---------------------------------------------------------------------
// FTS: block-type filter + preview-only indexing.
// ---------------------------------------------------------------------

func TestSQLiteStore_Search_BlockTypeFilter(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)

	sess := New()
	sess.AppendMessage(conversation.Message{
		Role: conversation.RoleAssistant,
		Blocks: []conversation.Block{
			{Type: conversation.BlockThinking, Text: "auth looks suspicious here"},
			{Type: conversation.BlockText, Text: "mentioning auth in plain text"},
		},
	})
	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// All text-bearing types.
	all, err := store.Search(ctx, "auth", "", nil, 10)
	if err != nil {
		t.Fatalf("Search all: %v", err)
	}
	if len(all) < 2 {
		t.Errorf("unfiltered hits = %d, want >= 2", len(all))
	}

	// Only thinking blocks.
	thinkOnly, err := store.Search(ctx, "auth", "", []conversation.BlockType{conversation.BlockThinking}, 10)
	if err != nil {
		t.Fatalf("Search thinking: %v", err)
	}
	if len(thinkOnly) != 1 {
		t.Fatalf("thinking hits = %d, want 1", len(thinkOnly))
	}
	if thinkOnly[0].BlockType != conversation.BlockThinking {
		t.Errorf("hit type = %q, want thinking", thinkOnly[0].BlockType)
	}
}

func TestSQLiteStore_Search_PreviewOnlyForBigToolResults(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)

	prefix := "STACKLLM_PREVIEW_TOKEN_XYZ\n"
	middle := strings.Repeat("x", 100_000)
	tail := "STACKLLM_TAIL_TOKEN_QRS"
	big := prefix + middle + tail

	sess := New()
	sess.AppendMessage(conversation.Message{
		Role: conversation.RoleTool,
		Blocks: []conversation.Block{
			{Type: conversation.BlockToolResult, ToolCallID: "c1", Text: big},
		},
	})
	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("Save: %v", err)
	}

	previewHits, err := store.Search(ctx, "STACKLLM_PREVIEW_TOKEN_XYZ", "", nil, 10)
	if err != nil {
		t.Fatalf("Search preview: %v", err)
	}
	if len(previewHits) != 1 {
		t.Errorf("preview hits = %d, want 1", len(previewHits))
	}

	tailHits, err := store.Search(ctx, "STACKLLM_TAIL_TOKEN_QRS", "", nil, 10)
	if err != nil {
		t.Fatalf("Search tail: %v", err)
	}
	if len(tailHits) != 0 {
		t.Errorf("tail hits = %d, want 0 (tail lives only in artifact, not FTS)", len(tailHits))
	}
}

// ---------------------------------------------------------------------
// JSONL export / import.
// ---------------------------------------------------------------------

func TestSQLiteStore_JSONLRoundTrip(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)

	sess, bigPayload, imgPayload := makeInterleavedSession(t)
	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("Save: %v", err)
	}

	var buf bytes.Buffer
	if err := store.ExportJSONL(ctx, sess.ID, &buf); err != nil {
		t.Fatalf("ExportJSONL: %v", err)
	}

	// Import into a fresh store and assert every block type survived.
	store2 := newFileStore(t)
	newID, err := store2.ImportJSONL(ctx, &buf)
	if err != nil {
		t.Fatalf("ImportJSONL: %v", err)
	}

	imported, err := store2.Load(ctx, newID)
	if err != nil {
		t.Fatalf("Load imported: %v", err)
	}
	if len(imported.Messages) != len(sess.Messages) {
		t.Fatalf("imported messages = %d, want %d", len(imported.Messages), len(sess.Messages))
	}

	// Big tool_result should be hydratable in the new store.
	tool := imported.Messages[3]
	if tool.Blocks[1].ArtifactRef == nil {
		t.Fatal("imported big tool_result missing ArtifactRef")
	}
	data, _, err := store2.HydrateArtifact(ctx, tool.Blocks[1].ArtifactRef.ID)
	if err != nil {
		t.Fatalf("HydrateArtifact after import: %v", err)
	}
	if string(data) != bigPayload {
		t.Error("imported tool_result payload mismatch")
	}

	// Image.
	user := imported.Messages[1]
	if user.Blocks[1].ArtifactRef == nil {
		t.Fatal("imported image missing ArtifactRef")
	}
	imgData, _, err := store2.HydrateArtifact(ctx, user.Blocks[1].ArtifactRef.ID)
	if err != nil {
		t.Fatalf("HydrateArtifact image: %v", err)
	}
	if !bytes.Equal(imgData, imgPayload) {
		t.Errorf("imported image payload mismatch: got %d bytes want %d bytes", len(imgData), len(imgPayload))
	}

	// Redacted thinking.
	redacted := imported.Messages[5]
	if redacted.Blocks[0].ArtifactRef == nil {
		t.Fatal("imported redacted thinking missing ArtifactRef")
	}
	rd, _, err := store2.HydrateArtifact(ctx, redacted.Blocks[0].ArtifactRef.ID)
	if err != nil {
		t.Fatalf("HydrateArtifact redacted: %v", err)
	}
	if string(rd) != "super secret reasoning bytes" {
		t.Errorf("imported redacted bytes lost: %q", string(rd))
	}
}

// ---------------------------------------------------------------------
// Concurrent readers (WAL).
// ---------------------------------------------------------------------

func TestSQLiteStore_ConcurrentReadersWAL(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	path := filepath.Join(t.TempDir(), "wal.db")
	writer, err := OpenSQLiteStore(SQLiteConfig{Path: path})
	if err != nil {
		t.Fatalf("writer open: %v", err)
	}
	defer writer.Close()

	// Seed a single session the reader will repeatedly Load.
	sess := New()
	sess.AppendMessage(conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "seed"}},
	})
	if err := writer.Save(ctx, sess); err != nil {
		t.Fatalf("seed Save: %v", err)
	}

	readerDB, err := sql.Open("sqlite", "file:"+path+"?_pragma=foreign_keys(1)&_pragma=journal_mode(wal)")
	if err != nil {
		t.Fatalf("reader sql.Open: %v", err)
	}
	defer readerDB.Close()
	reader, err := NewSQLiteStore(readerDB)
	if err != nil {
		t.Fatalf("reader NewSQLiteStore: %v", err)
	}

	var writerErr, readerErr atomic.Value
	done := make(chan struct{})

	go func() {
		defer close(done)
		for i := 0; i < 50; i++ {
			sess.AppendMessage(conversation.Message{
				Role: conversation.RoleUser,
				Blocks: []conversation.Block{
					{Type: conversation.BlockText, Text: fmt.Sprintf("msg %d", i)},
				},
			})
			if err := writer.Save(ctx, sess); err != nil {
				writerErr.Store(err)
				return
			}
		}
	}()

	// Reader loop runs in this goroutine until writer is done.
	readCount := 0
	for {
		select {
		case <-done:
			goto finished
		default:
		}
		loaded, err := reader.Load(ctx, sess.ID)
		if err != nil {
			readerErr.Store(err)
			break
		}
		if len(loaded.Messages) < 1 {
			readerErr.Store(fmt.Errorf("reader saw empty messages on iter %d", readCount))
			break
		}
		readCount++
	}
finished:
	<-done

	if v := writerErr.Load(); v != nil {
		t.Fatalf("writer err: %v", v)
	}
	if v := readerErr.Load(); v != nil {
		t.Fatalf("reader err: %v", v)
	}
	if readCount == 0 {
		t.Error("reader never ran a successful Load")
	}

	// Final state: writer's 51 messages are readable through the
	// reader handle.
	finalLoad, err := reader.Load(ctx, sess.ID)
	if err != nil {
		t.Fatalf("final Load: %v", err)
	}
	if len(finalLoad.Messages) != 51 {
		t.Errorf("final messages = %d, want 51", len(finalLoad.Messages))
	}
}

// ---------------------------------------------------------------------
// Schema version guard.
// ---------------------------------------------------------------------

func TestSQLiteStore_SchemaVersionGuard(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "future.db")

	store, err := OpenSQLiteStore(SQLiteConfig{Path: path})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	// Force the stored version past what this build supports.
	if _, err := store.db.ExecContext(ctx, `UPDATE stackllm_schema_version SET version = ?`, latestSchemaVersion+1); err != nil {
		t.Fatalf("bump version: %v", err)
	}
	store.Close()

	_, err = OpenSQLiteStore(SQLiteConfig{Path: path})
	if err == nil {
		t.Fatal("expected error reopening newer DB")
	}
	if !strings.Contains(err.Error(), "newer than supported") {
		t.Errorf("error = %v, want 'newer than supported'", err)
	}
}

// ---------------------------------------------------------------------
// Cycle detection.
// ---------------------------------------------------------------------

func TestSQLiteStore_CycleDetection(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)

	// Create a 2-message session normally.
	sess := New()
	sess.AppendMessage(conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "m0"}},
	})
	sess.AppendMessage(conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "m1"}},
	})
	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Corrupt: set m0's parent_id to m1, creating m0->m1->m0.
	if _, err := store.db.ExecContext(ctx,
		`UPDATE stackllm_messages SET parent_id = ? WHERE id = ?`,
		sess.Messages[1].ID, sess.Messages[0].ID); err != nil {
		t.Fatalf("corrupt parent: %v", err)
	}

	_, err := store.Load(ctx, sess.ID)
	if err == nil {
		t.Fatal("expected cycle error")
	}
	if !strings.Contains(err.Error(), "cycle") {
		t.Errorf("error = %v, want 'cycle'", err)
	}
}

// ---------------------------------------------------------------------
// Cascade delete (artifacts remain — v1 known limitation).
// ---------------------------------------------------------------------

func TestSQLiteStore_DeleteCascades(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)

	sess := New()
	big := strings.Repeat("cascade me\n", 10_000)
	sess.AppendMessage(conversation.Message{
		Role: conversation.RoleTool,
		Blocks: []conversation.Block{
			{Type: conversation.BlockToolResult, ToolCallID: "c1", Text: big},
		},
	})
	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Pre-check: one row in every table that cascades.
	mustCount := func(q string) int {
		var n int
		if err := store.db.QueryRowContext(ctx, q).Scan(&n); err != nil {
			t.Fatalf("%s: %v", q, err)
		}
		return n
	}

	if n := mustCount(`SELECT COUNT(*) FROM stackllm_sessions`); n != 1 {
		t.Fatalf("pre sessions = %d", n)
	}
	if n := mustCount(`SELECT COUNT(*) FROM stackllm_messages`); n != 1 {
		t.Fatalf("pre messages = %d", n)
	}
	if n := mustCount(`SELECT COUNT(*) FROM stackllm_blocks`); n != 1 {
		t.Fatalf("pre blocks = %d", n)
	}
	if n := mustCount(`SELECT COUNT(*) FROM stackllm_artifacts`); n != 1 {
		t.Fatalf("pre artifacts = %d", n)
	}
	if n := mustCount(`SELECT COUNT(*) FROM stackllm_blocks_fts WHERE stackllm_blocks_fts MATCH 'cascade'`); n != 1 {
		t.Fatalf("pre fts = %d", n)
	}

	if err := store.Delete(ctx, sess.ID); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	if n := mustCount(`SELECT COUNT(*) FROM stackllm_sessions`); n != 0 {
		t.Errorf("post sessions = %d, want 0", n)
	}
	if n := mustCount(`SELECT COUNT(*) FROM stackllm_messages`); n != 0 {
		t.Errorf("post messages = %d, want 0", n)
	}
	if n := mustCount(`SELECT COUNT(*) FROM stackllm_blocks`); n != 0 {
		t.Errorf("post blocks = %d, want 0", n)
	}
	if n := mustCount(`SELECT COUNT(*) FROM stackllm_blocks_fts WHERE stackllm_blocks_fts MATCH 'cascade'`); n != 0 {
		t.Errorf("post fts = %d, want 0", n)
	}
	// Artifacts are explicitly NOT cleaned up in v1.
	if n := mustCount(`SELECT COUNT(*) FROM stackllm_artifacts`); n != 1 {
		t.Errorf("post artifacts = %d, want 1 (v1 keeps orphans)", n)
	}
}

// ---------------------------------------------------------------------
// FTS5 availability guard.
// ---------------------------------------------------------------------

func TestSQLiteStore_FTS5AvailabilityProbe(t *testing.T) {
	t.Parallel()
	// Positive case: a freshly-opened modernc store must pass the
	// FTS5 probe. This is the guard that fires in production if a
	// future SQLite build ever drops ENABLE_FTS5.
	store := newFileStore(t)
	if err := checkFTS5Available(context.Background(), store.db); err != nil {
		t.Fatalf("expected modernc.org/sqlite to ship FTS5: %v", err)
	}
}

// TestSQLiteStore_FTS5Missing_BootstrapFails simulates a SQLite build
// that does not define ENABLE_FTS5 by swapping in a stub probe, and
// asserts that NewSQLiteStore refuses to bootstrap — the whole point
// of the guard is to fail fast with a clear message before any
// stackllm_* table is created.
//
// Not parallel: mutates the package-level probeCompileOptions hook.
func TestSQLiteStore_FTS5Missing_BootstrapFails(t *testing.T) {
	orig := probeCompileOptions
	t.Cleanup(func() { probeCompileOptions = orig })

	probeCompileOptions = func(ctx context.Context, db *sql.DB) (map[string]struct{}, error) {
		// Return a non-empty set that lacks ENABLE_FTS5, so the
		// check traverses the "rows iterated fine, flag absent"
		// code path rather than getting a spurious empty result.
		return map[string]struct{}{
			"THREADSAFE=1":        {},
			"ENABLE_MATH_FUNCTIONS": {},
		}, nil
	}

	path := filepath.Join(t.TempDir(), "no-fts5.db")
	db, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatalf("sql.Open: %v", err)
	}
	defer db.Close()

	_, err = NewSQLiteStore(db)
	if err == nil {
		t.Fatal("NewSQLiteStore succeeded; expected ENABLE_FTS5 guard to fire")
	}
	if !strings.Contains(err.Error(), "ENABLE_FTS5") {
		t.Errorf("error = %q, want mention of ENABLE_FTS5", err)
	}

	// Bootstrap must fail before any stackllm_* table exists. Check
	// sqlite_master directly to catch a partial migration that
	// leaked through the guard.
	var count int
	if err := db.QueryRow(
		`SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name LIKE 'stackllm_%'`,
	).Scan(&count); err != nil {
		t.Fatalf("inspect sqlite_master: %v", err)
	}
	if count != 0 {
		t.Errorf("stackllm_* tables created despite guard: %d", count)
	}
}

// ---------------------------------------------------------------------
// List ordering sanity.
// ---------------------------------------------------------------------

func TestSQLiteStore_ListOrderedByUpdated(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)

	s1 := New()
	s1.AppendMessage(conversation.Message{Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "a"}}})
	if err := store.Save(ctx, s1); err != nil {
		t.Fatalf("save s1: %v", err)
	}
	time.Sleep(5 * time.Millisecond)

	s2 := New()
	s2.AppendMessage(conversation.Message{Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "b"}}})
	if err := store.Save(ctx, s2); err != nil {
		t.Fatalf("save s2: %v", err)
	}

	list, err := store.List(ctx)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(list) != 2 {
		t.Fatalf("list len = %d", len(list))
	}
	if list[0].ID != s2.ID {
		t.Errorf("first list entry = %s, want most-recent %s", list[0].ID, s2.ID)
	}
	// List must not hydrate Messages.
	if len(list[0].Messages) != 0 {
		t.Errorf("List populated Messages (%d), should be empty", len(list[0].Messages))
	}
}

// ---------------------------------------------------------------------
// Helpers: extractPreview unit coverage.
// ---------------------------------------------------------------------

func TestExtractPreview_SmallReturnedWhole(t *testing.T) {
	t.Parallel()
	got := extractPreview("short string")
	if got != "short string" {
		t.Errorf("preview = %q", got)
	}
}

func TestExtractPreview_NewlinePulledBack(t *testing.T) {
	t.Parallel()
	// 2048 bytes of content then a newline past the cut then more.
	// Preview should end at or before the last newline within the
	// first 2 KB to avoid showing half a line.
	head := strings.Repeat("a\n", 1000) // 2000 bytes, ending in \n
	body := strings.Repeat("x", 500)
	s := head + body
	got := extractPreview(s)
	if !strings.HasSuffix(got, "\n") {
		t.Errorf("preview should end at a newline boundary: tail=%q", got[len(got)-5:])
	}
	if len(got) > artifactPreviewBytes {
		t.Errorf("preview length %d > threshold", len(got))
	}
}

func TestSHA256Hex_Stable(t *testing.T) {
	t.Parallel()
	a := sha256Hex([]byte("hello"))
	b := sha256Hex([]byte("hello"))
	if a != b {
		t.Error("sha256 not stable")
	}
	if len(a) != 64 {
		t.Errorf("hex length = %d, want 64", len(a))
	}
}

// ---------------------------------------------------------------------
// Regression: state_json must persist across subsequent saves.
// ---------------------------------------------------------------------

// TestSQLiteStore_SaveUpdatesStateJSON pins the contract that SetState
// mutations persist on every save, not only on the initial insert.
// Regression: the UPDATE path in Save used to omit state_json, so any
// state written after the first save silently round-tripped as stale.
func TestSQLiteStore_SaveUpdatesStateJSON(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)

	sess := New()
	sess.SetState("mode", "initial")
	sess.AppendMessage(conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hi"}},
	})
	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("first save: %v", err)
	}

	// Mutate state post-persist and save again. The UPDATE path must
	// now reflect the new value.
	sess.SetState("mode", "updated")
	sess.SetState("counter", 7.0) // json numbers unmarshal as float64
	sess.AppendMessage(conversation.Message{
		Role:   conversation.RoleAssistant,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hello"}},
	})
	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("second save: %v", err)
	}

	loaded, err := store.Load(ctx, sess.ID)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got, _ := loaded.GetState("mode"); got != "updated" {
		t.Errorf("mode = %v, want updated", got)
	}
	if got, _ := loaded.GetState("counter"); got != 7.0 {
		t.Errorf("counter = %v, want 7", got)
	}

	// Delete a key and save a third time — the removed key must
	// disappear from the persisted map, not linger from the old blob.
	delete(sess.State, "counter")
	sess.SetState("mode", "final")
	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("third save: %v", err)
	}
	loaded, err = store.Load(ctx, sess.ID)
	if err != nil {
		t.Fatalf("Load after delete: %v", err)
	}
	if got, _ := loaded.GetState("mode"); got != "final" {
		t.Errorf("mode after delete = %v, want final", got)
	}
	if _, ok := loaded.GetState("counter"); ok {
		t.Errorf("counter should be gone, got present")
	}
}

// TestSQLiteStore_NewSQLiteStore_AppliesWALAtBootstrap pins the
// contract that NewSQLiteStore flips a caller-owned file DB into WAL
// mode at bootstrap, even when the caller's DSN did not request it.
// WAL is file-level and persists across connections, so one PRAGMA at
// bootstrap is enough for the whole pool.
func TestSQLiteStore_NewSQLiteStore_AppliesWALAtBootstrap(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "caller-owned.db")

	// Deliberately omit journal_mode from the caller's DSN — the
	// store must flip it on its own at bootstrap.
	db, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatalf("sql.Open: %v", err)
	}
	defer db.Close()

	// Confirm the DB is NOT in WAL before bootstrap, so the test
	// actually exercises the transition rather than a preset state.
	var before string
	if err := db.QueryRowContext(ctx, `PRAGMA journal_mode`).Scan(&before); err != nil {
		t.Fatalf("pragma journal_mode (before): %v", err)
	}
	if before == "wal" {
		t.Fatalf("file DB already in wal before NewSQLiteStore — test would be vacuous")
	}

	if _, err := NewSQLiteStore(db); err != nil {
		t.Fatalf("NewSQLiteStore: %v", err)
	}

	var after string
	if err := db.QueryRowContext(ctx, `PRAGMA journal_mode`).Scan(&after); err != nil {
		t.Fatalf("pragma journal_mode (after): %v", err)
	}
	if after != "wal" {
		t.Errorf("journal_mode after NewSQLiteStore = %q, want wal", after)
	}
}

// TestSQLiteStore_Begin_AppliesPerTxPragmas verifies that every
// transaction opened by the store applies foreign_keys and
// busy_timeout to its checked-out connection. synchronous cannot be
// set inside a transaction (SQLite rejects it), so NewSQLiteStore
// explicitly leaves it to the caller's DSN — see NewSQLiteStore doc.
func TestSQLiteStore_Begin_AppliesPerTxPragmas(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "pragma-check.db")

	// Open without foreign_keys / busy_timeout in the DSN so begin()'s
	// per-tx application is what's actually under test.
	db, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatalf("sql.Open: %v", err)
	}
	defer db.Close()
	// Pin to one connection so the PRAGMA probes below hit the same
	// connection the transaction used.
	db.SetMaxOpenConns(1)

	store, err := NewSQLiteStore(db)
	if err != nil {
		t.Fatalf("NewSQLiteStore: %v", err)
	}

	conn, tx, err := store.begin(ctx)
	if err != nil {
		t.Fatalf("begin: %v", err)
	}
	defer releaseTx(conn, tx)

	var fk int
	if err := tx.QueryRowContext(ctx, `PRAGMA foreign_keys`).Scan(&fk); err != nil {
		t.Fatalf("pragma foreign_keys: %v", err)
	}
	if fk != 1 {
		t.Errorf("foreign_keys in tx = %d, want 1", fk)
	}

	var busy int
	if err := tx.QueryRowContext(ctx, `PRAGMA busy_timeout`).Scan(&busy); err != nil {
		t.Fatalf("pragma busy_timeout: %v", err)
	}
	if busy != 5000 {
		t.Errorf("busy_timeout in tx = %d, want 5000", busy)
	}
}

// TestSQLiteStore_ListReflectsStateUpdates asserts that the List
// metadata-only path returns freshly-persisted state alongside Load.
// If the UPDATE ever drops state_json again, this will catch it from
// the listing code path too.
func TestSQLiteStore_ListReflectsStateUpdates(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)

	sess := New()
	sess.Name = "listtest"
	sess.SetState("phase", "one")
	sess.AppendMessage(conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hi"}},
	})
	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("first save: %v", err)
	}

	sess.SetState("phase", "two")
	if err := store.Save(ctx, sess); err != nil {
		t.Fatalf("second save: %v", err)
	}

	list, err := store.List(ctx)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	var found *Session
	for _, s := range list {
		if s.ID == sess.ID {
			found = s
			break
		}
	}
	if found == nil {
		t.Fatalf("session %s not in list", sess.ID)
	}
	if got, _ := found.GetState("phase"); got != "two" {
		t.Errorf("List state phase = %v, want two", got)
	}
}
