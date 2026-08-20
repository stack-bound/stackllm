package sqlitestore

import (
	"bytes"
	"context"
	"database/sql"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/session"

	_ "modernc.org/sqlite"
)

// simpleSession returns a saved single-message session for tests that
// need real rows to corrupt or query.
func simpleSession(t *testing.T, store *Store, name string) *session.Session {
	t.Helper()
	sess := session.New()
	sess.Name = name
	sess.AppendMessage(conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hello " + name}},
	})
	if err := store.Save(context.Background(), sess); err != nil {
		t.Fatalf("Save: %v", err)
	}
	return sess
}

func TestStore_DB_ExposesSharedHandle(t *testing.T) {
	t.Parallel()
	store := newFileStore(t)

	db := store.DB()
	if db == nil {
		t.Fatal("DB() returned nil")
	}
	// The handle must be usable against the stackllm schema: the version
	// row written by migrations is the ground truth.
	var version int
	if err := db.QueryRow(`SELECT version FROM stackllm_schema_version`).Scan(&version); err != nil {
		t.Fatalf("query schema version via DB(): %v", err)
	}
	if version != latestSchemaVersion {
		t.Errorf("schema version = %d, want %d", version, latestSchemaVersion)
	}
}

func TestTruncatePreview(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		in   string
		n    int
		want string
	}{
		{name: "short returned whole", in: "abc", n: 10, want: "abc"},
		{name: "exactly at limit", in: "abcde", n: 5, want: "abcde"},
		{name: "newlines collapsed to spaces", in: "a\nb\nc", n: 10, want: "a b c"},
		{name: "ascii over limit cut at n runes", in: "abcdefgh", n: 4, want: "abcd"},
		{
			// 4 runes but 8 bytes: byte length exceeds n while rune
			// length does not — must be returned whole, not sliced.
			name: "multibyte within rune budget",
			in:   "日本語x",
			n:    5,
			want: "日本語x",
		},
		{
			// 6 runes, n=4: cut must land on a rune boundary.
			name: "multibyte over rune budget",
			in:   "日本語です!",
			n:    4,
			want: "日本語で",
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := truncatePreview(tt.in, tt.n)
			if got != tt.want {
				t.Errorf("truncatePreview(%q, %d) = %q, want %q", tt.in, tt.n, got, tt.want)
			}
			if !utf8.ValidString(got) {
				t.Errorf("truncatePreview produced invalid UTF-8: %q", got)
			}
		})
	}
}

func TestExtractPreview_MultibyteBoundary(t *testing.T) {
	t.Parallel()

	// A 3-byte rune straddles the preview threshold: bytes
	// artifactPreviewBytes-1 .. artifactPreviewBytes+1. The cut at
	// artifactPreviewBytes lands mid-rune and must be pulled back.
	s := strings.Repeat("a", artifactPreviewBytes-1) + "世" + strings.Repeat("b", 10)

	got := extractPreview(s)
	if !utf8.ValidString(got) {
		t.Fatalf("preview is not valid UTF-8 (%d bytes)", len(got))
	}
	want := strings.Repeat("a", artifactPreviewBytes-1)
	if got != want {
		t.Errorf("preview length = %d, want %d ascii bytes (cut pulled back before straddling rune)", len(got), len(want))
	}
}

func TestStore_ClosedDB_ErrorPaths(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	path := filepath.Join(t.TempDir(), "closed.db")
	store, err := Open(Config{Path: path})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	sess := simpleSession(t, store, "pre-close")
	if err := store.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	tests := []struct {
		name string
		op   func() error
	}{
		{"Save", func() error { return store.Save(ctx, sess) }},
		{"Load", func() error { _, err := store.Load(ctx, sess.ID); return err }},
		{"Delete", func() error { return store.Delete(ctx, sess.ID) }},
		{"List", func() error { _, err := store.List(ctx); return err }},
		{"ListPage", func() error { _, err := store.ListPage(ctx, session.ListOptions{}); return err }},
		{"Fork", func() error { _, err := store.Fork(ctx, sess.ID, sess.Messages[0].ID, "f"); return err }},
		{"Rewind", func() error { return store.Rewind(ctx, sess.ID, sess.Messages[0].ID) }},
		{"ListBranches", func() error { _, err := store.ListBranches(ctx, sess.ID, ""); return err }},
		{"Search", func() error { _, err := store.Search(ctx, "hello", "", nil, 10); return err }},
		{"ExportJSONL", func() error { return store.ExportJSONL(ctx, sess.ID, &bytes.Buffer{}) }},
		{"ImportJSONL", func() error {
			_, err := store.ImportJSONL(ctx, strings.NewReader(`{"kind":"session_header","id":"x"}`))
			return err
		}},
		{"HydrateArtifact", func() error { _, _, err := store.HydrateArtifact(ctx, "some-id"); return err }},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.op(); err == nil {
				t.Errorf("%s on closed DB succeeded, want error", tt.name)
			}
		})
	}
}

func TestSave_Validation(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)

	if err := store.Save(ctx, nil); err == nil || !strings.Contains(err.Error(), "nil session") {
		t.Errorf("Save(nil) = %v, want nil-session error", err)
	}
	if err := store.Save(ctx, &session.Session{}); err == nil || !strings.Contains(err.Error(), "no ID") {
		t.Errorf("Save(no ID) = %v, want no-ID error", err)
	}
}

func TestSave_UnmarshalableStateFails(t *testing.T) {
	t.Parallel()
	store := newFileStore(t)

	sess := session.New()
	sess.SetState("bad", make(chan int))
	err := store.Save(context.Background(), sess)
	if err == nil || !strings.Contains(err.Error(), "marshal state") {
		t.Errorf("Save with channel in state = %v, want marshal state error", err)
	}
}

func TestLoad_Validation(t *testing.T) {
	t.Parallel()
	store := newFileStore(t)

	if _, err := store.Load(context.Background(), ""); err == nil || !strings.Contains(err.Error(), "empty id") {
		t.Errorf("Load(\"\") = %v, want empty-id error", err)
	}
	if _, err := store.Load(context.Background(), "does-not-exist"); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Errorf("Load(unknown) = %v, want not-found error", err)
	}
}

func TestLoad_CorruptStateJSON(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)
	sess := simpleSession(t, store, "corrupt-state")

	if _, err := store.DB().Exec(
		`UPDATE stackllm_sessions SET state_json = '{corrupt' WHERE id = ?`, sess.ID,
	); err != nil {
		t.Fatalf("corrupt state_json: %v", err)
	}

	_, err := store.Load(ctx, sess.ID)
	if err == nil || !strings.Contains(err.Error(), "state") {
		t.Errorf("Load with corrupt state = %v, want state error", err)
	}
}

func TestLoad_UnparseableTimestampsYieldZeroTimes(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)
	sess := simpleSession(t, store, "bad-times")

	// One unparseable value and one empty value cover both parseTime
	// fallback branches.
	if _, err := store.DB().Exec(
		`UPDATE stackllm_sessions SET created_at = 'yesterday-ish', updated_at = '' WHERE id = ?`, sess.ID,
	); err != nil {
		t.Fatalf("corrupt timestamps: %v", err)
	}

	loaded, err := store.Load(ctx, sess.ID)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !loaded.Created.IsZero() {
		t.Errorf("Created = %v, want zero time for unparseable value", loaded.Created)
	}
	if !loaded.Updated.IsZero() {
		t.Errorf("Updated = %v, want zero time for empty value", loaded.Updated)
	}
}

func TestNew_SchemaVersionReadError(t *testing.T) {
	t.Parallel()

	db := newSharedDB(t)
	// A version table without a version column produces a query error
	// that is NOT "no such table" — bootstrap must fail loudly.
	if _, err := db.Exec(`CREATE TABLE stackllm_schema_version(other INTEGER)`); err != nil {
		t.Fatalf("create decoy table: %v", err)
	}

	_, err := New(db)
	if err == nil || !strings.Contains(err.Error(), "read schema version") {
		t.Errorf("New = %v, want read schema version error", err)
	}
}

func TestNew_ClosedDBFailsBootstrap(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "x.db")
	db, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatalf("sql.Open: %v", err)
	}
	db.Close()

	if _, err := New(db); err == nil {
		t.Error("New with closed DB should fail bootstrap")
	}
}

func TestOpen_DataDirCreationFails(t *testing.T) {
	t.Parallel()

	blocker := filepath.Join(t.TempDir(), "blocker")
	if err := os.WriteFile(blocker, []byte("x"), 0o600); err != nil {
		t.Fatalf("write blocker: %v", err)
	}

	_, err := Open(Config{Path: filepath.Join(blocker, "sub", "state.db")})
	if err == nil || !strings.Contains(err.Error(), "create data dir") {
		t.Errorf("Open = %v, want create data dir error", err)
	}
}

func TestConfig_ResolvePath_HomeFallback(t *testing.T) {
	// Mutates env — cannot be parallel.
	t.Setenv("XDG_DATA_HOME", "")

	got, err := Config{AppName: "coverapp"}.resolvePath()
	if err != nil {
		t.Fatalf("resolvePath: %v", err)
	}
	want := filepath.Join(".local", "share", "coverapp", "state.db")
	if !strings.HasSuffix(got, want) {
		t.Errorf("resolvePath = %q, want suffix %q", got, want)
	}

	// With no home dir either, resolution must fail.
	t.Setenv("HOME", "")
	if _, err := (Config{AppName: "coverapp"}).resolvePath(); err == nil {
		t.Error("resolvePath with no XDG_DATA_HOME and no HOME should fail")
	}
}

func TestHydrateArtifact_NotFound(t *testing.T) {
	t.Parallel()
	store := newFileStore(t)

	_, _, err := store.HydrateArtifact(context.Background(), "no-such-artifact")
	if err == nil || !strings.Contains(err.Error(), "not found") {
		t.Errorf("HydrateArtifact = %v, want not-found error", err)
	}
}

func TestBranching_UnknownTargets(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)
	sess := simpleSession(t, store, "branching")

	if _, err := store.Fork(ctx, "no-such-session", sess.Messages[0].ID, "f"); err == nil {
		t.Error("Fork with unknown session should fail")
	}
	if _, err := store.Fork(ctx, sess.ID, "no-such-message", "f"); err == nil {
		t.Error("Fork with unknown message should fail")
	}
	if err := store.Rewind(ctx, sess.ID, "no-such-message"); err == nil {
		t.Error("Rewind to unknown message should fail")
	}
	if err := store.Rewind(ctx, "no-such-session", sess.Messages[0].ID); err == nil {
		t.Error("Rewind on unknown session should fail")
	}
}

func TestSearch_InvalidFTSQuery(t *testing.T) {
	t.Parallel()
	store := newFileStore(t)
	simpleSession(t, store, "searchable")

	_, err := store.Search(context.Background(), `"unbalanced`, "", nil, 10)
	if err == nil {
		t.Error("expected FTS5 syntax error for unbalanced quote")
	}
}

func TestSearch_DefaultLimit(t *testing.T) {
	t.Parallel()
	store := newFileStore(t)
	sess := simpleSession(t, store, "limits")

	hits, err := store.Search(context.Background(), "hello", "", nil, 0)
	if err != nil {
		t.Fatalf("Search with zero limit: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("expected at least one hit with default limit")
	}
	if hits[0].SessionID != sess.ID {
		t.Errorf("hit session = %q, want %q", hits[0].SessionID, sess.ID)
	}
}

func TestExportJSONL_UnknownSession(t *testing.T) {
	t.Parallel()
	store := newFileStore(t)

	err := store.ExportJSONL(context.Background(), "no-such-session", &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "not found") {
		t.Errorf("ExportJSONL = %v, want not-found error", err)
	}
}

func TestImportJSONL_MalformedInput(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)

	if _, err := store.ImportJSONL(ctx, strings.NewReader("not json at all")); err == nil {
		t.Error("expected error for non-JSON input")
	}
	if _, err := store.ImportJSONL(ctx, strings.NewReader(`{"kind":"message"}`)); err == nil || !strings.Contains(err.Error(), "session_header") {
		t.Errorf("error = %v, want session_header error", err)
	}
}

func TestListPage_NegativeLimitAndOffset(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newFileStore(t)
	for _, name := range []string{"one", "two", "three"} {
		simpleSession(t, store, name)
	}

	// Negative offset is normalised to 0; negative limit returns all.
	res, err := store.ListPage(ctx, session.ListOptions{Limit: -1, Offset: -9})
	if err != nil {
		t.Fatalf("ListPage: %v", err)
	}
	if res.Total != 3 || len(res.Sessions) != 3 {
		t.Errorf("Total=%d Sessions=%d, want 3/3", res.Total, len(res.Sessions))
	}

	// Negative limit with a positive offset uses the LIMIT -1 branch.
	res, err = store.ListPage(ctx, session.ListOptions{Limit: -1, Offset: 1})
	if err != nil {
		t.Fatalf("ListPage: %v", err)
	}
	if res.Total != 3 || len(res.Sessions) != 2 {
		t.Errorf("Total=%d Sessions=%d, want Total 3 with 2 sessions after offset", res.Total, len(res.Sessions))
	}
}
