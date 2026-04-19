// Package session — SQLite-backed persistent SessionStore.
//
// SQLiteStore implements SessionStore against a pure-Go SQLite build
// (modernc.org/sqlite), so embedders get durable session history
// without a separately-installed database or CGO. The schema matches
// the block-oriented conversation model introduced in Phase 1: every
// message row is a tree node with a parent_id, and every block row is
// one element of a message's ordered Blocks slice.
//
// Parent applications can safely create their own tables in the same
// SQLite file; all stackllm-owned tables are prefixed `stackllm_` and
// a parent app that avoids that prefix is guaranteed not to collide.
// See NewSQLiteStore for the shared-DB pattern.
package session

import (
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/stack-bound/stackllm/conversation"

	_ "modernc.org/sqlite"
)

// SQLiteConfig configures OpenSQLiteStore. Exactly one of AppName or
// Path must be set. Forgetting to set either is an error — stackllm
// never writes to a silent default path because two embedders using
// the same default would collide.
type SQLiteConfig struct {
	// AppName auto-builds $XDG_DATA_HOME/{AppName}/state.db.
	// Ignored if Path is set.
	AppName string

	// Path is an explicit file path for the database. Takes
	// precedence over AppName.
	Path string
}

// resolvePath turns the config into a concrete filesystem path.
func (cfg SQLiteConfig) resolvePath() (string, error) {
	if cfg.Path != "" {
		return cfg.Path, nil
	}
	if cfg.AppName == "" {
		return "", errors.New("session: SQLiteConfig requires either AppName or Path")
	}
	dir := os.Getenv("XDG_DATA_HOME")
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("session: resolve home: %w", err)
		}
		dir = filepath.Join(home, ".local", "share")
	}
	return filepath.Join(dir, cfg.AppName, "state.db"), nil
}

// SQLiteStore is the SQLite-backed SessionStore. Construct it with
// OpenSQLiteStore for simple embedders or NewSQLiteStore for parent
// apps that already own a *sql.DB.
type SQLiteStore struct {
	db    *sql.DB
	owned bool // true if OpenSQLiteStore allocated the DB — controls Close semantics
}

// BranchRef identifies one branch from a branch point in a session's
// message tree. Returned by ListBranches.
type BranchRef struct {
	MessageID string
	Role      conversation.Role
	CreatedAt time.Time
	// Preview is the first 80 characters of the first text-bearing
	// block in the message, intended for selection UIs.
	Preview string
}

// SearchHit is one match from a full-text search. Snippet is FTS5's
// snippet() output (with ellipses around the matched term).
type SearchHit struct {
	SessionID string
	MessageID string
	BlockID   string
	BlockType conversation.BlockType
	Snippet   string
	Rank      float64
}

// OpenSQLiteStore opens or creates a SQLite-backed store at the
// configured path, applies pragmas, runs migrations, and returns a
// ready-to-use store. The parent directory is created on demand.
//
// Callers that already own a *sql.DB (for example, to share a
// connection pool with parent-app tables in the same file) should use
// NewSQLiteStore instead.
func OpenSQLiteStore(cfg SQLiteConfig) (*SQLiteStore, error) {
	path, err := cfg.resolvePath()
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, fmt.Errorf("session: create data dir: %w", err)
	}

	// DSN-level pragmas apply on every new connection in the pool.
	// WAL + NORMAL sync is the standard "durable but fast" config for
	// SQLite; busy_timeout lets concurrent writers wait briefly
	// instead of returning SQLITE_BUSY immediately.
	dsn := buildDSN(path)

	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("session: open sqlite: %w", err)
	}

	store := &SQLiteStore{db: db, owned: true}
	if err := store.bootstrap(context.Background()); err != nil {
		db.Close()
		return nil, err
	}
	return store, nil
}

// NewSQLiteStore wraps an already-open *sql.DB. The caller owns the
// DB's lifecycle; Close() on the returned store is a no-op.
//
// Pragma semantics for caller-owned pools:
//   - journal_mode=WAL is applied once during bootstrap. WAL mode is a
//     file-level setting that persists across connections and across
//     process restarts, so a single PRAGMA is enough to give the whole
//     pool the concurrency guarantees OpenSQLiteStore promises. No-op
//     for :memory: (SQLite silently leaves in-memory databases in
//     MEMORY mode).
//   - foreign_keys=ON and busy_timeout=5000 are applied on every
//     transaction begin() opens, so any connection the pool checks out
//     for a stackllm transaction inherits cascade-delete and
//     busy-waiting behavior matching OpenSQLiteStore.
//   - synchronous=NORMAL CANNOT be changed from inside a transaction
//     (SQLite rejects it), and a caller-owned *sql.DB may check out a
//     different connection for every statement. stackllm therefore
//     does not touch synchronous for caller-owned pools — the
//     SQLite default (FULL) applies unless the caller sets NORMAL in
//     their own DSN. If you want the same "durable but fast" profile
//     as OpenSQLiteStore, add _pragma=synchronous(normal) to the DSN
//     you pass to sql.Open. See examples/sqlite for the full shape.
//
// IMPORTANT: parent apps that share the DB must not create tables
// using the `stackllm_` prefix; that namespace is reserved.
func NewSQLiteStore(db *sql.DB) (*SQLiteStore, error) {
	store := &SQLiteStore{db: db, owned: false}
	if err := store.bootstrap(context.Background()); err != nil {
		return nil, err
	}
	return store, nil
}

// buildDSN produces a modernc.org/sqlite DSN with pragmas encoded so
// every pooled connection gets the same settings. Pragmas set via the
// DSN are applied on every new connection; pragmas set via db.Exec
// only affect whichever connection handled that Exec.
func buildDSN(path string) string {
	params := url.Values{}
	params.Add("_pragma", "foreign_keys(1)")
	params.Add("_pragma", "journal_mode(wal)")
	params.Add("_pragma", "synchronous(normal)")
	params.Add("_pragma", "busy_timeout(5000)")
	return "file:" + path + "?" + params.Encode()
}

// bootstrap runs the FTS5 availability check, applies the one-shot
// file-level pragmas (journal_mode=WAL), and runs migrations. Called
// from both OpenSQLiteStore and NewSQLiteStore so shared-DB callers
// get the same schema and concurrency guarantees as standalone ones.
func (s *SQLiteStore) bootstrap(ctx context.Context) error {
	if err := checkFTS5Available(ctx, s.db); err != nil {
		return err
	}
	if err := applyFileLevelPragmas(ctx, s.db); err != nil {
		return err
	}
	if err := runMigrations(ctx, s.db); err != nil {
		return err
	}
	return nil
}

// applyFileLevelPragmas writes pragmas that persist at the database
// file level (so a single PRAGMA affects every future connection).
// Today that is just journal_mode=WAL. Synchronous and busy_timeout
// are per-connection and are applied in begin() instead.
//
// SQLite silently ignores journal_mode=WAL on :memory: databases
// (they always run in MEMORY mode), so this is a no-op there rather
// than an error.
func applyFileLevelPragmas(ctx context.Context, db *sql.DB) error {
	var mode string
	if err := db.QueryRowContext(ctx, `PRAGMA journal_mode = WAL`).Scan(&mode); err != nil {
		return fmt.Errorf("session: set journal_mode=wal: %w", err)
	}
	return nil
}

// probeCompileOptions returns the set of compile_options reported by
// the underlying SQLite build. Exposed as a package variable so tests
// can simulate a build that is missing ENABLE_FTS5 without shipping a
// second SQLite driver.
var probeCompileOptions = defaultProbeCompileOptions

func defaultProbeCompileOptions(ctx context.Context, db *sql.DB) (map[string]struct{}, error) {
	rows, err := db.QueryContext(ctx, `PRAGMA compile_options`)
	if err != nil {
		return nil, fmt.Errorf("session: probe compile_options: %w", err)
	}
	defer rows.Close()
	out := make(map[string]struct{})
	for rows.Next() {
		var opt string
		if err := rows.Scan(&opt); err != nil {
			return nil, fmt.Errorf("session: scan compile_option: %w", err)
		}
		out[opt] = struct{}{}
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("session: iterate compile_options: %w", err)
	}
	return out, nil
}

// checkFTS5Available verifies the underlying SQLite build includes
// ENABLE_FTS5. modernc.org/sqlite ships the full amalgamation today,
// but we fail fast with a clear message rather than letting a CREATE
// VIRTUAL TABLE error surface mid-migration.
func checkFTS5Available(ctx context.Context, db *sql.DB) error {
	opts, err := probeCompileOptions(ctx, db)
	if err != nil {
		return err
	}
	if _, ok := opts["ENABLE_FTS5"]; ok {
		return nil
	}
	return errors.New("session: SQLite build is missing ENABLE_FTS5 — stackllm requires FTS5 for block search")
}

// runMigrations applies pending schema migrations in a single
// transaction each. It's idempotent: running against an up-to-date
// DB does nothing.
func runMigrations(ctx context.Context, db *sql.DB) error {
	// Determine current version. If the table doesn't exist we're at
	// version 0 and run every migration.
	var current int
	err := db.QueryRowContext(ctx, `SELECT version FROM stackllm_schema_version`).Scan(&current)
	switch {
	case err == sql.ErrNoRows:
		current = 0
	case err != nil:
		// "no such table" is expected on a fresh DB; any other error
		// means something is actually wrong.
		if !strings.Contains(err.Error(), "no such table") {
			return fmt.Errorf("session: read schema version: %w", err)
		}
		current = 0
	}

	if current > latestSchemaVersion {
		return fmt.Errorf("session: database schema version %d is newer than supported version %d — upgrade stackllm to read this database", current, latestSchemaVersion)
	}

	for v := current; v < latestSchemaVersion; v++ {
		tx, err := db.BeginTx(ctx, nil)
		if err != nil {
			return fmt.Errorf("session: begin migration %d: %w", v+1, err)
		}
		if _, err := tx.ExecContext(ctx, migrations[v]); err != nil {
			tx.Rollback()
			return fmt.Errorf("session: apply migration %d: %w", v+1, err)
		}
		// Upsert the version row. A fresh DB has no row; upgrades
		// have exactly one.
		if _, err := tx.ExecContext(ctx, `DELETE FROM stackllm_schema_version`); err != nil {
			tx.Rollback()
			return fmt.Errorf("session: clear schema version: %w", err)
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO stackllm_schema_version(version) VALUES(?)`, v+1); err != nil {
			tx.Rollback()
			return fmt.Errorf("session: record schema version %d: %w", v+1, err)
		}
		if err := tx.Commit(); err != nil {
			return fmt.Errorf("session: commit migration %d: %w", v+1, err)
		}
	}
	return nil
}

// begin reserves a single connection from the pool, applies the
// per-connection pragmas OUTSIDE any transaction (where SQLite
// actually honors them), then starts the transaction on that same
// connection. The returned conn must be released by the caller after
// the transaction finishes — the idiom is:
//
//	conn, tx, err := s.begin(ctx)
//	if err != nil { ... }
//	defer releaseTx(conn, tx)
//
// Why the reservation: foreign_keys is documented as a no-op when set
// inside a transaction, so the previous shape ("BeginTx, then PRAGMA
// foreign_keys") silently did nothing for caller-owned pools whose
// DSN did not already set the pragma. Reserving a conn lets us apply
// foreign_keys and busy_timeout before BeginTx, guaranteeing
// cascade-delete and busy-wait semantics match OpenSQLiteStore for
// both store types.
//
// synchronous cannot be changed inside a transaction at all and we
// deliberately do not touch it here either — see the NewSQLiteStore
// doc comment for the rationale.
func (s *SQLiteStore) begin(ctx context.Context) (*sql.Conn, *sql.Tx, error) {
	conn, err := s.db.Conn(ctx)
	if err != nil {
		return nil, nil, fmt.Errorf("session: reserve connection: %w", err)
	}
	if _, err := conn.ExecContext(ctx, `PRAGMA foreign_keys = ON`); err != nil {
		conn.Close()
		return nil, nil, fmt.Errorf("session: enable foreign keys: %w", err)
	}
	if _, err := conn.ExecContext(ctx, `PRAGMA busy_timeout = 5000`); err != nil {
		conn.Close()
		return nil, nil, fmt.Errorf("session: set busy_timeout: %w", err)
	}
	tx, err := conn.BeginTx(ctx, nil)
	if err != nil {
		conn.Close()
		return nil, nil, fmt.Errorf("session: begin tx: %w", err)
	}
	return conn, tx, nil
}

// releaseTx rolls the transaction back (a no-op after Commit) and
// releases the reserved connection back to the pool. Call it with
// defer right after a successful begin() call.
func releaseTx(conn *sql.Conn, tx *sql.Tx) {
	if tx != nil {
		_ = tx.Rollback()
	}
	if conn != nil {
		_ = conn.Close()
	}
}

// DB exposes the underlying *sql.DB so parent apps that used
// OpenSQLiteStore can run their own queries against the same file.
// Parent-app tables MUST NOT use the stackllm_ prefix.
func (s *SQLiteStore) DB() *sql.DB {
	return s.db
}

// Close closes the underlying *sql.DB iff the store opened it.
// Stores created via NewSQLiteStore do not own the DB and Close is a
// no-op in that case.
func (s *SQLiteStore) Close() error {
	if !s.owned {
		return nil
	}
	return s.db.Close()
}

// ---------------------------------------------------------------------
// Save: diff-and-append with block granularity.
// ---------------------------------------------------------------------

// Save reconciles sess.Messages against the DB, appending anything
// new at the tail. The input is assumed to be the full linear history
// from the root to the current leaf — the same shape InMemoryStore
// accepts today and the same shape adapters pass back from
// agent.Run.
//
// Save is append-only. Editing a persisted block's Text in memory and
// calling Save again has no effect: the block's ID already exists in
// the DB and the in-memory mutation is silently dropped. Mutations
// must go through Rewind + a new message, or through Fork.
func (s *SQLiteStore) Save(ctx context.Context, sess *Session) error {
	if sess == nil {
		return errors.New("session: Save: nil session")
	}
	if sess.ID == "" {
		return errors.New("session: Save: session has no ID")
	}

	// Every message and block must have an ID before Save runs. This
	// matches the Phase 1 eager-ID invariant — producers mint IDs at
	// construction time. Persistence state is derived from the DB,
	// not from ID presence.
	for i := range sess.Messages {
		conversation.EnsureMessageIDs(&sess.Messages[i])
	}

	conn, tx, err := s.begin(ctx)
	if err != nil {
		return fmt.Errorf("session: Save: begin: %w", err)
	}
	defer releaseTx(conn, tx)

	now := time.Now().UTC()

	// Marshal state up front so both the insert and the trailing UPDATE
	// below use the same serialized value. Prior to this hoist the
	// UPDATE branch omitted state_json entirely and silently dropped
	// any SetState mutations after the first save.
	stateJSON, err := marshalState(sess.State)
	if err != nil {
		return fmt.Errorf("session: Save: marshal state: %w", err)
	}

	// Upsert the session row.
	var exists int
	if err := tx.QueryRowContext(ctx, `SELECT 1 FROM stackllm_sessions WHERE id = ?`, sess.ID).Scan(&exists); err != nil {
		if err != sql.ErrNoRows {
			return fmt.Errorf("session: Save: probe session: %w", err)
		}
		// Insert new session. metadata_json is reserved for future
		// stackllm-owned fields and starts as an empty object; state
		// carries the caller's KV map.
		created := sess.Created
		if created.IsZero() {
			created = now
		}
		promptTok, completionTok, totalTok := usageCols(sess.LastUsage)
		if _, err := tx.ExecContext(ctx, `
			INSERT INTO stackllm_sessions(
				id, name, project_path, model,
				metadata_json, state_json, current_leaf_id,
				created_at, updated_at,
				last_prompt_tokens, last_completion_tokens, last_total_tokens
			) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)`,
			sess.ID,
			nullString(sess.Name),
			nullString(sess.ProjectPath),
			nullString(sess.Model),
			"{}",
			stateJSON,
			formatTime(created),
			formatTime(now),
			promptTok, completionTok, totalTok,
		); err != nil {
			return fmt.Errorf("session: Save: insert session row: %w", err)
		}
		sess.Created = created
	}

	// Collect all message / block IDs for bulk existence lookup.
	msgIDs := make([]string, 0, len(sess.Messages))
	blockIDs := make([]string, 0, len(sess.Messages)*4)
	for _, m := range sess.Messages {
		msgIDs = append(msgIDs, m.ID)
		for _, b := range m.Blocks {
			blockIDs = append(blockIDs, b.ID)
		}
	}

	existingMsgs, existingParents, err := loadExistingMessages(ctx, tx, sess.ID, msgIDs)
	if err != nil {
		return fmt.Errorf("session: Save: lookup existing messages: %w", err)
	}
	existingBlocks, err := loadExistingBlockIDs(ctx, tx, blockIDs)
	if err != nil {
		return fmt.Errorf("session: Save: lookup existing blocks: %w", err)
	}

	// Walk the in-memory chain forward, validating existing rows and
	// inserting new ones. runningParent starts at NULL (the root has
	// no parent) and advances to the previous message's ID as we
	// walk.
	var runningParent sql.NullString
	for mi := range sess.Messages {
		msg := &sess.Messages[mi]

		if _, ok := existingMsgs[msg.ID]; ok {
			// Validate that the persisted parent matches the chain
			// we're walking. A mismatch means the in-memory history
			// diverged from the DB — typically a caller bug.
			dbParent := existingParents[msg.ID]
			if !parentMatches(dbParent, runningParent) {
				return fmt.Errorf("session: save: message %s parent chain diverged from DB", msg.ID)
			}
			// Append any newly-appended blocks at the tail. This is
			// the rare case where a persisted message grew (e.g. a
			// tool-role message gained an extra tool_result).
			if err := appendNewBlocksToExistingMessage(ctx, tx, msg, existingBlocks, now); err != nil {
				return fmt.Errorf("session: Save: extend message %s: %w", msg.ID, err)
			}
			runningParent = sql.NullString{String: msg.ID, Valid: true}
			continue
		}

		// Insert the new message row.
		createdAt := msg.CreatedAt
		if createdAt.IsZero() {
			createdAt = now
		}
		var durationMS sql.NullInt64
		if msg.Duration > 0 {
			durationMS = sql.NullInt64{Int64: msg.Duration.Milliseconds(), Valid: true}
		}
		if _, err := tx.ExecContext(ctx, `
			INSERT INTO stackllm_messages(
				id, session_id, parent_id, role, model, duration_ms, created_at
			) VALUES (?, ?, ?, ?, ?, ?, ?)`,
			msg.ID,
			sess.ID,
			runningParent,
			string(msg.Role),
			nullString(msg.Model),
			durationMS,
			formatTime(createdAt),
		); err != nil {
			return fmt.Errorf("session: Save: insert message %s: %w", msg.ID, err)
		}

		// Insert all blocks in order. Seq is the 0-based position.
		for bi := range msg.Blocks {
			if err := insertBlock(ctx, tx, msg.ID, bi, &msg.Blocks[bi], now); err != nil {
				return fmt.Errorf("session: Save: insert block %s: %w", msg.Blocks[bi].ID, err)
			}
		}

		runningParent = sql.NullString{String: msg.ID, Valid: true}
	}

	// Update current_leaf_id + updated_at + mutable metadata + state.
	// state_json MUST be written here (not only on insert) so SetState
	// mutations after the first save actually persist. LastUsage is
	// written here too so a caller that updates the field in memory
	// and saves without a new turn still round-trips the new value.
	promptTok, completionTok, totalTok := usageCols(sess.LastUsage)
	if _, err := tx.ExecContext(ctx, `
		UPDATE stackllm_sessions
		SET current_leaf_id = ?, updated_at = ?, name = ?, project_path = ?, model = ?, state_json = ?,
		    last_prompt_tokens = ?, last_completion_tokens = ?, last_total_tokens = ?
		WHERE id = ?`,
		runningParent,
		formatTime(now),
		nullString(sess.Name),
		nullString(sess.ProjectPath),
		nullString(sess.Model),
		stateJSON,
		promptTok, completionTok, totalTok,
		sess.ID,
	); err != nil {
		return fmt.Errorf("session: Save: update session row: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("session: Save: commit: %w", err)
	}
	sess.Updated = now
	return nil
}

// loadExistingMessages returns the IDs of the given messages that
// already exist for the given session, along with their persisted
// parent_id values. Used by Save to decide insert-vs-validate.
func loadExistingMessages(ctx context.Context, tx *sql.Tx, sessionID string, ids []string) (map[string]struct{}, map[string]sql.NullString, error) {
	if len(ids) == 0 {
		return map[string]struct{}{}, map[string]sql.NullString{}, nil
	}
	// Chunk to stay safely below SQLite's default 999-parameter cap
	// while leaving room for the session_id bind.
	const chunk = 500
	existing := make(map[string]struct{}, len(ids))
	parents := make(map[string]sql.NullString, len(ids))
	for start := 0; start < len(ids); start += chunk {
		end := start + chunk
		if end > len(ids) {
			end = len(ids)
		}
		batch := ids[start:end]
		q := "SELECT id, parent_id FROM stackllm_messages WHERE session_id = ? AND id IN (" + placeholders(len(batch)) + ")"
		args := make([]any, 0, len(batch)+1)
		args = append(args, sessionID)
		for _, id := range batch {
			args = append(args, id)
		}
		rows, err := tx.QueryContext(ctx, q, args...)
		if err != nil {
			return nil, nil, err
		}
		for rows.Next() {
			var id string
			var parent sql.NullString
			if err := rows.Scan(&id, &parent); err != nil {
				rows.Close()
				return nil, nil, err
			}
			existing[id] = struct{}{}
			parents[id] = parent
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return nil, nil, err
		}
		rows.Close()
	}
	return existing, parents, nil
}

// loadExistingBlockIDs returns the set of block IDs that already
// exist in stackllm_blocks. Used by Save and by the "extend existing
// message with new blocks" path.
func loadExistingBlockIDs(ctx context.Context, tx *sql.Tx, ids []string) (map[string]struct{}, error) {
	if len(ids) == 0 {
		return map[string]struct{}{}, nil
	}
	const chunk = 500
	existing := make(map[string]struct{}, len(ids))
	for start := 0; start < len(ids); start += chunk {
		end := start + chunk
		if end > len(ids) {
			end = len(ids)
		}
		batch := ids[start:end]
		q := "SELECT id FROM stackllm_blocks WHERE id IN (" + placeholders(len(batch)) + ")"
		args := make([]any, 0, len(batch))
		for _, id := range batch {
			args = append(args, id)
		}
		rows, err := tx.QueryContext(ctx, q, args...)
		if err != nil {
			return nil, err
		}
		for rows.Next() {
			var id string
			if err := rows.Scan(&id); err != nil {
				rows.Close()
				return nil, err
			}
			existing[id] = struct{}{}
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return nil, err
		}
		rows.Close()
	}
	return existing, nil
}

// appendNewBlocksToExistingMessage handles the rare case where a
// message that was already saved grew new blocks between saves (for
// example, a tool-role message that gained an extra tool_result).
// Existing blocks are left alone — the DB is the source of truth.
func appendNewBlocksToExistingMessage(ctx context.Context, tx *sql.Tx, msg *conversation.Message, existingBlocks map[string]struct{}, now time.Time) error {
	// Look up the current max seq for this message so appended
	// blocks get contiguous positions after the ones already stored.
	var maxSeq sql.NullInt64
	if err := tx.QueryRowContext(ctx,
		`SELECT MAX(seq) FROM stackllm_blocks WHERE message_id = ?`,
		msg.ID,
	).Scan(&maxSeq); err != nil {
		return err
	}
	next := int(maxSeq.Int64) + 1
	if !maxSeq.Valid {
		next = 0
	}

	for bi := range msg.Blocks {
		if _, ok := existingBlocks[msg.Blocks[bi].ID]; ok {
			continue
		}
		if err := insertBlock(ctx, tx, msg.ID, next, &msg.Blocks[bi], now); err != nil {
			return err
		}
		next++
	}
	return nil
}

// insertBlock writes one conversation.Block into stackllm_blocks at
// the given seq. Large payloads (images with inline data, redacted
// thinking, oversized tool_result text) are offloaded to
// stackllm_artifacts and referenced by artifact_id; the block row
// keeps a preview in text_content for scrollback and FTS.
func insertBlock(ctx context.Context, tx *sql.Tx, msgID string, seq int, b *conversation.Block, now time.Time) error {
	if b.ID == "" {
		b.ID = conversation.NewID()
	}

	var (
		textContent  sql.NullString
		toolCallID   sql.NullString
		toolName     sql.NullString
		toolArgsJSON sql.NullString
		toolIsErr    sql.NullInt64
		mimeType     sql.NullString
		imageURL     sql.NullString
		artifactID   sql.NullString
	)

	switch b.Type {
	case conversation.BlockText, conversation.BlockThinking:
		textContent = sql.NullString{String: b.Text, Valid: true}

	case conversation.BlockToolUse:
		toolCallID = nullString(b.ToolCallID)
		toolName = nullString(b.ToolName)
		toolArgsJSON = nullString(b.ToolArgsJSON)

	case conversation.BlockToolResult:
		toolCallID = nullString(b.ToolCallID)
		if b.ToolIsError {
			toolIsErr = sql.NullInt64{Int64: 1, Valid: true}
		} else {
			toolIsErr = sql.NullInt64{Int64: 0, Valid: true}
		}
		if shouldOffloadText(len(b.Text)) {
			// Offload: store the full payload as a text/plain
			// artifact, keep a 2 KB preview inline for FTS and
			// scrollback.
			id, err := upsertArtifact(ctx, tx, []byte(b.Text), "text/plain", now)
			if err != nil {
				return err
			}
			artifactID = sql.NullString{String: id, Valid: true}
			textContent = sql.NullString{String: extractPreview(b.Text), Valid: true}
			mimeType = sql.NullString{String: "text/plain", Valid: true}
		} else {
			textContent = sql.NullString{String: b.Text, Valid: true}
		}

	case conversation.BlockImage:
		mimeType = nullString(b.MimeType)
		if b.ImageURL != "" {
			imageURL = sql.NullString{String: b.ImageURL, Valid: true}
		}
		if len(b.ImageData) > 0 {
			mt := b.MimeType
			if mt == "" {
				mt = "application/octet-stream"
			}
			id, err := upsertArtifact(ctx, tx, b.ImageData, mt, now)
			if err != nil {
				return err
			}
			artifactID = sql.NullString{String: id, Valid: true}
		}

	case conversation.BlockRedactedThinking:
		// Opaque bytes always go to the artifact table. MIME is a
		// private sentinel so the loader knows how to rehydrate.
		if len(b.RedactedData) > 0 {
			id, err := upsertArtifact(ctx, tx, b.RedactedData, "application/x-stackllm-redacted-thinking", now)
			if err != nil {
				return err
			}
			artifactID = sql.NullString{String: id, Valid: true}
			mimeType = sql.NullString{String: "application/x-stackllm-redacted-thinking", Valid: true}
		}

	default:
		return fmt.Errorf("session: insertBlock: unknown block type %q", b.Type)
	}

	_, err := tx.ExecContext(ctx, `
		INSERT INTO stackllm_blocks(
			id, message_id, seq, type, text_content, tool_call_id,
			tool_name, tool_args_json, tool_is_error,
			mime_type, image_url, artifact_id, created_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		b.ID,
		msgID,
		seq,
		string(b.Type),
		textContent,
		toolCallID,
		toolName,
		toolArgsJSON,
		toolIsErr,
		mimeType,
		imageURL,
		artifactID,
		formatTime(now),
	)
	return err
}

// upsertArtifact either returns the ID of an existing artifact row
// with the same SHA-256 (dedupe) or inserts a new row. Dedupe is
// best-effort: two concurrent writers racing on the same hash may
// both insert, leaving the DB with two rows for the same content.
// That's wasted space, not a correctness problem; a future sweeper
// can merge.
func upsertArtifact(ctx context.Context, tx *sql.Tx, data []byte, mimeType string, now time.Time) (string, error) {
	hash := sha256Hex(data)
	var existing string
	err := tx.QueryRowContext(ctx,
		`SELECT id FROM stackllm_artifacts WHERE sha256 = ? LIMIT 1`,
		hash,
	).Scan(&existing)
	if err == nil {
		return existing, nil
	}
	if err != sql.ErrNoRows {
		return "", fmt.Errorf("session: artifact lookup: %w", err)
	}

	id := conversation.NewID()
	if _, err := tx.ExecContext(ctx, `
		INSERT INTO stackllm_artifacts(id, sha256, mime_type, byte_size, data, created_at)
		VALUES (?, ?, ?, ?, ?, ?)`,
		id, hash, mimeType, int64(len(data)), data, formatTime(now),
	); err != nil {
		return "", fmt.Errorf("session: insert artifact: %w", err)
	}
	return id, nil
}

// parentMatches reports whether a persisted parent_id equals the
// expected running parent. Both sides use sql.NullString so a NULL
// parent (the root) compares equal to a zero NullString.
func parentMatches(db sql.NullString, expected sql.NullString) bool {
	if db.Valid != expected.Valid {
		return false
	}
	if !db.Valid {
		return true
	}
	return db.String == expected.String
}

// ---------------------------------------------------------------------
// Load: walk the current-leaf chain, join blocks, lazy artifacts.
// ---------------------------------------------------------------------

// Load reconstructs the session with the full linear history from
// root to current_leaf. Large artifact payloads are NOT hydrated —
// blocks that live in stackllm_artifacts come back with a populated
// ArtifactRef and either a preview (tool_result) or empty
// Text/ImageData. Callers that need the full bytes call
// HydrateArtifact on demand.
func (s *SQLiteStore) Load(ctx context.Context, id string) (*Session, error) {
	if id == "" {
		return nil, errors.New("session: Load: empty id")
	}

	var (
		name            sql.NullString
		projectPath     sql.NullString
		model           sql.NullString
		metadataJSON    string
		stateJSON       string
		leafID          sql.NullString
		created         string
		updated         string
		promptTokens    sql.NullInt64
		completionTokens sql.NullInt64
		totalTokens     sql.NullInt64
	)
	err := s.db.QueryRowContext(ctx, `
		SELECT name, project_path, model, metadata_json, state_json,
		       current_leaf_id, created_at, updated_at,
		       last_prompt_tokens, last_completion_tokens, last_total_tokens
		FROM stackllm_sessions WHERE id = ?`, id,
	).Scan(&name, &projectPath, &model, &metadataJSON, &stateJSON, &leafID, &created, &updated,
		&promptTokens, &completionTokens, &totalTokens)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("session: %q not found", id)
	}
	if err != nil {
		return nil, fmt.Errorf("session: Load: session row: %w", err)
	}
	_ = metadataJSON // reserved for future stackllm-owned fields

	state, err := unmarshalState(stateJSON)
	if err != nil {
		return nil, fmt.Errorf("session: Load: state: %w", err)
	}

	sess := &Session{
		ID:          id,
		Name:        name.String,
		ProjectPath: projectPath.String,
		Model:       model.String,
		State:       state,
		LastUsage:   usageFromCols(promptTokens, completionTokens, totalTokens),
		Created:     parseTime(created),
		Updated:     parseTime(updated),
	}

	if !leafID.Valid {
		return sess, nil
	}

	// Walk from leaf back to root via parent_id with a visited-set
	// cycle guard. Corrupt data should produce a clear error instead
	// of an infinite loop.
	msgIDs, err := walkMessageChain(ctx, s.db, id, leafID.String)
	if err != nil {
		return nil, err
	}

	// Fetch all message rows in one query, reverse, preserve order.
	msgs, err := loadMessageRows(ctx, s.db, msgIDs)
	if err != nil {
		return nil, fmt.Errorf("session: Load: message rows: %w", err)
	}

	// Fetch all blocks for those messages in one query, ordered by
	// (message_id, seq). Group into each message's Blocks slice.
	blockMap, err := loadBlocksForMessages(ctx, s.db, msgIDs)
	if err != nil {
		return nil, fmt.Errorf("session: Load: block rows: %w", err)
	}
	for i := range msgs {
		msgs[i].Blocks = blockMap[msgs[i].ID]
	}

	sess.Messages = msgs
	return sess, nil
}

// walkMessageChain returns the ordered list of message IDs from root
// to the given leaf, following parent_id. Uses an inline recursive
// CTE so all the walking happens in SQLite; includes a depth guard
// and a Go-side duplicate guard against cycles in corrupt data.
func walkMessageChain(ctx context.Context, db *sql.DB, sessionID, leafID string) ([]string, error) {
	const maxDepth = 10_000
	rows, err := db.QueryContext(ctx, `
		WITH RECURSIVE chain(id, parent_id, depth) AS (
			SELECT id, parent_id, 0 FROM stackllm_messages WHERE id = ? AND session_id = ?
			UNION ALL
			SELECT m.id, m.parent_id, c.depth + 1
			FROM stackllm_messages m
			JOIN chain c ON m.id = c.parent_id
			WHERE c.depth < ?
		)
		SELECT id, depth FROM chain ORDER BY depth DESC`,
		leafID, sessionID, maxDepth,
	)
	if err != nil {
		return nil, fmt.Errorf("session: walk chain: %w", err)
	}
	defer rows.Close()

	var ids []string
	seen := make(map[string]struct{})
	for rows.Next() {
		var id string
		var depth int
		if err := rows.Scan(&id, &depth); err != nil {
			return nil, fmt.Errorf("session: scan chain: %w", err)
		}
		if _, dup := seen[id]; dup {
			return nil, fmt.Errorf("session: message chain contains a cycle at %s", id)
		}
		seen[id] = struct{}{}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("session: iterate chain: %w", err)
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("session: current_leaf_id %s not found in session %s", leafID, sessionID)
	}
	return ids, nil
}

// loadMessageRows fetches the rows for the given message IDs and
// returns them in the same order (matching the chain from root to
// leaf). Used by Load.
func loadMessageRows(ctx context.Context, db *sql.DB, ids []string) ([]conversation.Message, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	const chunk = 500
	byID := make(map[string]conversation.Message, len(ids))
	for start := 0; start < len(ids); start += chunk {
		end := start + chunk
		if end > len(ids) {
			end = len(ids)
		}
		batch := ids[start:end]
		q := "SELECT id, role, model, duration_ms, created_at FROM stackllm_messages WHERE id IN (" + placeholders(len(batch)) + ")"
		args := make([]any, 0, len(batch))
		for _, id := range batch {
			args = append(args, id)
		}
		rows, err := db.QueryContext(ctx, q, args...)
		if err != nil {
			return nil, err
		}
		for rows.Next() {
			var (
				id         string
				role       string
				model      sql.NullString
				durationMS sql.NullInt64
				created    string
			)
			if err := rows.Scan(&id, &role, &model, &durationMS, &created); err != nil {
				rows.Close()
				return nil, err
			}
			m := conversation.Message{
				ID:        id,
				Role:      conversation.Role(role),
				Model:     model.String,
				CreatedAt: parseTime(created),
			}
			if durationMS.Valid {
				m.Duration = time.Duration(durationMS.Int64) * time.Millisecond
			}
			byID[id] = m
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return nil, err
		}
		rows.Close()
	}
	out := make([]conversation.Message, 0, len(ids))
	for _, id := range ids {
		if m, ok := byID[id]; ok {
			out = append(out, m)
		}
	}
	return out, nil
}

// loadBlocksForMessages fetches every block row for the given message
// IDs in a single query (plus an optional artifact-metadata lookup
// for blocks that are artifact-backed). Returns a map keyed by
// message_id; values are ordered by seq.
func loadBlocksForMessages(ctx context.Context, db *sql.DB, msgIDs []string) (map[string][]conversation.Block, error) {
	if len(msgIDs) == 0 {
		return map[string][]conversation.Block{}, nil
	}
	const chunk = 500
	out := make(map[string][]conversation.Block, len(msgIDs))
	artifactIDs := make(map[string]struct{})

	for start := 0; start < len(msgIDs); start += chunk {
		end := start + chunk
		if end > len(msgIDs) {
			end = len(msgIDs)
		}
		batch := msgIDs[start:end]
		q := `SELECT
			id, message_id, seq, type, text_content, tool_call_id,
			tool_name, tool_args_json, tool_is_error,
			mime_type, image_url, artifact_id
		FROM stackllm_blocks WHERE message_id IN (` + placeholders(len(batch)) + `) ORDER BY message_id, seq`
		args := make([]any, 0, len(batch))
		for _, id := range batch {
			args = append(args, id)
		}
		rows, err := db.QueryContext(ctx, q, args...)
		if err != nil {
			return nil, err
		}
		for rows.Next() {
			var (
				id, msgID, typ string
				seq            int
				textContent    sql.NullString
				toolCallID     sql.NullString
				toolName       sql.NullString
				toolArgsJSON   sql.NullString
				toolIsErr      sql.NullInt64
				mimeType       sql.NullString
				imageURL       sql.NullString
				artifactID     sql.NullString
			)
			if err := rows.Scan(&id, &msgID, &seq, &typ, &textContent, &toolCallID, &toolName, &toolArgsJSON, &toolIsErr, &mimeType, &imageURL, &artifactID); err != nil {
				rows.Close()
				return nil, err
			}
			b := conversation.Block{
				ID:           id,
				Type:         conversation.BlockType(typ),
				Text:         textContent.String,
				ToolCallID:   toolCallID.String,
				ToolName:     toolName.String,
				ToolArgsJSON: toolArgsJSON.String,
				ToolIsError:  toolIsErr.Valid && toolIsErr.Int64 != 0,
				MimeType:     mimeType.String,
				ImageURL:     imageURL.String,
			}
			if artifactID.Valid {
				b.ArtifactRef = &conversation.ArtifactRef{ID: artifactID.String}
				artifactIDs[artifactID.String] = struct{}{}
			}
			out[msgID] = append(out[msgID], b)
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return nil, err
		}
		rows.Close()
	}

	// Populate ArtifactRef metadata (but not the blob) in one lookup.
	if len(artifactIDs) > 0 {
		ids := make([]string, 0, len(artifactIDs))
		for id := range artifactIDs {
			ids = append(ids, id)
		}
		meta, err := loadArtifactMetadata(ctx, db, ids)
		if err != nil {
			return nil, err
		}
		for msgID, blocks := range out {
			for i := range blocks {
				if blocks[i].ArtifactRef == nil {
					continue
				}
				if m, ok := meta[blocks[i].ArtifactRef.ID]; ok {
					blocks[i].ArtifactRef = m
				}
			}
			out[msgID] = blocks
		}
	}
	return out, nil
}

// loadArtifactMetadata fetches id/mime_type/byte_size/sha256 for each
// artifact ID. Note: the data column is deliberately not selected —
// artifact payloads are hydrated lazily via HydrateArtifact.
func loadArtifactMetadata(ctx context.Context, db *sql.DB, ids []string) (map[string]*conversation.ArtifactRef, error) {
	out := make(map[string]*conversation.ArtifactRef, len(ids))
	const chunk = 500
	for start := 0; start < len(ids); start += chunk {
		end := start + chunk
		if end > len(ids) {
			end = len(ids)
		}
		batch := ids[start:end]
		q := "SELECT id, mime_type, byte_size, sha256 FROM stackllm_artifacts WHERE id IN (" + placeholders(len(batch)) + ")"
		args := make([]any, 0, len(batch))
		for _, id := range batch {
			args = append(args, id)
		}
		rows, err := db.QueryContext(ctx, q, args...)
		if err != nil {
			return nil, err
		}
		for rows.Next() {
			var (
				id, mime, sha string
				size          int64
			)
			if err := rows.Scan(&id, &mime, &size, &sha); err != nil {
				rows.Close()
				return nil, err
			}
			out[id] = &conversation.ArtifactRef{
				ID:       id,
				MimeType: mime,
				ByteSize: size,
				SHA256:   sha,
			}
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return nil, err
		}
		rows.Close()
	}
	return out, nil
}

// HydrateArtifact fetches the full payload for an artifact-backed
// block. Callers call it lazily when rendering (or when they need to
// re-send the bytes to a provider).
func (s *SQLiteStore) HydrateArtifact(ctx context.Context, artifactID string) ([]byte, string, error) {
	var (
		data []byte
		mime string
	)
	err := s.db.QueryRowContext(ctx,
		`SELECT data, mime_type FROM stackllm_artifacts WHERE id = ?`,
		artifactID,
	).Scan(&data, &mime)
	if err == sql.ErrNoRows {
		return nil, "", fmt.Errorf("session: artifact %q not found", artifactID)
	}
	if err != nil {
		return nil, "", fmt.Errorf("session: hydrate artifact: %w", err)
	}
	return data, mime, nil
}

// ---------------------------------------------------------------------
// Delete / List
// ---------------------------------------------------------------------

// Delete removes the session and cascades to messages and blocks via
// foreign keys. Artifacts are intentionally NOT garbage-collected in
// v1 — a future sweeper will handle orphans.
func (s *SQLiteStore) Delete(ctx context.Context, id string) error {
	conn, tx, err := s.begin(ctx)
	if err != nil {
		return err
	}
	defer releaseTx(conn, tx)
	if _, err := tx.ExecContext(ctx, `DELETE FROM stackllm_sessions WHERE id = ?`, id); err != nil {
		return fmt.Errorf("session: Delete: %w", err)
	}
	return tx.Commit()
}

// listSelectColumns is the column set both List and ListPage scan;
// kept in one place so the two paths can never drift.
const listSelectColumns = `id, name, project_path, model, state_json, created_at, updated_at,
		       last_prompt_tokens, last_completion_tokens, last_total_tokens`

// scanSessionRow reads one stackllm_sessions row in the order
// listSelectColumns declares. Used by List and ListPage.
func scanSessionRow(rows *sql.Rows) (*Session, error) {
	var (
		id               string
		name             sql.NullString
		projectPath      sql.NullString
		model            sql.NullString
		stateJSON        string
		created          string
		updated          string
		promptTokens     sql.NullInt64
		completionTokens sql.NullInt64
		totalTokens      sql.NullInt64
	)
	if err := rows.Scan(&id, &name, &projectPath, &model, &stateJSON, &created, &updated,
		&promptTokens, &completionTokens, &totalTokens); err != nil {
		return nil, err
	}
	state, err := unmarshalState(stateJSON)
	if err != nil {
		return nil, err
	}
	return &Session{
		ID:          id,
		Name:        name.String,
		ProjectPath: projectPath.String,
		Model:       model.String,
		State:       state,
		LastUsage:   usageFromCols(promptTokens, completionTokens, totalTokens),
		Created:     parseTime(created),
		Updated:     parseTime(updated),
	}, nil
}

// List returns every session with metadata populated and Messages
// empty. Callers call Load on demand for the ones they want to read
// in full.
func (s *SQLiteStore) List(ctx context.Context) ([]*Session, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT `+listSelectColumns+`
		FROM stackllm_sessions
		ORDER BY updated_at DESC`)
	if err != nil {
		return nil, fmt.Errorf("session: List: %w", err)
	}
	defer rows.Close()
	var out []*Session
	for rows.Next() {
		sess, err := scanSessionRow(rows)
		if err != nil {
			return nil, fmt.Errorf("session: List scan: %w", err)
		}
		out = append(out, sess)
	}
	return out, rows.Err()
}

// ListPage returns one page of sessions ordered by updated_at desc
// plus the total row count (ignoring Limit/Offset). Implements
// SessionPaginator. A negative Limit disables the cap; a zero Limit
// uses DefaultListLimit. Negative Offset is treated as 0.
func (s *SQLiteStore) ListPage(ctx context.Context, opts ListOptions) (ListResult, error) {
	var total int
	if err := s.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM stackllm_sessions`,
	).Scan(&total); err != nil {
		return ListResult{}, fmt.Errorf("session: ListPage count: %w", err)
	}

	offset := opts.Offset
	if offset < 0 {
		offset = 0
	}
	limit := opts.Limit
	if limit == 0 {
		limit = DefaultListLimit
	}

	q := `SELECT ` + listSelectColumns + `
		FROM stackllm_sessions
		ORDER BY updated_at DESC, id ASC`
	var args []any
	if limit > 0 {
		q += ` LIMIT ? OFFSET ?`
		args = append(args, limit, offset)
	} else if offset > 0 {
		q += ` LIMIT -1 OFFSET ?`
		args = append(args, offset)
	}

	rows, err := s.db.QueryContext(ctx, q, args...)
	if err != nil {
		return ListResult{}, fmt.Errorf("session: ListPage: %w", err)
	}
	defer rows.Close()
	out := []*Session{}
	for rows.Next() {
		sess, err := scanSessionRow(rows)
		if err != nil {
			return ListResult{}, fmt.Errorf("session: ListPage scan: %w", err)
		}
		out = append(out, sess)
	}
	if err := rows.Err(); err != nil {
		return ListResult{}, err
	}
	return ListResult{Sessions: out, Total: total}, nil
}

// ---------------------------------------------------------------------
// Branching: Fork / Rewind / ListBranches
// ---------------------------------------------------------------------

// Fork creates a new session whose message/block tree is the copied
// chain from the source session's root up to and including
// atMessageID. Messages get fresh IDs; parent_id links are remapped
// to the new IDs. Blocks also get fresh IDs. Artifact rows are
// REUSED (the new blocks point at the same artifact_id values) —
// that's dedupe by design.
func (s *SQLiteStore) Fork(ctx context.Context, srcSessionID, atMessageID, newName string) (*Session, error) {
	conn, tx, err := s.begin(ctx)
	if err != nil {
		return nil, err
	}
	defer releaseTx(conn, tx)

	// Resolve the source chain from root to atMessageID.
	chain, err := walkMessageChainTx(ctx, tx, srcSessionID, atMessageID)
	if err != nil {
		return nil, fmt.Errorf("session: Fork: resolve chain: %w", err)
	}

	// Read source session metadata so we can copy name/model/project.
	var (
		srcName, srcProject, srcModel                sql.NullString
		stateJSON                                    string
		srcPromptTok, srcCompletionTok, srcTotalTok  sql.NullInt64
	)
	if err := tx.QueryRowContext(ctx, `
		SELECT name, project_path, model, state_json,
		       last_prompt_tokens, last_completion_tokens, last_total_tokens
		FROM stackllm_sessions WHERE id = ?`,
		srcSessionID,
	).Scan(&srcName, &srcProject, &srcModel, &stateJSON,
		&srcPromptTok, &srcCompletionTok, &srcTotalTok); err != nil {
		return nil, fmt.Errorf("session: Fork: read source: %w", err)
	}

	now := time.Now().UTC()
	newID := conversation.NewID()

	if _, err := tx.ExecContext(ctx, `
		INSERT INTO stackllm_sessions(
			id, name, project_path, model,
			metadata_json, state_json, current_leaf_id,
			created_at, updated_at,
			last_prompt_tokens, last_completion_tokens, last_total_tokens
		) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)`,
		newID,
		nullString(newName),
		srcProject,
		srcModel,
		"{}",
		stateJSON,
		formatTime(now),
		formatTime(now),
		srcPromptTok, srcCompletionTok, srcTotalTok,
	); err != nil {
		return nil, fmt.Errorf("session: Fork: insert session: %w", err)
	}
	_ = srcName // reserved; newName takes precedence for the forked copy

	// Copy each message in chain order, remapping parent_id via the
	// idMap we build as we go. Each source block is also copied with
	// a fresh ID but points at the same artifact_id when present.
	idMap := make(map[string]string, len(chain))
	var lastNewID string
	for _, oldMsgID := range chain {
		var (
			role       string
			model      sql.NullString
			durationMS sql.NullInt64
			parentID   sql.NullString
			createdAt  string
		)
		if err := tx.QueryRowContext(ctx, `
			SELECT role, model, duration_ms, parent_id, created_at
			FROM stackllm_messages WHERE id = ?`, oldMsgID,
		).Scan(&role, &model, &durationMS, &parentID, &createdAt); err != nil {
			return nil, fmt.Errorf("session: Fork: read message %s: %w", oldMsgID, err)
		}

		newMsgID := conversation.NewID()
		idMap[oldMsgID] = newMsgID

		var newParent sql.NullString
		if parentID.Valid {
			if mapped, ok := idMap[parentID.String]; ok {
				newParent = sql.NullString{String: mapped, Valid: true}
			}
		}

		if _, err := tx.ExecContext(ctx, `
			INSERT INTO stackllm_messages(
				id, session_id, parent_id, role, model, duration_ms, created_at
			) VALUES (?, ?, ?, ?, ?, ?, ?)`,
			newMsgID, newID, newParent, role, model, durationMS, createdAt,
		); err != nil {
			return nil, fmt.Errorf("session: Fork: insert message: %w", err)
		}

		// Copy all blocks. Keep seq order, reuse artifact_id.
		blockRows, err := tx.QueryContext(ctx, `
			SELECT seq, type, text_content, tool_call_id, tool_name,
			       tool_args_json, tool_is_error, mime_type, image_url,
			       artifact_id, created_at
			FROM stackllm_blocks WHERE message_id = ? ORDER BY seq`, oldMsgID)
		if err != nil {
			return nil, fmt.Errorf("session: Fork: read blocks: %w", err)
		}
		type blockCopy struct {
			seq                                                  int
			typ                                                  string
			text, tcID, tname, targs, mime, iurl, aid, createdAt sql.NullString
			tisErr                                               sql.NullInt64
		}
		var copies []blockCopy
		for blockRows.Next() {
			var bc blockCopy
			if err := blockRows.Scan(&bc.seq, &bc.typ, &bc.text, &bc.tcID, &bc.tname, &bc.targs, &bc.tisErr, &bc.mime, &bc.iurl, &bc.aid, &bc.createdAt); err != nil {
				blockRows.Close()
				return nil, fmt.Errorf("session: Fork: scan block: %w", err)
			}
			copies = append(copies, bc)
		}
		if err := blockRows.Err(); err != nil {
			blockRows.Close()
			return nil, err
		}
		blockRows.Close()

		for _, bc := range copies {
			if _, err := tx.ExecContext(ctx, `
				INSERT INTO stackllm_blocks(
					id, message_id, seq, type, text_content, tool_call_id,
					tool_name, tool_args_json, tool_is_error,
					mime_type, image_url, artifact_id, created_at
				) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
				conversation.NewID(),
				newMsgID,
				bc.seq,
				bc.typ,
				bc.text,
				bc.tcID,
				bc.tname,
				bc.targs,
				bc.tisErr,
				bc.mime,
				bc.iurl,
				bc.aid,
				bc.createdAt.String,
			); err != nil {
				return nil, fmt.Errorf("session: Fork: insert block: %w", err)
			}
		}
		lastNewID = newMsgID
	}

	if _, err := tx.ExecContext(ctx, `UPDATE stackllm_sessions SET current_leaf_id = ? WHERE id = ?`, lastNewID, newID); err != nil {
		return nil, fmt.Errorf("session: Fork: update leaf: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return nil, err
	}
	return s.Load(ctx, newID)
}

// walkMessageChainTx is the tx-scoped variant of walkMessageChain
// used by Fork so the chain resolution sees in-progress transaction
// state.
func walkMessageChainTx(ctx context.Context, tx *sql.Tx, sessionID, leafID string) ([]string, error) {
	const maxDepth = 10_000
	rows, err := tx.QueryContext(ctx, `
		WITH RECURSIVE chain(id, parent_id, depth) AS (
			SELECT id, parent_id, 0 FROM stackllm_messages WHERE id = ? AND session_id = ?
			UNION ALL
			SELECT m.id, m.parent_id, c.depth + 1
			FROM stackllm_messages m
			JOIN chain c ON m.id = c.parent_id
			WHERE c.depth < ?
		)
		SELECT id FROM chain ORDER BY depth DESC`,
		leafID, sessionID, maxDepth,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("message %s not found in session %s", leafID, sessionID)
	}
	return ids, nil
}

// Rewind sets the session's current_leaf_id to toMessageID. Existing
// messages are never deleted — subsequent Save calls append as
// children of toMessageID, creating a sibling branch. Callers that
// want to "return to" the original tip can Rewind there again.
func (s *SQLiteStore) Rewind(ctx context.Context, sessionID, toMessageID string) error {
	conn, tx, err := s.begin(ctx)
	if err != nil {
		return err
	}
	defer releaseTx(conn, tx)

	// Validate that toMessageID belongs to this session. Without
	// this check, a typo would silently point the leaf at a stranger
	// message in another session.
	var found int
	if err := tx.QueryRowContext(ctx,
		`SELECT 1 FROM stackllm_messages WHERE id = ? AND session_id = ?`,
		toMessageID, sessionID,
	).Scan(&found); err != nil {
		if err == sql.ErrNoRows {
			return fmt.Errorf("session: Rewind: message %s not in session %s", toMessageID, sessionID)
		}
		return fmt.Errorf("session: Rewind: validate target: %w", err)
	}

	now := time.Now().UTC()
	if _, err := tx.ExecContext(ctx,
		`UPDATE stackllm_sessions SET current_leaf_id = ?, updated_at = ? WHERE id = ?`,
		toMessageID, formatTime(now), sessionID,
	); err != nil {
		return fmt.Errorf("session: Rewind: update leaf: %w", err)
	}
	return tx.Commit()
}

// ListBranches returns every direct child of referenceMessageID — the
// set of divergent branches that share referenceMessageID as their
// branch point. Empty slice if there are no children. Preview is the
// first 80 characters of the first text-bearing block in each child
// message.
func (s *SQLiteStore) ListBranches(ctx context.Context, sessionID, referenceMessageID string) ([]BranchRef, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, role, created_at
		FROM stackllm_messages
		WHERE session_id = ? AND parent_id = ?
		ORDER BY created_at ASC`,
		sessionID, referenceMessageID,
	)
	if err != nil {
		return nil, fmt.Errorf("session: ListBranches: %w", err)
	}
	defer rows.Close()

	var refs []BranchRef
	for rows.Next() {
		var (
			id, role, created string
		)
		if err := rows.Scan(&id, &role, &created); err != nil {
			return nil, fmt.Errorf("session: ListBranches scan: %w", err)
		}
		refs = append(refs, BranchRef{
			MessageID: id,
			Role:      conversation.Role(role),
			CreatedAt: parseTime(created),
		})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	// Populate previews in a second pass — cheap enough for a
	// branch list (which is tiny) and keeps the main query simple.
	for i := range refs {
		var preview sql.NullString
		err := s.db.QueryRowContext(ctx, `
			SELECT text_content FROM stackllm_blocks
			WHERE message_id = ? AND text_content IS NOT NULL
			ORDER BY seq ASC LIMIT 1`,
			refs[i].MessageID,
		).Scan(&preview)
		if err != nil && err != sql.ErrNoRows {
			return nil, fmt.Errorf("session: ListBranches preview: %w", err)
		}
		refs[i].Preview = truncatePreview(preview.String, 80)
	}
	return refs, nil
}

// Search runs an FTS5 query. scopeSessionID may be empty to search
// globally; blockTypes may be nil to search every text-bearing block
// type. Results are ordered by FTS5 rank ascending (best first).
func (s *SQLiteStore) Search(ctx context.Context, query string, scopeSessionID string, blockTypes []conversation.BlockType, limit int) ([]SearchHit, error) {
	if query == "" {
		return nil, nil
	}
	if limit <= 0 {
		limit = 50
	}

	// Build a WHERE that stays composable whether the caller passed
	// filters or not. The FTS MATCH is always present.
	where := []string{"stackllm_blocks_fts MATCH ?"}
	args := []any{query}
	if scopeSessionID != "" {
		where = append(where, "session_id = ?")
		args = append(args, scopeSessionID)
	}
	if len(blockTypes) > 0 {
		place := placeholders(len(blockTypes))
		where = append(where, "block_type IN ("+place+")")
		for _, t := range blockTypes {
			args = append(args, string(t))
		}
	}
	args = append(args, limit)

	q := `SELECT session_id, message_id, block_id, block_type,
	             snippet(stackllm_blocks_fts, 0, '[', ']', '…', 16),
	             rank
		FROM stackllm_blocks_fts
		WHERE ` + strings.Join(where, " AND ") + `
		ORDER BY rank
		LIMIT ?`

	rows, err := s.db.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, fmt.Errorf("session: Search: %w", err)
	}
	defer rows.Close()
	var hits []SearchHit
	for rows.Next() {
		var (
			sessID, msgID, blockID, blockType, snippet string
			rank                                       float64
		)
		if err := rows.Scan(&sessID, &msgID, &blockID, &blockType, &snippet, &rank); err != nil {
			return nil, err
		}
		hits = append(hits, SearchHit{
			SessionID: sessID,
			MessageID: msgID,
			BlockID:   blockID,
			BlockType: conversation.BlockType(blockType),
			Snippet:   snippet,
			Rank:      rank,
		})
	}
	return hits, rows.Err()
}

// ---------------------------------------------------------------------
// JSONL export / import — portable serialization.
// ---------------------------------------------------------------------

// jsonlHeader is the first line written by ExportJSONL. It carries
// session metadata so ImportJSONL can reconstruct the session row
// before inserting messages.
type jsonlHeader struct {
	Kind        string         `json:"kind"` // always "session_header"
	ID          string         `json:"id"`
	Name        string         `json:"name,omitempty"`
	ProjectPath string         `json:"project_path,omitempty"`
	Model       string         `json:"model,omitempty"`
	State       map[string]any `json:"state,omitempty"`
	Created     time.Time      `json:"created"`
	Updated     time.Time      `json:"updated"`
	Leaf        string         `json:"leaf,omitempty"`
}

// jsonlMessage wraps a conversation.Message on the wire along with
// the parent_id that positions it in the tree. Blocks carrying
// artifact payloads are rewritten with inline base64 bytes so the
// import side can rebuild the artifact from scratch.
type jsonlMessage struct {
	Kind      string               `json:"kind"` // always "message"
	ID        string               `json:"id"`
	ParentID  string               `json:"parent_id,omitempty"`
	Role      conversation.Role    `json:"role"`
	Model     string               `json:"model,omitempty"`
	Duration  time.Duration        `json:"duration,omitempty"`
	CreatedAt time.Time            `json:"created_at"`
	Blocks    []jsonlExportedBlock `json:"blocks"`
}

// jsonlExportedBlock is conversation.Block plus an inline artifact
// payload for blocks whose real bytes live in stackllm_artifacts.
type jsonlExportedBlock struct {
	conversation.Block
	ArtifactB64      string `json:"artifact_b64,omitempty"`
	ArtifactMIME     string `json:"artifact_mime,omitempty"`
	ArtifactFullText string `json:"artifact_full_text,omitempty"`
}

// ExportJSONL writes one JSON object per line: a header followed by
// every message in chain order. Artifact payloads are inlined as
// base64 (or raw text for text/plain tool_result offloads) so the
// export is fully self-contained.
func (s *SQLiteStore) ExportJSONL(ctx context.Context, sessionID string, w io.Writer) error {
	sess, err := s.Load(ctx, sessionID)
	if err != nil {
		return err
	}

	var leaf string
	if len(sess.Messages) > 0 {
		leaf = sess.Messages[len(sess.Messages)-1].ID
	}
	header := jsonlHeader{
		Kind:        "session_header",
		ID:          sess.ID,
		Name:        sess.Name,
		ProjectPath: sess.ProjectPath,
		Model:       sess.Model,
		State:       sess.State,
		Created:     sess.Created,
		Updated:     sess.Updated,
		Leaf:        leaf,
	}
	if err := writeJSONLine(w, header); err != nil {
		return fmt.Errorf("session: Export: header: %w", err)
	}

	parentByID := make(map[string]string, len(sess.Messages))
	var prev string
	for _, msg := range sess.Messages {
		parentByID[msg.ID] = prev
		prev = msg.ID
	}

	for _, msg := range sess.Messages {
		exported := jsonlMessage{
			Kind:      "message",
			ID:        msg.ID,
			ParentID:  parentByID[msg.ID],
			Role:      msg.Role,
			Model:     msg.Model,
			Duration:  msg.Duration,
			CreatedAt: msg.CreatedAt,
		}
		for _, b := range msg.Blocks {
			eb := jsonlExportedBlock{Block: b}
			if b.ArtifactRef != nil {
				data, mime, err := s.HydrateArtifact(ctx, b.ArtifactRef.ID)
				if err != nil {
					return fmt.Errorf("session: Export: hydrate %s: %w", b.ArtifactRef.ID, err)
				}
				eb.ArtifactMIME = mime
				// Preserve text payloads as raw text so consumers of
				// the export file don't have to base64-decode just
				// to read a tool_result.
				if mime == "text/plain" && b.Type == conversation.BlockToolResult {
					eb.ArtifactFullText = string(data)
				} else {
					eb.ArtifactB64 = base64.StdEncoding.EncodeToString(data)
				}
				// Drop the runtime ref so importers don't see a
				// dangling ID that doesn't exist in their DB.
				eb.ArtifactRef = nil
			}
			exported.Blocks = append(exported.Blocks, eb)
		}
		if err := writeJSONLine(w, exported); err != nil {
			return fmt.Errorf("session: Export: message %s: %w", msg.ID, err)
		}
	}
	return nil
}

// ImportJSONL reads a file produced by ExportJSONL into this store.
// A fresh session ID is always allocated so re-importing the same
// file is safe; the original ID is discarded. Returns the new
// session ID.
func (s *SQLiteStore) ImportJSONL(ctx context.Context, r io.Reader) (string, error) {
	dec := json.NewDecoder(r)

	// Peek the header first. We keep the raw header fields and
	// reconstruct a *Session, then call Save on a per-message basis
	// via a dedicated low-level path so we can preserve ArtifactRef
	// metadata where present.
	var first map[string]any
	if err := dec.Decode(&first); err != nil {
		return "", fmt.Errorf("session: Import: header: %w", err)
	}
	if first["kind"] != "session_header" {
		return "", fmt.Errorf("session: Import: expected session_header, got %v", first["kind"])
	}

	newID := conversation.NewID()

	conn, tx, err := s.begin(ctx)
	if err != nil {
		return "", err
	}
	defer releaseTx(conn, tx)

	now := time.Now().UTC()
	name, _ := first["name"].(string)
	projectPath, _ := first["project_path"].(string)
	model, _ := first["model"].(string)
	stateMap, _ := first["state"].(map[string]any)
	stateJSON, err := marshalState(stateMap)
	if err != nil {
		return "", fmt.Errorf("session: Import: state: %w", err)
	}
	// Import does not carry usage forward — the imported JSONL format
	// pre-dates LastUsage and the three columns stay NULL until the
	// caller runs another turn.
	if _, err := tx.ExecContext(ctx, `
		INSERT INTO stackllm_sessions(
			id, name, project_path, model,
			metadata_json, state_json, current_leaf_id,
			created_at, updated_at,
			last_prompt_tokens, last_completion_tokens, last_total_tokens
		) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL, NULL, NULL)`,
		newID,
		nullString(name),
		nullString(projectPath),
		nullString(model),
		"{}",
		stateJSON,
		formatTime(now),
		formatTime(now),
	); err != nil {
		return "", fmt.Errorf("session: Import: insert session: %w", err)
	}

	// Walk messages. The exported parent_id values reference the
	// ORIGINAL session's message IDs; we remap them to freshly-
	// minted IDs so the tree still reconstructs correctly.
	idMap := make(map[string]string)
	var lastNew string
	for dec.More() {
		var msg jsonlMessage
		if err := dec.Decode(&msg); err != nil {
			return "", fmt.Errorf("session: Import: decode message: %w", err)
		}
		if msg.Kind != "message" {
			return "", fmt.Errorf("session: Import: expected message, got %q", msg.Kind)
		}

		newMsgID := conversation.NewID()
		idMap[msg.ID] = newMsgID
		var parentNS sql.NullString
		if msg.ParentID != "" {
			if mapped, ok := idMap[msg.ParentID]; ok {
				parentNS = sql.NullString{String: mapped, Valid: true}
			}
		}

		var durationMS sql.NullInt64
		if msg.Duration > 0 {
			durationMS = sql.NullInt64{Int64: msg.Duration.Milliseconds(), Valid: true}
		}
		createdAt := msg.CreatedAt
		if createdAt.IsZero() {
			createdAt = now
		}
		if _, err := tx.ExecContext(ctx, `
			INSERT INTO stackllm_messages(
				id, session_id, parent_id, role, model, duration_ms, created_at
			) VALUES (?, ?, ?, ?, ?, ?, ?)`,
			newMsgID, newID, parentNS, string(msg.Role), nullString(msg.Model), durationMS, formatTime(createdAt),
		); err != nil {
			return "", fmt.Errorf("session: Import: insert message: %w", err)
		}

		for seq, eb := range msg.Blocks {
			// If the exported block carried artifact bytes, rebuild
			// the artifact fresh in this store so the new block's
			// artifact_id is local.
			b := eb.Block
			b.ID = conversation.NewID()
			if eb.ArtifactB64 != "" {
				data, err := base64.StdEncoding.DecodeString(eb.ArtifactB64)
				if err != nil {
					return "", fmt.Errorf("session: Import: decode artifact bytes: %w", err)
				}
				switch b.Type {
				case conversation.BlockImage:
					b.ImageData = data
					if b.MimeType == "" {
						b.MimeType = eb.ArtifactMIME
					}
				case conversation.BlockRedactedThinking:
					b.RedactedData = data
				default:
					// Unknown types with binary artifacts round-trip
					// as blocks with ImageData — unusual but harmless.
					b.ImageData = data
					if b.MimeType == "" {
						b.MimeType = eb.ArtifactMIME
					}
				}
			}
			if eb.ArtifactFullText != "" {
				b.Text = eb.ArtifactFullText
			}
			b.ArtifactRef = nil
			if err := insertBlock(ctx, tx, newMsgID, seq, &b, now); err != nil {
				return "", fmt.Errorf("session: Import: insert block: %w", err)
			}
		}
		lastNew = newMsgID
	}

	if _, err := tx.ExecContext(ctx, `UPDATE stackllm_sessions SET current_leaf_id = ? WHERE id = ?`, lastNew, newID); err != nil {
		return "", fmt.Errorf("session: Import: set leaf: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return "", err
	}
	return newID, nil
}

// ---------------------------------------------------------------------
// Small helpers.
// ---------------------------------------------------------------------

// placeholders returns "?, ?, ?" for n placeholders.
func placeholders(n int) string {
	if n <= 0 {
		return ""
	}
	buf := make([]byte, 0, n*3)
	for i := 0; i < n; i++ {
		if i > 0 {
			buf = append(buf, ',', ' ')
		}
		buf = append(buf, '?')
	}
	return string(buf)
}

// nullString wraps a Go string as a sql.NullString, treating the
// empty string as NULL.
func nullString(s string) sql.NullString {
	if s == "" {
		return sql.NullString{}
	}
	return sql.NullString{String: s, Valid: true}
}

// usageCols unpacks a *TokenUsage into three sql.NullInt64 values for
// binding into the last_prompt_tokens / last_completion_tokens /
// last_total_tokens columns. A nil usage becomes three NULLs.
func usageCols(u *conversation.TokenUsage) (sql.NullInt64, sql.NullInt64, sql.NullInt64) {
	if u == nil {
		return sql.NullInt64{}, sql.NullInt64{}, sql.NullInt64{}
	}
	return sql.NullInt64{Int64: int64(u.PromptTokens), Valid: true},
		sql.NullInt64{Int64: int64(u.CompletionTokens), Valid: true},
		sql.NullInt64{Int64: int64(u.TotalTokens), Valid: true}
}

// usageFromCols rebuilds a *TokenUsage from three scanned columns.
// Returns nil when every column is NULL — the sentinel for "no usage
// recorded yet" that LastUsage exposes to callers.
func usageFromCols(prompt, completion, total sql.NullInt64) *conversation.TokenUsage {
	if !prompt.Valid && !completion.Valid && !total.Valid {
		return nil
	}
	return &conversation.TokenUsage{
		PromptTokens:     int(prompt.Int64),
		CompletionTokens: int(completion.Int64),
		TotalTokens:      int(total.Int64),
	}
}

// formatTime / parseTime use RFC3339Nano so times survive a DB
// round-trip with sub-second precision.
func formatTime(t time.Time) string {
	return t.UTC().Format(time.RFC3339Nano)
}

func parseTime(s string) time.Time {
	if s == "" {
		return time.Time{}
	}
	t, err := time.Parse(time.RFC3339Nano, s)
	if err != nil {
		return time.Time{}
	}
	return t
}

func marshalState(state map[string]any) (string, error) {
	if state == nil {
		return "{}", nil
	}
	b, err := json.Marshal(state)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func unmarshalState(s string) (map[string]any, error) {
	if s == "" || s == "null" {
		return map[string]any{}, nil
	}
	var out map[string]any
	if err := json.Unmarshal([]byte(s), &out); err != nil {
		return nil, err
	}
	if out == nil {
		return map[string]any{}, nil
	}
	return out, nil
}

// truncatePreview returns the first n runes of s, collapsing newlines
// to spaces so branch previews are single-line.
func truncatePreview(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= n {
		return s
	}
	// Operate on runes so we don't slice a multibyte char.
	rs := []rune(s)
	if len(rs) <= n {
		return s
	}
	return string(rs[:n])
}

// writeJSONLine writes v as a single JSON object followed by a
// newline. Used by ExportJSONL.
func writeJSONLine(w io.Writer, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if _, err := w.Write(b); err != nil {
		return err
	}
	_, err = w.Write([]byte("\n"))
	return err
}
