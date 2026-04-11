package session

// latestSchemaVersion is the current version of the stackllm SQLite
// schema. Opening a DB whose stored version exceeds this value is a
// hard error — the caller must upgrade stackllm.
const latestSchemaVersion = 2

// defaultArtifactThreshold is the byte size above which a
// BlockToolResult's text payload is offloaded to stackllm_artifacts
// instead of being stored inline on the block row. Images with inline
// ImageData and redacted thinking blocks are always offloaded
// regardless of size.
const defaultArtifactThreshold = 64 * 1024

// artifactPreviewBytes is the number of bytes of a large tool_result's
// output that are copied into stackllm_blocks.text_content as a
// scrollback + FTS preview. The full payload lives in the artifact.
const artifactPreviewBytes = 2 * 1024

// schemaV1 is the initial schema. All tables are prefixed stackllm_ so
// parent applications can safely create their own tables in the same
// SQLite file. Triggers keep the FTS5 index in sync with text-bearing
// blocks only — images, redacted thinking, and raw artifact bytes are
// never indexed.
const schemaV1 = `
CREATE TABLE IF NOT EXISTS stackllm_schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS stackllm_sessions (
    id               TEXT PRIMARY KEY,
    name             TEXT,
    project_path     TEXT,
    model            TEXT,
    metadata_json    TEXT NOT NULL,
    state_json       TEXT NOT NULL,
    current_leaf_id  TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_stackllm_sessions_updated  ON stackllm_sessions(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_stackllm_sessions_project  ON stackllm_sessions(project_path);

CREATE TABLE IF NOT EXISTS stackllm_messages (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES stackllm_sessions(id) ON DELETE CASCADE,
    parent_id     TEXT REFERENCES stackllm_messages(id),
    role          TEXT NOT NULL,
    model         TEXT,
    duration_ms   INTEGER,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_stackllm_messages_session ON stackllm_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_stackllm_messages_parent  ON stackllm_messages(parent_id);

CREATE TABLE IF NOT EXISTS stackllm_artifacts (
    id          TEXT PRIMARY KEY,
    sha256      TEXT NOT NULL,
    mime_type   TEXT NOT NULL,
    byte_size   INTEGER NOT NULL,
    data        BLOB NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_stackllm_artifacts_sha ON stackllm_artifacts(sha256);

CREATE TABLE IF NOT EXISTS stackllm_blocks (
    id              TEXT PRIMARY KEY,
    message_id      TEXT NOT NULL REFERENCES stackllm_messages(id) ON DELETE CASCADE,
    seq             INTEGER NOT NULL,
    type            TEXT NOT NULL,
    text_content    TEXT,
    tool_call_id    TEXT,
    tool_name       TEXT,
    tool_args_json  TEXT,
    tool_is_error   INTEGER,
    mime_type       TEXT,
    image_url       TEXT,
    artifact_id     TEXT REFERENCES stackllm_artifacts(id),
    created_at      TEXT NOT NULL,
    UNIQUE(message_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_stackllm_blocks_message   ON stackllm_blocks(message_id, seq);
CREATE INDEX IF NOT EXISTS idx_stackllm_blocks_toolcall  ON stackllm_blocks(tool_call_id);
CREATE INDEX IF NOT EXISTS idx_stackllm_blocks_type      ON stackllm_blocks(type);

CREATE VIRTUAL TABLE IF NOT EXISTS stackllm_blocks_fts USING fts5(
    content,
    block_id   UNINDEXED,
    message_id UNINDEXED,
    session_id UNINDEXED,
    block_type UNINDEXED,
    tokenize = 'porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS stackllm_blocks_ai
AFTER INSERT ON stackllm_blocks
WHEN new.text_content IS NOT NULL AND new.type IN ('text','thinking','tool_result')
BEGIN
    INSERT INTO stackllm_blocks_fts(rowid, content, block_id, message_id, session_id, block_type)
    SELECT new.rowid, new.text_content, new.id, new.message_id, m.session_id, new.type
    FROM stackllm_messages m WHERE m.id = new.message_id;
END;

CREATE TRIGGER IF NOT EXISTS stackllm_blocks_ad
AFTER DELETE ON stackllm_blocks
BEGIN
    DELETE FROM stackllm_blocks_fts WHERE rowid = old.rowid;
END;
`

// schemaV2 adds per-session token usage columns so that the TUI can
// restore the "N / MAX (pct%)" display when reopening a session
// without having to re-run a turn to learn the numbers.
const schemaV2 = `
ALTER TABLE stackllm_sessions ADD COLUMN last_prompt_tokens INTEGER;
ALTER TABLE stackllm_sessions ADD COLUMN last_completion_tokens INTEGER;
ALTER TABLE stackllm_sessions ADD COLUMN last_total_tokens INTEGER;
`

// migrations is the ordered list of schema migrations. Index i is the
// SQL that upgrades the DB from version i to version i+1, so
// migrations[0] applies to a fresh (version-0) database. To add a
// migration: append the SQL here and bump latestSchemaVersion.
var migrations = []string{
	schemaV1,
	schemaV2,
}
