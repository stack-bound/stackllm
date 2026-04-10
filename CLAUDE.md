# stackllm

A minimal, embeddable Go agent orchestration library. Single binary. Full context control.

Module: `github.com/stack-bound/stackllm`

## Architecture

```
conversation/  ← Foundation types (Message, Block, Role, compaction)
     ↑
auth/          ← Token acquisition and storage (static, Copilot OAuth, OpenAI OAuth)
     ↑
tools/         ← Tool interface, JSON Schema generation, registry + dispatch
     ↑
provider/      ← LLM provider (single OpenAI-compat impl for 5 backends)
     ↑
agent/         ← ReAct agent loop (Step, Run, hooks)
     ↑
session/       ← Session state and persistence
     ↑
tui/           ← Bubbletea TUI adapter (optional)
web/           ← HTTP/SSE adapter (optional)
```

All five supported providers use the OpenAI chat completions wire format:

| Provider | Config helper | Auth |
|---|---|---|
| OpenAI | `provider.OpenAIConfig()` | Static API key |
| Azure OpenAI | `provider.AzureConfig()` | API key or Entra token |
| Ollama | `provider.OllamaConfig()` | None |
| GitHub Copilot | `provider.CopilotConfig()` | Two-phase OAuth via `auth.NewCopilotSource()` |
| Gemini | `provider.GeminiConfig()` | Google API key |

## Build and test

```bash
go build ./...
go test ./...
go vet ./...
```

No external services required for tests — all provider tests use `httptest.NewServer`.

## Package guide

### conversation/

Foundation types imported by every other package.

- `Message` — `ID`, `Role`, `Blocks []Block`, `Model`, `CreatedAt`, `Duration`. Assistant output is an ordered slice of typed `Block`s — the order is the replay timeline.
- `Block` — `ID`, `Type`, plus type-specific fields. One of:
  - `BlockText` — visible text
  - `BlockThinking` — model reasoning / chain-of-thought
  - `BlockRedactedThinking` — opaque encrypted reasoning (Anthropic)
  - `BlockImage` — image input (inline bytes or external URL)
  - `BlockToolUse` — model requesting a tool invocation (id, name, args JSON)
  - `BlockToolResult` — tool output linked back to a tool_use by call id
- `ArtifactRef` — populated by the store on Load when a block's payload was offloaded to an artifacts table; callers hydrate on demand
- `EnsureMessageIDs(&msg)` — assigns a stable UUIDv7 (via `conversation.NewID`) to the message and each of its blocks if they don't already have one
- Message helpers: `TextContent()` / `ThinkingText()` concatenate blocks of the matching type; `ToolUses()` / `ToolResults()` filter; `HasToolUses()`, `IsSystem()` predicates
- `Builder` — fluent API. Message starters: `System(s)` / `User(s)` / `Assistant(s)` / `ToolResult(id, s)`. Block appenders that target the most recently added message: `Text(s)` / `Thinking(s)` / `ToolUse(id, name, args)` / `Image(mime, data)` / `ImageURL(mime, url)` / `ToolResultBlock(id, s, isErr)`. `Add(msg)` inserts an arbitrary message.
- `KeepLast(msgs, n)` — keep last n non-system messages, always preserve system
- `TokenBudget(msgs, max, countFn)` — drop oldest until under budget. Pass nil for a chars/4 heuristic across all text-bearing blocks, or supply your own tokeniser

### auth/

Token management with pluggable storage.

- `NewStatic(key)` — for API keys (OpenAI, Gemini, Azure, Ollama)
- `NewCopilotSource(cfg)` — two-phase GitHub device flow → Copilot token exchange
- `NewOpenAIDeviceSource(cfg)` — headless device code flow
- `NewOpenAIWebFlowSource(cfg)` — PKCE flow with local callback server
- `NewCachingSource(inner)` — wrap any source to cache until expiry
- `FileStore` — persists to `~/.config/stackllm/auth.json` (atomic writes)
- `MemoryStore` — in-memory, for tests

### tools/

Tool definition and dispatch.

```go
type ReadArgs struct {
    Path string `json:"path" jsonschema:"description=File path,required"`
}
registry := tools.NewRegistry()
registry.Register("read_file", "Read a file", func(ctx context.Context, args ReadArgs) (string, error) {
    data, err := os.ReadFile(args.Path)
    return string(data), err
})
```

- `SchemaOf(v)` — generates JSON Schema from Go structs via reflection
- Struct tags: `json:"name"`, `jsonschema:"description=...,required,enum=a|b|c"`
- `Registry.Dispatch(ctx, name, argsJSON)` — unmarshal + call + return result

### provider/

Single implementation for all five OpenAI-compatible backends.

```go
p := provider.New(provider.OpenAIConfig("gpt-4o", auth.NewStatic(os.Getenv("OPENAI_API_KEY"))))
events, _ := p.Complete(ctx, provider.Request{Messages: msgs, Stream: true})
for ev := range events {
    // ev.Type: EventTypeBlockStart, EventTypeBlockDelta, EventTypeBlockEnd,
    //          EventTypeToolCall, EventTypeDone, EventTypeError
}
```

- **Block-oriented streaming.** Providers emit an ordered sequence of
  block events so the agent can reconstruct interleaved thinking → text
  → tool_use output faithfully:
  - `EventTypeBlockStart` fires once when a new block opens (`ev.BlockType` set).
  - `EventTypeBlockDelta` fires repeatedly while the block streams in (`ev.BlockType`, `ev.Content` set).
  - `EventTypeBlockEnd` fires once when the block closes with the fully accumulated block in `ev.Block`.
  - `EventTypeToolCall` is a convenience alias fired after the matching `BlockEnd` for tool_use blocks, for callers that only care about dispatched tool calls.
- **Wire format.** The chat completions endpoint is lossy — `buildRequestBody` flattens blocks by concatenating text, hoisting `BlockToolUse` into `tool_calls`, emitting multi-part `content` arrays for images, and **dropping thinking blocks** (the legacy API has no slot for them). The `/responses` endpoint (`Endpoint=EndpointResponses`) preserves reasoning and is the only wire format that can faithfully replay interleaved blocks.
- **Reasoning parsing.** `readChatSSE` recognises `delta.reasoning_content` / `delta.reasoning` alongside `delta.content` and emits separate `BlockThinking` / `BlockText` blocks in the order the model switches between them. `readResponsesSSE` maps each `output_item` (reasoning / message / function_call) to a block, closed in the order `output_item.done` fires.
- Auto-retry on 429/5xx with exponential backoff (configurable MaxRetries)
- Auth injected via `authRoundTripper` wrapping the HTTP client

### agent/

ReAct loop: call LLM → dispatch tools → repeat.

```go
a := agent.New(p, agent.WithTools(registry), agent.WithMaxSteps(10))

// Low-level: one round-trip
msgs, result, err := a.Step(ctx, msgs)

// High-level: full loop until done or max steps
events, _ := a.Run(ctx, msgs)
for ev := range events {
    // ev.Type: EventBlockStart, EventBlockDelta, EventBlockEnd,
    //          EventToken, EventToolCall, EventToolResult,
    //          EventStepDone, EventComplete, EventError
}
```

- **Block accumulation.** `Step` collects blocks from `EventTypeBlockEnd` events in the order the provider closes them, then builds one assistant `Message` whose `Blocks` is the full interleaved timeline. When the assistant message contains one or more `BlockToolUse` blocks, the agent dispatches them and appends **one** tool-role `Message` containing one `BlockToolResult` per tool_use (matching the Anthropic shape).
- **Stable IDs.** Assistant and tool messages are passed through `conversation.EnsureMessageIDs` before being returned, so every persisted message and block has a stable identifier.
- `Hooks` — `BeforeCall`, `OnBlockStart`, `OnBlockDelta`, `OnBlockEnd`, `OnToken` (convenience wrapper that only fires for `BlockText` deltas), `OnToolCall`, `OnToolResult`, `AfterComplete`
- Tool errors become `"Error: ..."` messages in the conversation (with `ToolIsError = true` on the block), not Go errors

### session/

Session state management.

```go
sess := session.New()               // random ID
sess.AppendMessage(msg)             // assigns message/block IDs, updates timestamp
sess.SetState("key", value)         // arbitrary KV

store := session.NewInMemoryStore()  // or implement SessionStore for Redis/file/DB
store.Save(ctx, sess)
```

`Session.AppendMessage` calls `conversation.EnsureMessageIDs` so every
message and block held by a session has a stable identifier, regardless
of whether the caller built the message via `conversation.Builder` or
assembled it by hand.

#### Session persistence (SQLiteStore)

`session.SQLiteStore` is the durable `SessionStore` implementation,
built on `modernc.org/sqlite` so it requires no CGO and no
separately-installed database. There are two constructors:

```go
// Simple embedders: let stackllm own the file.
store, _ := session.OpenSQLiteStore(session.SQLiteConfig{AppName: "myapp"})
defer store.Close()

// Parent apps that already own a *sql.DB and want to share the same
// SQLite file with their own tables:
db, _ := sql.Open("sqlite", dsn)
store, _ := session.NewSQLiteStore(db) // Close is a no-op in this mode
```

- **No silent default path.** `OpenSQLiteStore` errors if neither
  `AppName` nor `Path` is set. `AppName` resolves to
  `$XDG_DATA_HOME/{AppName}/state.db`; `Path` is an explicit
  override. Two embedders with different `AppName`s cannot collide.
- **Table prefix rule.** Every stackllm-owned table is prefixed
  `stackllm_` (`stackllm_sessions`, `stackllm_messages`,
  `stackllm_blocks`, `stackllm_artifacts`, `stackllm_blocks_fts`,
  `stackllm_schema_version`). Parent apps that share the DB MUST
  NOT create tables with that prefix; everything else is fair game.
  `examples/sqlite/main.go` demonstrates the sharing pattern.
- **Blocks are rows.** Each message is a tree node with a
  `parent_id`; each block within a message is its own row in
  `stackllm_blocks` ordered by `seq`. Save is append-only and
  diffs the in-memory session against the DB via a bulk
  `SELECT id IN (...)` — existing messages are validated (parent
  chain must match) and new ones are inserted at the tail. Editing
  a persisted block's `Text` in memory is silently ignored;
  history mutation must go through `Fork` or `Rewind`.
- **Artifact offload.** Tool results larger than 64 KB, inline
  image bytes, and redacted-thinking payloads are offloaded to
  `stackllm_artifacts` and referenced by `artifact_id`. The block
  row keeps a 2 KB UTF-8-safe preview in `text_content` so
  scrollback and FTS still work. `Load` does not hydrate artifact
  bytes — call `store.HydrateArtifact(ctx, ref.ID)` lazily.
  Artifacts are deduped by SHA-256; two sessions that store the
  same payload share one artifact row. Orphaned artifacts are
  NOT garbage-collected in v1.
- **FTS5 search.** `store.Search(ctx, query, scopeSession, blockTypes, limit)`
  runs an FTS5 query across text-bearing blocks (`text`,
  `thinking`, `tool_result`). `blockTypes` filters match types;
  pass nil for all. For offloaded tool results, only the preview
  is indexed — full content search requires hydration.
- **Branching.** `Fork`, `Rewind`, and `ListBranches` let callers
  create sibling branches at any message boundary. Fork copies
  messages and blocks with fresh IDs but reuses artifact rows.
  Rewind sets `current_leaf_id` to a past message; subsequent
  Save calls append as children of that message, creating a new
  branch without deleting the old one.
- **WAL + concurrent readers.** Pragmas (`journal_mode=wal`,
  `foreign_keys=1`, `busy_timeout=5000`, `synchronous=normal`)
  are set via DSN on `OpenSQLiteStore`; `NewSQLiteStore` relies on
  the caller for journal mode and enforces `foreign_keys` per
  transaction.

### tui/

Bubbletea interactive terminal UI.

```go
m := tui.New(agent, store)
p := tea.NewProgram(m)
p.Run()
```

- `RenderMessage(msg)` / `RenderConversation(msgs)` — walks `msg.Blocks` in order and renders each typed block: text inline, thinking dimmed with a `thinking:` prefix, tool_use as `⚡ tool_name(args)`, tool_result with call id, image as `[image: mime, bytes]` placeholder, redacted thinking as a byte-count placeholder. Interleaving is preserved because blocks are rendered in slice order.
- Live streaming uses `agent.EventBlockStart` / `EventBlockDelta` / `EventBlockEnd` so thinking text appears in a dimmed region between the surrounding text segments as the model produces it.
- `DeviceCodePrompt(code, url)` — boxed auth prompt for device flows
- `WebFlowPrompt(url)` — boxed auth prompt for web flows

### web/

HTTP/SSE adapter for serving agents over HTTP.

```go
h := web.NewHandler(agent, store)
http.ListenAndServe(":8080", h)
```

Routes:
- `POST /chat` — block-shaped request body → SSE stream. The canonical
  payload is `{"session_id": "...", "message": {"role": "user",
  "blocks": [{"type": "text", "text": "..."}]}}`. For backward
  compatibility the handler also accepts the legacy shape
  `{"session_id": "...", "message": "plain string"}`, which is
  converted to a single `BlockText`-bearing user message.
- `GET /sessions/{id}` — retrieve session as JSON. Field names are
  lowercase-snake (`id`, `role`, `blocks`, `tool_call_id`, …) because
  `conversation.Message` and `conversation.Block` carry `json:` struct
  tags.
- `DELETE /sessions/{id}` — delete session

SSE events: `block_start`, `block_delta`, `block_end`, `done`, `error`. Each `block_end` payload carries the fully accumulated block under a `block` key; each `block_delta` carries `block_type` + `delta`. Binary payloads (image / redacted data) are emitted as byte lengths, not raw bytes, to keep SSE lines small.

## Conventions

- Errors: `fmt.Errorf("package: action: %w", err)`
- Context: every exported I/O function takes `ctx context.Context` first
- No global state, no `init()`, no package-level vars except constants
- Functional options for config: `agent.New(p, agent.WithMaxSteps(10))`
- Tests: table-driven, `t.Parallel()`, `httptest.NewServer` for HTTP
- Interfaces defined where consumed, not where implemented

## Implementation rules

These rules are mandatory. Do not skip or shortcut any of them.

### No stubs, no placeholders
- Every function must be fully implemented. A TODO comment is not an implementation.
- If a function cannot be implemented yet (e.g. missing dependency), do not mark the phase as complete. Leave it explicitly incomplete and say why.
- "Fire and forget" goroutines that discard results (e.g. `go Login(ctx)` with no way to get the result back) are stubs. Implement the actual data flow.

### Persist state correctly
- Any adapter (web, TUI, etc.) that calls `agent.Run()` MUST write the full evolved conversation (assistant messages, tool calls, tool results) back into the session before saving. Passing messages in and ignoring the output is a bug.
- OAuth token responses that include `expires_in` MUST store and honour the expiry. Storing only the bare access token string and reloading it as never-expiring is a bug.

### Tests must verify behaviour, not just structure
- Tests must assert on the actual outcomes that matter: persisted data contains expected content, tokens expire when they should, auth flows complete end-to-end.
- A test that only checks HTTP status codes or "file exists" is insufficient. Test the payload.
- If a code path cannot be tested (e.g. real external API), write the test anyway with a mock server via `httptest.NewServer` — do not skip it.

### Self-review checklist (run before marking anything complete)
1. Does every code path actually execute, or are there dead branches / unreachable returns?
2. Does persisted state (sessions, tokens) round-trip correctly — write then read back and verify contents?
3. Are all plan requirements implemented, not just the easy ones?
4. Do tests fail if the feature is broken, or do they pass vacuously?

## Dependencies

| Package | Purpose |
|---|---|
| `github.com/charmbracelet/bubbletea` | TUI framework (tui/ only) |
| `github.com/charmbracelet/bubbles` | TUI components (tui/ only) |
| `github.com/charmbracelet/lipgloss` | TUI styling (tui/ only) |
| `github.com/google/uuid` | UUIDv7 generation for message / block / session IDs |
| `modernc.org/sqlite` | Pure-Go SQLite driver powering `session.SQLiteStore` — no CGO, no sidecar database |

**What "minimal" means here.** stackllm is built to have **no external
runtime requirements**: no sidecar processes, no separately-installed
databases, no auxiliary binaries an embedder has to ship next to the
Go binary. Everything stackllm needs must compile into the same
`go build` output as the parent app. Go package dependencies are
fine — they vendor into the binary — so the test we apply when
adding a dep is: does this introduce a *runtime* requirement beyond
the compiled binary, or is it just code that will be linked in?

Under this rule, well-maintained pure-Go libraries (UUID generation,
a pure-Go SQLite driver like `modernc.org/sqlite`) are welcome when
they earn their place. CGO is avoided because it breaks the
cross-compile story. System services (Postgres, Redis, a separate
`opencode` binary) are not welcome because they force the embedder to
deploy and operate something extra.
