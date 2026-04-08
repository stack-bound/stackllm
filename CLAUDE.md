# stackllm

A minimal, embeddable Go agent orchestration library. Single binary. Full context control.

Module: `github.com/stack-bound/stackllm`

## Architecture

```
conversation/  ← Foundation types (Message, Role, ToolCall, compaction)
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

- `Message` — role, content, tool calls, tool call ID
- `Builder` — fluent API: `NewBuilder().System("...").User("...").Build()`
- `KeepLast(msgs, n)` — keep last n non-system messages, always preserve system
- `TokenBudget(msgs, max, countFn)` — drop oldest until under budget. Pass nil for chars/4 heuristic, or supply your own tokeniser

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
    // ev.Type: EventTypeToken, EventTypeToolCall, EventTypeDone, EventTypeError
}
```

- Streaming SSE parsing with tool call accumulation across deltas
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
    // ev.Type: EventToken, EventToolCall, EventToolResult, EventStepDone, EventComplete, EventError
}
```

- `Hooks` — BeforeCall, OnToken, OnToolCall, OnToolResult, AfterComplete
- Tool errors become `"Error: ..."` messages in the conversation, not Go errors

### session/

Session state management.

```go
sess := session.New()               // random ID
sess.AppendMessage(msg)             // updates timestamp
sess.SetState("key", value)         // arbitrary KV

store := session.NewInMemoryStore()  // or implement SessionStore for Redis/file/DB
store.Save(ctx, sess)
```

### tui/

Bubbletea interactive terminal UI.

```go
m := tui.New(agent, store)
p := tea.NewProgram(m)
p.Run()
```

- `RenderMessage(msg)` / `RenderConversation(msgs)` — styled message formatting
- `DeviceCodePrompt(code, url)` — boxed auth prompt for device flows
- `WebFlowPrompt(url)` — boxed auth prompt for web flows

### web/

HTTP/SSE adapter for serving agents over HTTP.

```go
h := web.NewHandler(agent, store)
http.ListenAndServe(":8080", h)
```

Routes:
- `POST /chat` — `{"session_id": "...", "message": "..."}` → SSE stream
- `GET /sessions/{id}` — retrieve session
- `DELETE /sessions/{id}` — delete session

SSE events: `token`, `tool_call`, `tool_result`, `done`, `error`

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

No other external dependencies. Standard library covers HTTP, JSON, crypto, reflect.
