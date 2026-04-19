# stackllm

A minimal, embeddable Go agent orchestration library. Single binary. Full context control.

You own the `[]Message` — the library provides the loop and the plumbing.

## Features

- **One OpenAI-compatible provider** — talks to OpenAI, Azure OpenAI, Ollama, GitHub Copilot, and Gemini through a single implementation
- **Unified provider manager** — one entry point for login, logout, status, model discovery, and default persistence across every supported provider
- **Full context control** — the caller owns the message slice and decides what goes in it
- **Block-shaped messages** — each message holds an ordered slice of typed blocks (text, thinking, tool_use, tool_result, image, redacted_thinking), so interleaved assistant output replays faithfully
- **Durable session store** — pure-Go SQLite (`modernc.org/sqlite`) backing for branching message trees, artifact offload for large tool outputs and images, and FTS5 search across block types; shares the same file with parent-app tables
- **Built-in auth** — static API keys, GitHub device flow for Copilot, device and web/PKCE OAuth for OpenAI, with file-backed storage and expiry handling
- **Tool system** — register Go functions as tools with auto-generated JSON Schema
- **ReAct agent loop** — streaming `Step`/`Run` with per-block hooks, plus runtime provider/model swap
- **Model discovery** — list and switch between every chat-capable model on every authenticated provider
- **Optional adapters** — Bubbletea TUI (with slash commands) and HTTP/SSE server

## Install

```bash
go get github.com/stack-bound/stackllm
```

## Quick start

The fastest path uses the `profile` manager to handle auth and model selection for you. On the first call it walks the user through login; after that, `LoadDefault` returns a ready-to-use provider.

```go
package main

import (
    "context"
    "fmt"

    "github.com/stack-bound/stackllm/agent"
    "github.com/stack-bound/stackllm/conversation"
    "github.com/stack-bound/stackllm/profile"
    "github.com/stack-bound/stackllm/tools"
)

type GreetArgs struct {
    Name string `json:"name" jsonschema:"description=Name to greet,required"`
}

func main() {
    ctx := context.Background()

    // 1. Load the persisted default provider. Run `go run ./examples/login`
    //    once beforehand to pick a provider and model interactively.
    mgr := profile.New()
    p, err := mgr.LoadDefault(ctx)
    if err != nil {
        panic(err)
    }

    // 2. Register tools.
    registry := tools.NewRegistry()
    registry.Register("greet", "Greet someone", func(ctx context.Context, args GreetArgs) (string, error) {
        return fmt.Sprintf("Hello, %s!", args.Name), nil
    })

    // 3. Create the agent.
    a := agent.New(p,
        agent.WithTools(registry),
        agent.WithMaxSteps(10),
        agent.WithHooks(agent.Hooks{
            OnToken: func(ctx context.Context, delta string) { fmt.Print(delta) },
        }),
    )

    // 4. Run.
    msgs := conversation.NewBuilder().
        System("You are a helpful assistant. Use the greet tool when asked.").
        User("Say hello to Alice").
        Build()

    events, _ := a.Run(ctx, msgs)
    for range events {
    }
    fmt.Println()
}
```

If you would rather wire a provider up directly without the manager, skip to [Providers](#providers).

## Provider management

`profile.Manager` composes `auth`, `config`, and `provider` into a single object that knows how to log into any supported backend, discover its models, and persist the user's choice.

```go
mgr := profile.New(profile.WithCallbacks(profile.Callbacks{
    OnDeviceCode: func(userCode, verifyURL string) { /* show code */ },
    OnPromptKey:  func(providerName string) (string, error) { /* read API key */ },
    OnPromptURL:  func(providerName, defaultURL string) (string, error) { /* read URL */ },
}))

// Authenticate — API key for OpenAI/Gemini, GitHub device flow for Copilot,
// base URL prompt for Ollama.
mgr.Login(ctx, profile.ProviderCopilot)

// See which providers are authenticated and which is the default.
statuses, _ := mgr.Status(ctx)

// List chat-capable models across every authenticated provider, sorted.
models, _ := mgr.ListAllModels(ctx)

// Persist the user's choice, preserving any routing metadata.
mgr.SetDefaultModel(models[0])

// Later, in your app:
p, _ := mgr.LoadDefault(ctx)
```

Recently selected models are tracked via `RecentModels` / `TrackRecentModel` so interactive pickers can surface them at the top of the list. Credentials are stored in `~/.config/stackllm/auth.json` and preferences in `~/.config/stackllm/config.json` (or the equivalent `XDG_CONFIG_HOME` path).

## Providers

Wire any supported provider directly if you don't want the manager:

```go
provider.OpenAIConfig("gpt-4o", auth.NewStatic(key))
provider.AzureConfig(endpoint, deployment, apiVersion, tokenSource)
provider.OllamaConfig("http://localhost:11434", "llama3")
provider.CopilotConfig("gpt-4o", auth.NewCopilotSource(cfg))
provider.GeminiConfig("gemini-2.5-pro", auth.NewStatic(key))
```

All five share the same `Complete(ctx, Request)` surface and return a streaming channel of block events (`BlockStart`, `BlockDelta`, `BlockEnd`, `ToolCall`, `Done`, `Error`). Each `BlockEnd` carries the fully accumulated `conversation.Block`; the agent concatenates them in order to build the assistant message, preserving any interleaving of thinking, text, and tool_use the model produced.

## Packages

| Package | Purpose |
|---|---|
| `conversation/` | Block-shaped `Message` types, builder, context compaction |
| `auth/` | Token sources, storage, OAuth flows |
| `config/` | User preferences (default provider/model, provider settings) |
| `profile/` | Provider manager: login, status, model discovery, defaults |
| `tools/` | Tool interface, JSON Schema generation, registry |
| `provider/` | OpenAI-compatible LLM provider |
| `agent/` | ReAct agent loop with hooks |
| `session/` | Session state and persistence |
| `tui/` | Bubbletea terminal UI adapter |
| `web/` | HTTP/SSE server adapter |

## Examples

Runnable examples live in `examples/`.

| Example | Description |
|---|---|
| `examples/login` | Interactive CLI for provider management — login, logout, status, browse models, and set the default. Subcommand shortcuts (`login copilot`, `status`, `models`, `default copilot/gpt-4o`) for scripting. |
| `examples/simple` | Minimal agent with `greet` and `add` tools. Walks the user through provider login and model selection on first run, then uses the persisted default on subsequent runs. |
| `examples/copilot` | Direct Copilot wiring without the manager — shows the two-phase GitHub device flow and caching token source. |
| `examples/tui` | Full Bubbletea TUI agent. Streams tokens as they arrive, renders tool calls and results inline, supports Ctrl+V image paste (inserts a `[Image #N]` placeholder and attaches a `BlockImage` on send), and supports slash commands: `/models` to switch provider/model at runtime (with recently-used models surfaced first), `/new` to start a fresh session. Uses the persisted default or falls back to `OPENAI_API_KEY`. |
| `examples/sqlite` | Shared-DB demo for `session.SQLiteStore`: opens a single SQLite file, runs a parent-app migration (`memories` table), hands the same `*sql.DB` to `session.NewSQLiteStore`, saves a conversation, and queries both namespaces to prove coexistence. No network calls — usable as a CI smoke test. |
| `examples/web` | Browser-only embedding. Serves `web.ManagedHandler` under `/api/*` and a minimal single-page UI at `/` that drives provider login (API keys, Ollama URL, Copilot device flow), model selection, default setting, and streaming chat — all over HTTP with no TUI. |

Run any of them with `go run ./examples/<name>`.

## License

MIT
