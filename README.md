# stackllm

A minimal, embeddable Go agent orchestration library. Single binary. Full context control.

You own the `[]Message` — the library provides the loop and the plumbing.

## Features

- **Single OpenAI-compatible provider** — works with OpenAI, Azure OpenAI, Ollama, GitHub Copilot, and Gemini
- **Full context control** — the caller owns the message slice and decides what goes in it
- **Built-in auth** — device flow and web flow OAuth for Copilot and OpenAI
- **Tool system** — register Go functions as tools with auto-generated JSON Schema
- **ReAct agent loop** — streaming Step/Run with hooks at every stage
- **Optional adapters** — Bubbletea TUI and HTTP/SSE server

## Install

```bash
go get github.com/stack-bound/stackllm
```

## Quick start

```go
package main

import (
    "context"
    "fmt"
    "os"

    "github.com/stack-bound/stackllm/agent"
    "github.com/stack-bound/stackllm/auth"
    "github.com/stack-bound/stackllm/conversation"
    "github.com/stack-bound/stackllm/provider"
    "github.com/stack-bound/stackllm/tools"
)

func main() {
    // 1. Set up provider
    p := provider.New(provider.OpenAIConfig(
        "gpt-4o",
        auth.NewStatic(os.Getenv("OPENAI_API_KEY")),
    ))

    // 2. Register tools
    type GreetArgs struct {
        Name string `json:"name" jsonschema:"description=Name to greet,required"`
    }
    registry := tools.NewRegistry()
    registry.Register("greet", "Greet someone", func(ctx context.Context, args GreetArgs) (string, error) {
        return fmt.Sprintf("Hello, %s!", args.Name), nil
    })

    // 3. Create agent
    a := agent.New(p,
        agent.WithTools(registry),
        agent.WithMaxSteps(10),
        agent.WithHooks(agent.Hooks{
            OnToken: func(ctx context.Context, delta string) {
                fmt.Print(delta)
            },
        }),
    )

    // 4. Run
    msgs := conversation.NewBuilder().
        System("You are a helpful assistant. Use the greet tool when asked.").
        User("Say hello to Alice").
        Build()

    events, _ := a.Run(context.Background(), msgs)
    for range events {
    }
    fmt.Println()
}
```

## Packages

| Package | Purpose |
|---|---|
| `conversation/` | Message types, builder, context compaction |
| `auth/` | Token sources, storage, OAuth flows |
| `tools/` | Tool interface, JSON Schema gen, registry |
| `provider/` | OpenAI-compatible LLM provider |
| `agent/` | ReAct agent loop with hooks |
| `session/` | Session state and persistence |
| `tui/` | Bubbletea terminal UI adapter |
| `web/` | HTTP/SSE server adapter |

## Providers

```go
provider.OpenAIConfig("gpt-4o", auth.NewStatic(key))
provider.AzureConfig(endpoint, deployment, apiVersion, tokenSource)
provider.OllamaConfig("http://localhost:11434", "llama3")
provider.CopilotConfig("gpt-4o", auth.NewCopilotSource(cfg))
provider.GeminiConfig("gemini-2.5-pro", auth.NewStatic(key))
```

See [CLAUDE.md](CLAUDE.md) for full documentation.

## License

MIT
