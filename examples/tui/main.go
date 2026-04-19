// Command tui demonstrates the Bubbletea interactive terminal agent.
//
// Uses the default provider if configured (via go run ./examples/login),
// otherwise falls back to OPENAI_API_KEY environment variable.
//
// To sign in with a ChatGPT account (no API key needed), run:
//
//	go run ./examples/login login openai device   # headless device code
//	go run ./examples/login login openai web      # browser PKCE flow
//	go run ./examples/login login openai key      # paste an API key
//
// Then pick a model with:
//
//	go run ./examples/login default
//
// Usage:
//
//	go run ./examples/tui
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/stack-bound/stackllm/agent"
	"github.com/stack-bound/stackllm/auth"
	"github.com/stack-bound/stackllm/profile"
	"github.com/stack-bound/stackllm/provider"
	"github.com/stack-bound/stackllm/session"
	"github.com/stack-bound/stackllm/tools"
	"github.com/stack-bound/stackllm/tui"

	tea "github.com/charmbracelet/bubbletea"
)

type TimeArgs struct{}

func main() {
	mgr := profile.New()
	p, err := resolveProvider(mgr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		fmt.Fprintln(os.Stderr, "Either run: go run ./examples/login")
		fmt.Fprintln(os.Stderr, "Or set:    export OPENAI_API_KEY=sk-...")
		os.Exit(1)
	}

	// Tools.
	registry := tools.NewRegistry()
	registry.Register("current_time", "Get the current time", func(ctx context.Context, args TimeArgs) (string, error) {
		return "2024-01-15 14:30:00 UTC", nil
	})

	// Agent with system prompt baked into hooks.
	a := agent.New(p,
		agent.WithTools(registry),
		agent.WithMaxSteps(10),
	)

	// TUI. Use a persistent SQLite-backed store so session history
	// survives across runs and /sessions, /fork, /export all work.
	store, err := session.OpenSQLiteStore(session.SQLiteConfig{AppName: "stackllm-tui-example"})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening session store: %v\n", err)
		os.Exit(1)
	}
	defer store.Close()

	m := tui.New(a, store, tui.WithModelLister(mgr))

	prog := tea.NewProgram(m, tea.WithAltScreen())
	if _, err := prog.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// resolveProvider tries the persisted default first, then OPENAI_API_KEY.
func resolveProvider(mgr *profile.Manager) (*provider.OpenAIProvider, error) {
	p, err := mgr.LoadDefault(context.Background())
	if err == nil {
		return p, nil
	}

	// Fallback: OPENAI_API_KEY environment variable.
	if key := os.Getenv("OPENAI_API_KEY"); key != "" {
		return provider.New(provider.OpenAIConfig("gpt-5.4", auth.NewStatic(key))), nil
	}

	return nil, fmt.Errorf("no default provider configured and OPENAI_API_KEY not set")
}
