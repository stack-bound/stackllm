// Command tui demonstrates the Bubbletea interactive terminal agent.
//
// Usage:
//
//	export OPENAI_API_KEY=sk-...
//	go run ./examples/tui
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/stack-bound/stackllm/agent"
	"github.com/stack-bound/stackllm/auth"
	"github.com/stack-bound/stackllm/provider"
	"github.com/stack-bound/stackllm/session"
	"github.com/stack-bound/stackllm/tools"
	"github.com/stack-bound/stackllm/tui"

	tea "github.com/charmbracelet/bubbletea"
)

type TimeArgs struct{}

func main() {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		fmt.Fprintln(os.Stderr, "Set OPENAI_API_KEY environment variable")
		os.Exit(1)
	}

	// Provider.
	p := provider.New(provider.OpenAIConfig("gpt-4o", auth.NewStatic(key)))

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

	// TUI.
	store := session.NewInMemoryStore()
	m := tui.New(a, store)

	prog := tea.NewProgram(m, tea.WithAltScreen())
	if _, err := prog.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
