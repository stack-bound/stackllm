// Command copilot demonstrates using GitHub Copilot as the LLM provider.
//
// On first run it will prompt you to authenticate via the GitHub device flow.
// The token is persisted to ~/.config/stackllm/auth.json.
//
// Usage:
//
//	go run ./examples/copilot
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/stack-bound/stackllm/agent"
	"github.com/stack-bound/stackllm/auth"
	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/provider"
)

func main() {
	// Auth: Copilot two-phase flow with file-backed token storage.
	store := &auth.FileStore{AppName: "stackllm"}
	tokenSource := auth.NewCachingSource(auth.NewCopilotSource(auth.CopilotConfig{
		Store: store,
		OnDeviceCode: func(userCode, verifyURL string) {
			fmt.Printf("\nOpen %s and enter code: %s\n\n", verifyURL, userCode)
		},
		OnPolling: func() {
			fmt.Print(".")
		},
		OnSuccess: func() {
			fmt.Println("\nAuthenticated!")
		},
	}))

	// Provider: Copilot with GPT-5.4.
	p := provider.New(provider.CopilotConfig("gpt-5.4", tokenSource))

	// Agent.
	a := agent.New(p,
		agent.WithMaxSteps(5),
		agent.WithHooks(agent.Hooks{
			OnToken: func(ctx context.Context, delta string) {
				fmt.Print(delta)
			},
		}),
	)

	// Conversation.
	msgs := conversation.NewBuilder().
		System("You are a helpful coding assistant.").
		User("Explain what a goroutine is in 2 sentences.").
		Build()

	// Run.
	events, err := a.Run(context.Background(), msgs)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	for range events {
	}
	fmt.Println()
}
