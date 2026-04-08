// Command simple demonstrates a minimal agent that can greet people and do math.
//
// On first run it walks you through provider login and model selection.
// Subsequent runs use the persisted default automatically.
//
// Usage:
//
//	go run ./examples/simple
package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/stack-bound/stackllm/agent"
	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/profile"
	"github.com/stack-bound/stackllm/provider"
	"github.com/stack-bound/stackllm/tools"
)

type GreetArgs struct {
	Name string `json:"name" jsonschema:"description=Name to greet,required"`
}

type AddArgs struct {
	A float64 `json:"a" jsonschema:"description=First number,required"`
	B float64 `json:"b" jsonschema:"description=Second number,required"`
}

func main() {
	ctx := context.Background()
	scanner := bufio.NewScanner(os.Stdin)

	mgr := profile.New(profile.WithCallbacks(profile.Callbacks{
		OnDeviceCode: func(userCode, verifyURL string) {
			fmt.Printf("\nOpen %s and enter code: %s\n\n", verifyURL, userCode)
		},
		OnPolling: func() {
			fmt.Print(".")
		},
		OnSuccess: func() {
			fmt.Println("\nAuthenticated!")
		},
		OnPromptKey: func(providerName string) (string, error) {
			fmt.Printf("Enter API key for %s: ", providerName)
			if !scanner.Scan() {
				return "", fmt.Errorf("no input")
			}
			return strings.TrimSpace(scanner.Text()), nil
		},
		OnPromptURL: func(providerName, defaultURL string) (string, error) {
			fmt.Printf("Enter base URL for %s [%s]: ", providerName, defaultURL)
			if !scanner.Scan() {
				return defaultURL, nil
			}
			v := strings.TrimSpace(scanner.Text())
			if v == "" {
				return defaultURL, nil
			}
			return v, nil
		},
	}))

	// Try loading the default provider.
	p, err := mgr.LoadDefault(ctx)
	if err != nil {
		// No default set — walk the user through setup.
		fmt.Println("No default provider configured. Let's set one up.")
		p, err = interactiveSetup(ctx, mgr, scanner)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Setup failed: %v\n", err)
			os.Exit(1)
		}
	}

	runAgent(ctx, p)
}

func interactiveSetup(ctx context.Context, mgr *profile.Manager, scanner *bufio.Scanner) (*provider.OpenAIProvider, error) {
	// Step 1: Pick a provider to log in to.
	providers := mgr.AvailableProviders()
	fmt.Println()
	fmt.Println("Available providers:")
	for i, p := range providers {
		fmt.Printf("  %d) %s\n", i+1, p)
	}
	fmt.Println()
	fmt.Print("Choose provider: ")

	if !scanner.Scan() {
		return nil, fmt.Errorf("no input")
	}
	idx, err := strconv.Atoi(strings.TrimSpace(scanner.Text()))
	if err != nil || idx < 1 || idx > len(providers) {
		return nil, fmt.Errorf("invalid choice")
	}
	providerName := providers[idx-1]

	// Step 2: Log in.
	if err := mgr.Login(ctx, providerName); err != nil {
		return nil, fmt.Errorf("login %s: %w", providerName, err)
	}
	fmt.Printf("Logged in to %s.\n", providerName)

	// Step 3: List models and pick one.
	fmt.Println()
	fmt.Println("Fetching models...")
	models, err := mgr.ListModels(ctx, providerName)
	if err != nil {
		return nil, fmt.Errorf("list models: %w", err)
	}

	if len(models) == 0 {
		return nil, fmt.Errorf("no models available for %s", providerName)
	}

	fmt.Println()
	fmt.Println("Available models:")
	for i, m := range models {
		fmt.Printf("  %d) %s\n", i+1, m)
	}
	fmt.Println()
	fmt.Print("Choose model [1]: ")

	if !scanner.Scan() {
		return nil, fmt.Errorf("no input")
	}
	input := strings.TrimSpace(scanner.Text())
	if input == "" {
		input = "1"
	}
	midx, err := strconv.Atoi(input)
	if err != nil || midx < 1 || midx > len(models) {
		return nil, fmt.Errorf("invalid choice")
	}
	model := models[midx-1]

	// Step 4: Set as default.
	defaultStr := providerName + "/" + model
	if err := mgr.SetDefault(defaultStr); err != nil {
		return nil, fmt.Errorf("set default: %w", err)
	}
	fmt.Printf("\nDefault set: %s\n\n", defaultStr)

	// Step 5: Load the provider.
	return mgr.LoadDefault(ctx)
}

func runAgent(ctx context.Context, p *provider.OpenAIProvider) {
	// Tools.
	registry := tools.NewRegistry()
	registry.Register("greet", "Greet someone by name", func(ctx context.Context, args GreetArgs) (string, error) {
		return fmt.Sprintf("Hello, %s!", args.Name), nil
	})
	registry.Register("add", "Add two numbers", func(ctx context.Context, args AddArgs) (string, error) {
		return fmt.Sprintf("%g", args.A+args.B), nil
	})

	// Agent.
	a := agent.New(p,
		agent.WithTools(registry),
		agent.WithMaxSteps(5),
		agent.WithHooks(agent.Hooks{
			OnToken: func(ctx context.Context, delta string) {
				fmt.Print(delta)
			},
			OnToolCall: func(ctx context.Context, call conversation.ToolCall) {
				fmt.Printf("\n[tool: %s(%s)]\n", call.Name, call.Arguments)
			},
			OnToolResult: func(ctx context.Context, call conversation.ToolCall, result string, err error) {
				fmt.Printf("[result: %s]\n", result)
			},
		}),
	)

	// Conversation.
	msgs := conversation.NewBuilder().
		System("You are a helpful assistant. Use the available tools when appropriate.").
		User("Greet Alice, then add 17 and 25.").
		Build()

	// Run.
	events, err := a.Run(ctx, msgs)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	for range events {
	}
	fmt.Println()
}
