// Command simple demonstrates a minimal agent that can greet people and do math.
//
// Usage:
//
//	export OPENAI_API_KEY=sk-...
//	go run ./examples/simple
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

type GreetArgs struct {
	Name string `json:"name" jsonschema:"description=Name to greet,required"`
}

type AddArgs struct {
	A float64 `json:"a" jsonschema:"description=First number,required"`
	B float64 `json:"b" jsonschema:"description=Second number,required"`
}

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
	events, err := a.Run(context.Background(), msgs)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	for range events {
	}
	fmt.Println()
}
