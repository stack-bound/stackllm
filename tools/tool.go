package tools

import "context"

// Definition is what gets sent to the LLM in the tools array.
type Definition struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// Tool is anything that can be called by the agent loop.
type Tool interface {
	Definition() Definition
	Call(ctx context.Context, arguments string) (string, error)
}
