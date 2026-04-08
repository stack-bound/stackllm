package agent

import "github.com/stack-bound/stackllm/tools"

// Option configures an Agent.
type Option func(*options)

type options struct {
	maxSteps    int
	model       string
	temperature *float64
	maxTokens   int
	hooks       Hooks
	registry    *tools.Registry
}

func defaultOptions() options {
	return options{
		maxSteps: 20,
		registry: tools.NewRegistry(),
	}
}

// WithMaxSteps sets the maximum number of agent loop iterations.
func WithMaxSteps(n int) Option {
	return func(o *options) { o.maxSteps = n }
}

// WithModel overrides the provider's default model.
func WithModel(model string) Option {
	return func(o *options) { o.model = model }
}

// WithTemperature sets the sampling temperature.
func WithTemperature(t float64) Option {
	return func(o *options) { o.temperature = &t }
}

// WithMaxTokens sets the maximum output tokens.
func WithMaxTokens(n int) Option {
	return func(o *options) { o.maxTokens = n }
}

// WithHooks sets the agent hooks.
func WithHooks(h Hooks) Option {
	return func(o *options) { o.hooks = h }
}

// WithTools sets the tool registry.
func WithTools(r *tools.Registry) Option {
	return func(o *options) { o.registry = r }
}
