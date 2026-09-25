package agent

import (
	"maps"

	"github.com/stack-bound/stackllm/tools"
)

// Option configures an Agent.
type Option func(*options)

type options struct {
	maxSteps        int
	model           string
	temperature     *float64
	maxTokens       int
	reasoningEffort string
	extraBody       map[string]any
	hooks           Hooks
	registry        *tools.Registry
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

// WithReasoningEffort sets how hard a reasoning model thinks before it
// answers, as one of the provider.ReasoningEffort* levels.
// provider.ReasoningEffortNone skips thinking, which is what a
// latency-sensitive caller wants; unset leaves the model on its own
// default, which for the gpt-5 family is several seconds of thinking on
// every turn.
func WithReasoningEffort(effort string) Option {
	return func(o *options) { o.reasoningEffort = effort }
}

// WithExtraBody sets vendor-specific top-level fields sent in every
// request body, such as OpenRouter's provider routing object:
//
//	agent.WithExtraBody(map[string]any{
//		"provider": map[string]any{"order": []string{"groq"}, "allow_fallbacks": false},
//	})
//
// They are passed as provider.Request.ExtraBody, so they override the
// provider's Config.ExtraBody per key. The map is copied (shallowly),
// so later changes to the caller's map do not leak into running calls.
func WithExtraBody(extra map[string]any) Option {
	return func(o *options) { o.extraBody = maps.Clone(extra) }
}

// WithHooks sets the agent hooks.
func WithHooks(h Hooks) Option {
	return func(o *options) { o.hooks = h }
}

// WithTools sets the tool registry.
func WithTools(r *tools.Registry) Option {
	return func(o *options) { o.registry = r }
}
