package provider

import "strings"

// KnownContextWindow is one entry in the hardcoded fallback table used
// when a provider's /models endpoint does not expose per-model context
// length. The prefix is matched case-insensitively against the start
// of the model name so that versioned suffixes (e.g.
// "gpt-4o-2024-08-06") resolve to the base family. Entries are
// consulted in order, so longer / more specific prefixes must precede
// shorter ones to win the match (e.g. "gpt-4o-mini" before "gpt-4o").
type KnownContextWindow struct {
	Prefix string
	Tokens int
}

// knownContextWindows is the ordered fallback table. Keep longer
// prefixes ahead of the families they share a stem with so prefix
// matching picks the more specific entry first.
var knownContextWindows = []KnownContextWindow{
	// OpenAI
	{Prefix: "gpt-4o-mini", Tokens: 128_000},
	{Prefix: "gpt-4o", Tokens: 128_000},
	{Prefix: "gpt-4-turbo", Tokens: 128_000},
	{Prefix: "gpt-4.1", Tokens: 1_000_000},
	{Prefix: "gpt-4", Tokens: 8_192},
	{Prefix: "gpt-3.5-turbo-16k", Tokens: 16_385},
	{Prefix: "gpt-3.5-turbo", Tokens: 16_385},
	{Prefix: "gpt-5", Tokens: 200_000},
	{Prefix: "o1-mini", Tokens: 128_000},
	{Prefix: "o1-preview", Tokens: 128_000},
	{Prefix: "o1", Tokens: 200_000},
	{Prefix: "o3-mini", Tokens: 200_000},
	{Prefix: "o3", Tokens: 200_000},
	{Prefix: "o4-mini", Tokens: 200_000},

	// Anthropic (via Copilot, Bedrock proxies, etc.)
	{Prefix: "claude-3-5-sonnet", Tokens: 200_000},
	{Prefix: "claude-3.5-sonnet", Tokens: 200_000},
	{Prefix: "claude-3-5-haiku", Tokens: 200_000},
	{Prefix: "claude-3.5-haiku", Tokens: 200_000},
	{Prefix: "claude-3-opus", Tokens: 200_000},
	{Prefix: "claude-3-sonnet", Tokens: 200_000},
	{Prefix: "claude-3-haiku", Tokens: 200_000},
	{Prefix: "claude-sonnet-4", Tokens: 200_000},
	{Prefix: "claude-opus-4", Tokens: 200_000},
	{Prefix: "claude-haiku-4", Tokens: 200_000},

	// Google Gemini
	{Prefix: "gemini-1.5-pro", Tokens: 2_000_000},
	{Prefix: "gemini-1.5-flash", Tokens: 1_000_000},
	{Prefix: "gemini-2.0-flash", Tokens: 1_000_000},
	{Prefix: "gemini-2.5-pro", Tokens: 2_000_000},
	{Prefix: "gemini-2.5-flash", Tokens: 1_000_000},
	{Prefix: "gemini-pro", Tokens: 32_760},

	// Open-weights (Ollama defaults — embedders can override via
	// provider metadata where the runtime actually knows.)
	{Prefix: "llama3.1", Tokens: 131_072},
	{Prefix: "llama3.2", Tokens: 131_072},
	{Prefix: "llama3", Tokens: 8_192},
	{Prefix: "mistral", Tokens: 32_768},
	{Prefix: "mixtral", Tokens: 32_768},
	{Prefix: "qwen2.5", Tokens: 131_072},
	{Prefix: "qwen2", Tokens: 131_072},
	{Prefix: "deepseek", Tokens: 131_072},
}

// ContextWindow returns the maximum context length in tokens for a
// model by matching its name against a hardcoded prefix table of
// well-known families. It is the fallback path for providers whose
// /models endpoint does not report per-model limits (OpenAI, Gemini,
// Ollama). Returns 0 when the model is unknown; callers should treat
// 0 as "unknown" and degrade their UI accordingly rather than
// guessing.
//
// Matching is case-insensitive and strictly prefix-based so that
// versioned suffixes like "-2024-08-06" or "-preview" resolve to the
// base family.
func ContextWindow(model string) int {
	if model == "" {
		return 0
	}
	m := strings.ToLower(model)
	for _, kw := range knownContextWindows {
		if strings.HasPrefix(m, strings.ToLower(kw.Prefix)) {
			return kw.Tokens
		}
	}
	return 0
}

// KnownContextWindows returns a copy of the fallback table. It exists
// so tests (and any embedder curious about what's known) can iterate
// the table without duplicating the numbers — per the project rule
// that tests should not hardcode values that have a single source of
// truth in production code.
func KnownContextWindows() []KnownContextWindow {
	out := make([]KnownContextWindow, len(knownContextWindows))
	copy(out, knownContextWindows)
	return out
}
