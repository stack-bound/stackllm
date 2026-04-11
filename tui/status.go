package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"

	"github.com/stack-bound/stackllm/conversation"
)

// statusSuffixStyle dims the model + context info on the right of the
// status line so it reads as chrome rather than content.
var statusSuffixStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("8"))

// formatModelStatus renders the right-aligned status suffix: model
// name, token count, max context window, and percent used. The branch
// structure encodes "what do we know":
//
//   - no model           → empty string (caller omits the suffix)
//   - no usage + no CW   → model only
//   - no usage + CW      → model + "0 / MAX (0.0%)"
//   - usage + no CW      → model + "N tok"
//   - usage + CW         → model + "N / MAX (p%)"
//
// It does not apply any styling — callers colour the result as they
// prefer. Thousands-grouping uses plain commas; the "·" separator is
// U+00B7 (middle dot) to keep the row compact.
func formatModelStatus(model string, usage *conversation.TokenUsage, window int) string {
	if model == "" {
		return ""
	}
	if usage == nil && window <= 0 {
		return model
	}
	if usage == nil {
		// Context window known but no usage yet — show zero so the
		// row shape stays stable once the first turn lands.
		return fmt.Sprintf("%s · %s / %s (0.0%%)", model, formatInt(0), formatInt(window))
	}
	if window <= 0 {
		return fmt.Sprintf("%s · %s tok", model, formatInt(usage.PromptTokens))
	}
	pct := float64(usage.PromptTokens) / float64(window) * 100
	return fmt.Sprintf("%s · %s / %s (%.1f%%)", model, formatInt(usage.PromptTokens), formatInt(window), pct)
}

// formatInt groups an int with commas every three digits. Negative
// values are passed through with the sign preserved.
func formatInt(n int) string {
	if n < 0 {
		return "-" + formatInt(-n)
	}
	s := fmt.Sprintf("%d", n)
	if len(s) <= 3 {
		return s
	}
	// Walk backwards inserting a comma every third digit.
	var b strings.Builder
	rem := len(s) % 3
	if rem > 0 {
		b.WriteString(s[:rem])
		if len(s) > rem {
			b.WriteString(",")
		}
	}
	for i := rem; i < len(s); i += 3 {
		b.WriteString(s[i : i+3])
		if i+3 < len(s) {
			b.WriteString(",")
		}
	}
	return b.String()
}

// padBetween returns left and right joined by enough spaces to fill
// `width` visible columns. If the combined visible width exceeds
// `width`, the right half is truncated from the end until it fits
// (with an ellipsis when truncation happens). When `width` is 0 or
// negative the two strings are joined with a single space — Bubbletea
// passes a zero width on the very first render before WindowSize has
// arrived.
//
// Visible width is measured via lipgloss.Width, which ignores ANSI
// escapes — this is critical because `left` typically carries the
// status-line colour codes.
func padBetween(left, right string, width int) string {
	if right == "" {
		return left
	}
	if left == "" {
		return right
	}
	if width <= 0 {
		return left + " " + right
	}
	lw := lipgloss.Width(left)
	rw := lipgloss.Width(right)
	if lw+rw+1 <= width {
		pad := width - lw - rw
		return left + strings.Repeat(" ", pad) + right
	}
	// Not enough room for both. Keep the left half intact and
	// truncate the right to what's left after one separating space.
	budget := width - lw - 1
	if budget <= 1 {
		// No room for even a minimal right half; drop it.
		return left
	}
	// Truncate the right string visibly — since the right side is
	// plain ASCII (no ANSI escapes in formatModelStatus's output),
	// byte length equals visible width and a simple slice is safe.
	if budget < len(right) {
		if budget <= 1 {
			return left
		}
		right = right[:budget-1] + "…"
	}
	return left + " " + right
}
