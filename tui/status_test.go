package tui

import (
	"fmt"
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"

	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/provider"
)

// TestFormatModelStatus_AllBranches covers every branch of the
// formatter: empty model, model only, window-only, usage-only,
// and both. The expected strings read the max-context value from
// provider.ContextWindow so the test does not hardcode a token count
// that could drift — consistent with the no-hardcoded-values rule.
func TestFormatModelStatus_AllBranches(t *testing.T) {
	t.Parallel()

	cw := provider.ContextWindow("gpt-4o")
	if cw == 0 {
		t.Fatal("gpt-4o not in the known context-window table; update the test")
	}

	usage := &conversation.TokenUsage{PromptTokens: 12345, CompletionTokens: 200, TotalTokens: 12545}
	pct := float64(usage.PromptTokens) / float64(cw) * 100

	cases := []struct {
		name   string
		model  string
		usage  *conversation.TokenUsage
		window int
		want   string
	}{
		{
			name: "empty model returns empty",
			want: "",
		},
		{
			name:  "model only",
			model: "gpt-4o",
			want:  "gpt-4o",
		},
		{
			name:   "window but no usage",
			model:  "gpt-4o",
			window: cw,
			want:   fmt.Sprintf("gpt-4o · 0 / %s (0.0%%)", formatInt(cw)),
		},
		{
			name:  "usage but no window",
			model: "custom-thing",
			usage: usage,
			want:  "custom-thing · 12,345 tok",
		},
		{
			name:   "usage and window",
			model:  "gpt-4o",
			usage:  usage,
			window: cw,
			want:   fmt.Sprintf("gpt-4o · 12,345 / %s (%.1f%%)", formatInt(cw), pct),
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := formatModelStatus(tc.model, tc.usage, tc.window)
			if got != tc.want {
				t.Errorf("formatModelStatus = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestFormatInt_ThousandsGrouping asserts the thousands separator
// handling across one-, three-, four-, and seven-digit inputs, plus
// the negative case. Prevents an off-by-one in the rem / i loop.
func TestFormatInt_ThousandsGrouping(t *testing.T) {
	t.Parallel()

	cases := map[int]string{
		0:       "0",
		7:       "7",
		42:      "42",
		999:     "999",
		1000:    "1,000",
		12345:   "12,345",
		123456:  "123,456",
		1234567: "1,234,567",
		-12345:  "-12,345",
	}
	for in, want := range cases {
		if got := formatInt(in); got != want {
			t.Errorf("formatInt(%d) = %q, want %q", in, got, want)
		}
	}
}

// TestPadBetween_ANSISafe uses an ANSI-coloured left half to prove
// the padding math uses visual width (via lipgloss.Width), not byte
// length. If padBetween counted escape bytes, the total row would be
// wider than `width` and wrap in the terminal.
func TestPadBetween_ANSISafe(t *testing.T) {
	t.Parallel()

	left := lipgloss.NewStyle().Foreground(lipgloss.Color("12")).Render("● ready")
	right := "gpt-4o · 12,345 / 128,000 (9.6%)"
	const width = 80

	got := padBetween(left, right, width)
	if w := lipgloss.Width(got); w != width {
		t.Errorf("padBetween visible width = %d, want %d", w, width)
	}
	if !strings.HasSuffix(stripANSI(got), right) {
		t.Errorf("padBetween output did not end with right half: %q", got)
	}
}

// TestPadBetween_Truncates shrinks the terminal below the combined
// visible width and asserts the right half is truncated rather than
// allowed to overflow. The left half must always survive intact so
// the state label stays readable.
func TestPadBetween_Truncates(t *testing.T) {
	t.Parallel()

	left := "● ready"
	right := "gpt-4o · 12,345 / 128,000 (9.6%)"
	// Enough room for left + one space + a few chars of right.
	width := len(left) + 8

	got := padBetween(left, right, width)
	if w := lipgloss.Width(got); w > width {
		t.Errorf("padBetween visible width = %d, exceeds %d", w, width)
	}
	if !strings.HasPrefix(got, left) {
		t.Errorf("left half was truncated: %q", got)
	}
}

// TestPadBetween_ZeroWidth covers the pre-resize case where m.width
// is still 0: the function should still produce a usable string
// rather than an empty line.
func TestPadBetween_ZeroWidth(t *testing.T) {
	t.Parallel()

	got := padBetween("● ready", "gpt-4o", 0)
	if got != "● ready gpt-4o" {
		t.Errorf("padBetween(0 width) = %q, want '● ready gpt-4o'", got)
	}
}

// stripANSI removes CSI-style escape sequences from a lipgloss-styled
// string so assertions can look at the visible characters directly.
// Narrow enough for test use only — do not promote to production.
func stripANSI(s string) string {
	var b strings.Builder
	in := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == 0x1b {
			in = true
			continue
		}
		if in {
			if (c >= '@' && c <= '~') && c != '[' {
				in = false
			}
			continue
		}
		b.WriteByte(c)
	}
	return b.String()
}
