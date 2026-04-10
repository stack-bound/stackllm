package tui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/x/cellbuf"
	tea "github.com/charmbracelet/bubbletea"

	"github.com/stack-bound/stackllm/agent"
	"github.com/stack-bound/stackllm/provider"
	"github.com/stack-bound/stackllm/session"
)

func newTestModel(t *testing.T) *Model {
	t.Helper()
	p := provider.New(provider.OllamaConfig("http://localhost", "test"))
	a := agent.New(p)
	return New(a, session.NewInMemoryStore())
}

func countVisualLines(text string, width int) int {
	visual := 0
	for _, line := range strings.Split(text, "\n") {
		if len(line) == 0 {
			visual++
		} else {
			visual += strings.Count(cellbuf.Wrap(line, width, ""), "\n") + 1
		}
	}
	return visual
}

// TestResizeTextareaGrowsHeight verifies that the textarea grows
// when text wraps, by simulating the real Update flow.
func TestResizeTextareaGrowsHeight(t *testing.T) {
	t.Parallel()
	m := newTestModel(t)

	updated, _ := m.Update(tea.WindowSizeMsg{Width: 80, Height: 40})
	m = updated.(*Model)

	longText := "some text I am writing sdfsd fds f sfds fdssssssssssssssssssssssssssssssssssdfffffffffff dsfffffffffffffffffff sdfffffffffffffffffff sdffffffffffff dsfffffffffff dfssdf sdf sdfds dssdf sdf sd"
	m.textarea.SetValue(longText)
	m.resizeTextarea()

	predicted := countVisualLines(longText, m.textarea.Width())

	if m.textarea.Height() <= 1 {
		t.Errorf("textarea height should be > 1 for wrapped text, got %d", m.textarea.Height())
	}
	if m.textarea.Height() != predicted {
		t.Errorf("textarea height %d != predicted %d", m.textarea.Height(), predicted)
	}
}

// TestPromptIsEmpty verifies the prompt is removed so textarea.Width()
// equals the full terminal width.
func TestPromptIsEmpty(t *testing.T) {
	t.Parallel()
	m := newTestModel(t)

	if m.textarea.Prompt != "" {
		t.Errorf("expected empty prompt, got %q", m.textarea.Prompt)
	}

	updated, _ := m.Update(tea.WindowSizeMsg{Width: 100, Height: 40})
	m = updated.(*Model)

	if m.textarea.Width() != 100 {
		t.Errorf("expected Width()=100 with empty prompt, got %d", m.textarea.Width())
	}
}

// TestResizeTextareaAfterKeypress simulates individual keypresses
// to verify the textarea grows during typing.
func TestResizeTextareaAfterKeypress(t *testing.T) {
	t.Parallel()
	m := newTestModel(t)

	updated, _ := m.Update(tea.WindowSizeMsg{Width: 40, Height: 30})
	m = updated.(*Model)

	text := "the quick brown fox jumps over the lazy dog and keeps running"
	for _, r := range text {
		updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
		m = updated.(*Model)
	}

	predicted := countVisualLines(m.textarea.Value(), m.textarea.Width())

	if m.textarea.Height() != predicted {
		t.Errorf("textarea height %d != predicted %d after typing", m.textarea.Height(), predicted)
	}
	if m.textarea.Height() <= 1 {
		t.Errorf("textarea height should be > 1 for %d chars at width %d, got %d",
			len(text), m.textarea.Width(), m.textarea.Height())
	}
}
