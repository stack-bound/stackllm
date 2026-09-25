package tui

import (
	"context"
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/stack-bound/stackllm/provider"
)

// EffortStore persists the reasoning effort picked with /effort so the
// next session starts with it. profile.Manager satisfies it.
type EffortStore interface {
	ReasoningEffort(ctx context.Context) (string, error)
	SetReasoningEffort(ctx context.Context, effort string) error
}

// WithEffortStore injects an EffortStore. New applies the persisted
// effort to the agent when one has been saved, overriding whatever the
// agent was built with, and /effort saves each new choice to it.
// Without a store /effort still works but only for the current run.
func WithEffortStore(s EffortStore) Option {
	return func(m *Model) { m.effortStore = s }
}

// effortChoice is one row of the /effort picker. An empty level is the
// "default" row, which stops sending an effort at all.
type effortChoice struct {
	level string
	hint  string
}

// effortChoices lists the picker rows: "default" first, then every
// provider level from least to most thinking.
func effortChoices() []effortChoice {
	hints := map[string]string{
		provider.ReasoningEffortNone:    "no thinking, fastest",
		provider.ReasoningEffortMinimal: "barely any thinking (gpt-5 / gpt-5.1)",
		provider.ReasoningEffortLow:     "quick thinking",
		provider.ReasoningEffortMedium:  "balanced",
		provider.ReasoningEffortHigh:    "thorough, slower",
		provider.ReasoningEffortXHigh:   "very thorough (gpt-5.6 onwards)",
		provider.ReasoningEffortMax:     "as much as it takes (gpt-5.6 onwards)",
	}
	out := []effortChoice{{level: "", hint: "let the model decide"}}
	for _, level := range provider.ReasoningEffortLevels() {
		out = append(out, effortChoice{level: level, hint: hints[level]})
	}
	return out
}

// effortLabel is how a level is shown to the user.
func effortLabel(level string) string {
	if level == "" {
		return "default"
	}
	return level
}

// applyStoredEffort copies the persisted effort onto the agent. A load
// error is shown in the scrollback rather than failing New, since the
// TUI is still usable on the agent's own effort.
func (m *Model) applyStoredEffort() {
	if m.effortStore == nil {
		return
	}
	effort, err := m.effortStore.ReasoningEffort(context.Background())
	if err != nil {
		m.appendOutput(m.errorStyle.Render("Warning: failed to load reasoning effort: "+err.Error()) + "\n\n")
		return
	}
	if effort != "" {
		m.agent.SetReasoningEffort(effort)
	}
}

// openEffortPicker shows the /effort picker with the cursor on the
// agent's current level, so Enter straight away changes nothing.
func (m *Model) openEffortPicker() tea.Cmd {
	m.effortChoices = effortChoices()
	m.effortCursor = 0
	current := m.agent.ReasoningEffort()
	for i, c := range m.effortChoices {
		if c.level == current {
			m.effortCursor = i
			break
		}
	}
	m.state = stateEffortPicker
	return nil
}

// closeEffortPicker returns to idle without changing anything.
func (m *Model) closeEffortPicker() {
	m.effortChoices = nil
	m.effortCursor = 0
	m.state = stateIdle
}

// selectEffort applies the highlighted level to the agent and saves it.
// Enter is ignored while a run is in progress, so the agent is never
// mutated under a live Run.
func (m *Model) selectEffort() {
	if len(m.effortChoices) == 0 {
		m.closeEffortPicker()
		return
	}
	level := m.effortChoices[m.effortCursor].level
	m.closeEffortPicker()
	m.agent.SetReasoningEffort(level)
	if m.effortStore != nil {
		if err := m.effortStore.SetReasoningEffort(context.Background(), level); err != nil {
			m.appendOutput(m.errorStyle.Render("Warning: failed to save reasoning effort: "+err.Error()) + "\n")
		}
	}
	m.appendOutput(m.toolStyle.Render("Reasoning effort: "+effortLabel(level)) + "\n\n")
}

func (m *Model) renderEffortPicker() string {
	current := m.agent.ReasoningEffort()
	width := 0
	for _, c := range m.effortChoices {
		if n := len(effortLabel(c.level)); n > width {
			width = n
		}
	}
	var b strings.Builder
	for i, c := range m.effortChoices {
		line := fmt.Sprintf("%-*s  %s", width, effortLabel(c.level), c.hint)
		if c.level == current {
			line += "  (current)"
		}
		if i == m.effortCursor {
			b.WriteString(m.menuCursorStyle.Render("> " + line))
		} else {
			b.WriteString(m.menuStyle.Render("  " + line))
		}
		if i < len(m.effortChoices)-1 {
			b.WriteString("\n")
		}
	}
	return b.String()
}

// effortErrorHint explains a 400 when an effort is set, because the
// usual cause is a level the current model does not accept (gpt-5
// rejects "xhigh", gpt-5.6 rejects "minimal"). Returns "" otherwise.
func (m *Model) effortErrorHint(err error) string {
	effort := m.agent.ReasoningEffort()
	if err == nil || effort == "" || !strings.Contains(err.Error(), "status 400") {
		return ""
	}
	return fmt.Sprintf("Reasoning effort is %q; if this model does not support it, pick another with /effort.", effort)
}
