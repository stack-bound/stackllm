package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// modalKind discriminates between the two modal flavours the text
// modal supports. Both share the same overlay chrome, input, and
// submit/cancel loop — only the title, prompt, and submit action
// differ.
type modalKind int

const (
	modalRename modalKind = iota
	modalExport
)

// modal styles, mirroring the existing picker colour palette.
var (
	modalBorderStyle = lipgloss.NewStyle().
				Border(lipgloss.RoundedBorder()).
				BorderForeground(lipgloss.Color("12")).
				Padding(1, 2)

	modalTitleStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("12")).
			Bold(true)

	modalHintStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("8")).
			Italic(true)

	modalPromptStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("7"))
)

// openRenameModal transitions the TUI into a pre-populated rename
// modal. The input is seeded with the current session name so the
// user can tweak instead of retyping.
func (m *Model) openRenameModal() tea.Cmd {
	m.modalKind = modalRename
	m.modalTitle = "Rename session"
	m.modalPrompt = "New name:"
	m.configureModalInput("session name", m.session.Name)
	m.state = stateTextModal
	return textinput.Blink
}

// openExportModal transitions into a pre-populated export modal. The
// input defaults to a sensible filename in the user's home directory
// so Enter is an immediate accept path.
func (m *Model) openExportModal() tea.Cmd {
	if m.exporter == nil {
		m.appendOutput(m.errorStyle.Render("✗ /export: store does not support export (need SQLite)") + "\n\n")
		return nil
	}
	m.modalKind = modalExport
	m.modalTitle = "Export session"
	m.modalPrompt = "Path:"
	short := m.session.ID
	if len(short) > 8 {
		short = short[:8]
	}
	m.configureModalInput("~/stackllm-<id>.jsonl", fmt.Sprintf("~/stackllm-%s.jsonl", short))
	m.state = stateTextModal
	return textinput.Blink
}

// configureModalInput resets the shared textinput for a new modal
// invocation with the given placeholder and initial value.
func (m *Model) configureModalInput(placeholder, value string) {
	ti := textinput.New()
	ti.Placeholder = placeholder
	ti.CharLimit = 200
	ti.SetValue(value)
	// Width the input to fill most of the terminal width so long
	// paths don't immediately truncate. Clamp to a reasonable max.
	w := m.width - 12
	if w < 20 {
		w = 20
	}
	if w > 80 {
		w = 80
	}
	ti.Width = w
	ti.Focus()
	m.modalInput = ti
}

// submitModal consumes the current modal input, dispatches the action
// matching the active modalKind, and returns to idle. Success/error
// feedback is pushed to the viewport by the submit handlers.
func (m *Model) submitModal() tea.Cmd {
	value := m.modalInput.Value()
	kind := m.modalKind
	m.closeModal()
	switch kind {
	case modalRename:
		m.submitRename(value)
	case modalExport:
		m.submitExport(value)
	}
	return nil
}

// closeModal clears the modal state and returns the TUI to idle.
func (m *Model) closeModal() {
	m.modalInput.Blur()
	m.modalInput.SetValue("")
	m.modalTitle = ""
	m.modalPrompt = ""
	m.state = stateIdle
}

// renderModal draws the centered modal composition for the current
// state. The rest of the UI is hidden behind whitespace to focus the
// user on the input; View() already bypasses the normal chrome when
// stateTextModal is active.
func (m *Model) renderModal() string {
	if m.width == 0 || m.height == 0 {
		return ""
	}

	// Build the body — title, blank, prompt+input, blank, hint — and
	// wrap it in the bordered box style.
	var body strings.Builder
	body.WriteString(modalTitleStyle.Render(m.modalTitle))
	body.WriteString("\n\n")
	body.WriteString(modalPromptStyle.Render(m.modalPrompt))
	body.WriteString(" ")
	body.WriteString(m.modalInput.View())
	body.WriteString("\n\n")
	body.WriteString(modalHintStyle.Render("enter save · esc cancel"))

	box := modalBorderStyle.Render(body.String())

	return lipgloss.Place(
		m.width, m.height,
		lipgloss.Center, lipgloss.Center,
		box,
	)
}
