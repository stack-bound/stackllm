package tui

import (
	"context"
	"strings"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/cellbuf"

	"github.com/stack-bound/stackllm/agent"
	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/session"
)

type modelState int

const (
	stateIdle modelState = iota
	stateRunning
	stateToolCall
	stateError
)

// agentEventMsg wraps agent events for the Bubbletea update loop.
type agentEventMsg struct {
	event agent.Event
}

// agentDoneMsg signals the agent has finished.
type agentDoneMsg struct{}

// Model is a Bubbletea model that drives an agent.Agent interactively.
type Model struct {
	agent    *agent.Agent
	session  *session.Session
	store    session.SessionStore
	textarea textarea.Model
	viewport viewport.Model
	spinner  spinner.Model
	state    modelState
	output   strings.Builder
	err      error
	width    int
	height   int
	cancel   context.CancelFunc

	// Styles
	userStyle      lipgloss.Style
	assistantStyle lipgloss.Style
	toolStyle      lipgloss.Style
	errorStyle     lipgloss.Style
}

const (
	// chromeHeight is the number of rows used by the status line and newlines
	// between the viewport, status, and textarea in View().
	chromeHeight = 2

	// maxInputHeight is the maximum number of rows the textarea can grow to.
	maxInputHeight = 8
)

// New creates a new TUI Model.
func New(a *agent.Agent, store session.SessionStore) *Model {
	ta := textarea.New()
	ta.Placeholder = "Type a message..."
	ta.Prompt = ""
	ta.Focus()
	ta.SetHeight(1)
	ta.MaxHeight = maxInputHeight
	ta.ShowLineNumbers = false

	vp := viewport.New(80, 20)

	sp := spinner.New()
	sp.Spinner = spinner.Dot
	sp.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("11"))

	return &Model{
		agent:          a,
		session:        session.New(),
		store:          store,
		textarea:       ta,
		viewport:       vp,
		spinner:        sp,
		state:          stateIdle,
		userStyle:      lipgloss.NewStyle().Foreground(lipgloss.Color("12")).Bold(true),
		assistantStyle: lipgloss.NewStyle().Foreground(lipgloss.Color("10")),
		toolStyle:      lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Italic(true),
		errorStyle:     lipgloss.NewStyle().Foreground(lipgloss.Color("9")).Bold(true),
	}
}

// Init implements tea.Model.
func (m *Model) Init() tea.Cmd {
	return tea.Batch(textarea.Blink, m.spinner.Tick)
}

// Update implements tea.Model.
func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd
	var skipTextarea bool

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC:
			if m.cancel != nil {
				m.cancel()
			}
			return m, tea.Quit
		case tea.KeyCtrlJ: // Shift+Enter sends \n (linefeed) in many terminals
			m.textarea.InsertString("\n")
			skipTextarea = true
		case tea.KeyEnter:
			if msg.Alt {
				break // pass Alt+Enter to textarea for newline insertion
			}
			skipTextarea = true // consume plain Enter, don't let textarea add a newline
			if m.state == stateRunning {
				break
			}
			input := strings.TrimSpace(m.textarea.Value())
			if input == "" {
				break
			}
			m.textarea.Reset()
			m.appendOutput(m.userStyle.Render("You: ") + input + "\n\n")
			m.session.AppendMessage(conversation.Message{
				Role:    conversation.RoleUser,
				Content: input,
			})
			m.state = stateRunning
			cmds = append(cmds, m.runAgent())
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.viewport.Width = msg.Width
		m.textarea.SetWidth(msg.Width)
		// Pad the placeholder with Braille Pattern Blank (U+2800) so the
		// CursorLine highlight fills the full width. The textarea's
		// placeholder renderer uses strings.TrimSpace which strips normal
		// spaces, but U+2800 is not a Unicode space so it survives.
		base := "Type a message..."
		if w := m.textarea.Width(); w > len(base) {
			m.textarea.Placeholder = base + strings.Repeat("\u2800", w-len(base))
		}
		m.resizeTextarea()
		m.refreshViewport()

	case spinner.TickMsg:
		if m.state == stateRunning || m.state == stateToolCall {
			var cmd tea.Cmd
			m.spinner, cmd = m.spinner.Update(msg)
			cmds = append(cmds, cmd)
		}

	case agentEventMsg:
		cmds = append(cmds, m.handleAgentEvent(msg.event))

	case agentDoneMsg:
		m.state = stateIdle
		m.appendOutput("\n\n")
		if m.store != nil {
			m.store.Save(context.Background(), m.session)
		}
	}

	var cmd tea.Cmd
	if !skipTextarea {
		m.textarea, cmd = m.textarea.Update(msg)
		cmds = append(cmds, cmd)
	}
	m.resizeTextarea()

	m.viewport, cmd = m.viewport.Update(msg)
	cmds = append(cmds, cmd)

	return m, tea.Batch(cmds...)
}

// View implements tea.Model.
func (m *Model) View() string {
	var status string
	switch m.state {
	case stateRunning:
		status = m.spinner.View() + " thinking..."
	case stateToolCall:
		status = m.spinner.View() + " running tool..."
	case stateError:
		status = m.errorStyle.Render("● error: " + m.err.Error())
	default:
		status = lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Render("● ready")
	}

	return m.viewport.View() + "\n" +
		status + "\n" +
		m.textarea.View()
}

// resizeTextarea adjusts the textarea height to fit its content and
// recalculates the viewport height to fill the remaining space.
func (m *Model) resizeTextarea() {
	w := m.textarea.Width()
	if w <= 0 {
		return
	}

	val := m.textarea.Value()
	visual := 0
	for _, line := range strings.Split(val, "\n") {
		if len(line) == 0 {
			visual++
		} else {
			visual += strings.Count(cellbuf.Wrap(line, w, ""), "\n") + 1
		}
	}

	h := visual
	if h < 1 {
		h = 1
	}
	if h > maxInputHeight {
		h = maxInputHeight
	}

	prev := m.textarea.Height()
	m.textarea.SetHeight(h)
	if h != prev {
		// The textarea's Update ran repositionView with the old height,
		// leaving a stale viewport scroll offset. Reset by re-setting the
		// value (which calls viewport.GotoTop internally), then trigger
		// repositionView via a no-op Update with the new height.
		m.textarea.SetValue(m.textarea.Value())
		m.textarea, _ = m.textarea.Update(nil)
	}
	if m.height > 0 {
		m.viewport.Height = m.height - h - chromeHeight
	}
}

func (m *Model) appendOutput(s string) {
	m.output.WriteString(s)
	m.refreshViewport()
	m.viewport.GotoBottom()
}

func (m *Model) refreshViewport() {
	content := m.output.String()
	if m.width > 0 {
		content = cellbuf.Wrap(content, m.width, "")
	}
	m.viewport.SetContent(content)
}

func (m *Model) runAgent() tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithCancel(context.Background())
		m.cancel = cancel

		events, err := m.agent.Run(ctx, m.session.Messages)
		if err != nil {
			m.err = err
			m.state = stateError
			return agentDoneMsg{}
		}

		// Read all events and forward them.
		// We can't easily send multiple msgs from one Cmd, so we process inline.
		for ev := range events {
			switch ev.Type {
			case agent.EventToken:
				m.appendOutput(ev.Content)
			case agent.EventToolCall:
				m.appendOutput(m.toolStyle.Render("⚡ "+ev.ToolCall.Name) + "\n")
			case agent.EventToolResult:
				m.appendOutput(m.toolStyle.Render("  → "+truncate(ev.ToolResult, 200)) + "\n")
			case agent.EventComplete:
				m.session.Messages = append([]conversation.Message(nil), ev.Messages...)
			case agent.EventError:
				if len(ev.Messages) > 0 {
					m.session.Messages = append([]conversation.Message(nil), ev.Messages...)
				}
				m.err = ev.Err
				m.appendOutput(m.errorStyle.Render("Error: "+ev.Err.Error()) + "\n")
			}
		}

		return agentDoneMsg{}
	}
}

func (m *Model) handleAgentEvent(ev agent.Event) tea.Cmd {
	switch ev.Type {
	case agent.EventToken:
		m.state = stateRunning
		m.appendOutput(ev.Content)
	case agent.EventToolCall:
		m.state = stateToolCall
		m.appendOutput(m.toolStyle.Render("⚡ "+ev.ToolCall.Name) + "\n")
	case agent.EventToolResult:
		m.state = stateRunning
		m.appendOutput(m.toolStyle.Render("  → "+truncate(ev.ToolResult, 200)) + "\n")
	case agent.EventError:
		m.err = ev.Err
		m.state = stateError
		m.appendOutput(m.errorStyle.Render("Error: "+ev.Err.Error()) + "\n")
	}
	return nil
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}
