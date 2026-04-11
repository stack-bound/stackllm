package tui

import (
	"context"
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/cellbuf"

	"github.com/stack-bound/stackllm/agent"
	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/profile"
	"github.com/stack-bound/stackllm/provider"
	"github.com/stack-bound/stackllm/session"
)

type modelState int

const (
	stateIdle modelState = iota
	stateRunning
	stateToolCall
	stateError
	stateCommandMenu
	stateModelLoading
	stateModelPicker
	stateSessionLoading
	stateSessionPicker
	stateForkPicker
	stateTextModal
	stateConfirmModal
)

// ModelLister is the subset of profile.Manager that the TUI needs to
// support the /models slash command. Implementing this interface lets
// callers provide their own model source (e.g. for tests).
//
// RecentModels and TrackRecentModel back the "recently used" section
// at the top of the picker. RecentModels returns the persisted MRU
// list (most recent first); TrackRecentModel is called after a
// successful switch so the next session can offer the same model
// again without scrolling.
type ModelLister interface {
	ListAllModels(ctx context.Context) ([]profile.ModelInfo, error)
	LoadProviderForModel(ctx context.Context, info profile.ModelInfo) (*provider.OpenAIProvider, error)
	RecentModels(ctx context.Context) ([]profile.ModelInfo, error)
	TrackRecentModel(ctx context.Context, info profile.ModelInfo) error
}

// Option configures a Model.
type Option func(*Model)

// WithModelLister injects a ModelLister so the /models command can list
// available models and switch the agent to a different provider/model
// at runtime.
func WithModelLister(l ModelLister) Option {
	return func(m *Model) { m.modelLister = l }
}

// modelsLoadedMsg is delivered after an async model list has finished.
// recentCount is the number of leading entries in models that came
// from the recent-used list (already deduped against the full list).
type modelsLoadedMsg struct {
	models      []profile.ModelInfo
	recentCount int
	err         error
}

// modelSwitchedMsg is delivered after an async provider switch.
type modelSwitchedMsg struct {
	provider provider.Provider
	info     profile.ModelInfo
	err      error
}

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

	// Slash command popup state.
	cmdFiltered []Command
	cmdCursor   int

	// Model picker state. modelRecentCount is the number of leading
	// entries in models that came from the recent-used list, used to
	// render a divider between the "recent" section and the rest.
	models           []profile.ModelInfo
	modelCursor      int
	modelRecentCount int
	modelLister      ModelLister

	// Session picker state.
	sessions             []*session.Session
	sessionCursor        int
	sessionVisibleOffset int

	// Fork picker state — the message list comes straight from
	// m.session.Messages, so we only need the cursor index.
	forkCursor int

	// Text modal state shared by /rename and /export.
	modalKind   modalKind
	modalInput  textinput.Model
	modalTitle  string
	modalPrompt string

	// Confirm modal state — a y/n prompt with a deferred action that
	// runs when the user confirms. Used by /delete so a stray Enter
	// can't silently throw away the current session.
	// confirmReturnState is the state the TUI restores on close so a
	// confirm opened from the session picker lands back in the picker
	// instead of dropping the user out to idle.
	confirmTitle       string
	confirmPrompt      string
	confirmAction      func() tea.Cmd
	confirmReturnState modelState

	// Cached store capabilities, assigned once in New() so command
	// handlers don't re-type-assert on every keypress.
	forker   SessionForker
	exporter SessionExporter

	// Image paste state. pendingImages is keyed by the monotonic
	// nextImageIdx, which is never rewound within a single Model's
	// lifetime so that a stale `[Image #1]` placeholder in the
	// scrollback can't collide with a freshly-pasted `[Image #1]`.
	// The map is cleared on send (and on /new, which also resets the
	// counter since a brand-new session starts numbering at 1).
	pendingImages   map[int]pendingImage
	nextImageIdx    int
	clipboardReader ClipboardReader

	// Model + context-window display state. currentModel mirrors
	// agent.Model() at construction and after /models switches;
	// contextWindow is the resolved max-prompt limit for that model
	// (upstream metadata first, provider.ContextWindow fallback
	// otherwise). Zero means unknown.
	currentModel  string
	contextWindow int

	// Styles
	userStyle       lipgloss.Style
	assistantStyle  lipgloss.Style
	toolStyle       lipgloss.Style
	errorStyle      lipgloss.Style
	menuStyle       lipgloss.Style
	menuCursorStyle lipgloss.Style
}

const (
	// chromeHeight is the number of rows used by the status line and newlines
	// between the viewport, status, and textarea in View().
	chromeHeight = 2

	// maxInputHeight is the maximum number of rows the textarea can grow to.
	maxInputHeight = 8

	// maxModelPickerVisible bounds the rendered model list height.
	maxModelPickerVisible = 10
)

// New creates a new TUI Model.
func New(a *agent.Agent, store session.SessionStore, opts ...Option) *Model {
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

	ti := textinput.New()
	ti.CharLimit = 200

	m := &Model{
		agent:           a,
		session:         session.New(),
		store:           store,
		textarea:        ta,
		viewport:        vp,
		spinner:         sp,
		state:           stateIdle,
		modalInput:      ti,
		pendingImages:   map[int]pendingImage{},
		clipboardReader: defaultClipboardReader,
		currentModel:    a.Model(),
		contextWindow:   provider.ContextWindow(a.Model()),
		userStyle:       lipgloss.NewStyle().Foreground(lipgloss.Color("12")).Bold(true),
		assistantStyle:  lipgloss.NewStyle().Foreground(lipgloss.Color("10")),
		toolStyle:       lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Italic(true),
		errorStyle:      lipgloss.NewStyle().Foreground(lipgloss.Color("9")).Bold(true),
		menuStyle:       lipgloss.NewStyle().Foreground(lipgloss.Color("7")),
		menuCursorStyle: lipgloss.NewStyle().Foreground(lipgloss.Color("12")).Bold(true),
	}
	// Cache capability interfaces once — handlers rely on these being
	// set (or nil for unsupported stores) rather than re-asserting on
	// every keystroke.
	if f, ok := store.(SessionForker); ok {
		m.forker = f
	}
	if e, ok := store.(SessionExporter); ok {
		m.exporter = e
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
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
			if m.state == stateIdle || m.state == stateCommandMenu {
				m.textarea.InsertString("\n")
				skipTextarea = true
			}
		case tea.KeyEsc:
			switch m.state {
			case stateCommandMenu:
				m.textarea.Reset()
				m.cmdFiltered = nil
				m.cmdCursor = 0
				m.state = stateIdle
				skipTextarea = true
			case stateModelPicker:
				m.state = stateIdle
				skipTextarea = true
			case stateModelLoading:
				// allow cancel of loading by returning to idle
				m.state = stateIdle
				skipTextarea = true
			case stateSessionPicker, stateSessionLoading:
				m.sessions = nil
				m.sessionCursor = 0
				m.sessionVisibleOffset = 0
				m.state = stateIdle
				skipTextarea = true
			case stateForkPicker:
				m.forkCursor = 0
				m.state = stateIdle
				skipTextarea = true
			case stateTextModal:
				m.closeModal()
				skipTextarea = true
			case stateConfirmModal:
				m.closeConfirmModal()
				skipTextarea = true
			}
		case tea.KeyUp:
			if m.state == stateCommandMenu {
				if m.cmdCursor > 0 {
					m.cmdCursor--
				}
				skipTextarea = true
			}
			if m.state == stateModelPicker {
				if m.modelCursor > 0 {
					m.modelCursor--
				}
				skipTextarea = true
			}
			if m.state == stateSessionPicker {
				if m.sessionCursor > 0 {
					m.sessionCursor--
				}
				skipTextarea = true
			}
			if m.state == stateForkPicker {
				if m.forkCursor > 0 {
					m.forkCursor--
				}
				skipTextarea = true
			}
		case tea.KeyDown:
			if m.state == stateCommandMenu {
				if m.cmdCursor < len(m.cmdFiltered)-1 {
					m.cmdCursor++
				}
				skipTextarea = true
			}
			if m.state == stateModelPicker {
				if m.modelCursor < len(m.models)-1 {
					m.modelCursor++
				}
				skipTextarea = true
			}
			if m.state == stateSessionPicker {
				if m.sessionCursor < len(m.sessions)-1 {
					m.sessionCursor++
				}
				skipTextarea = true
			}
			if m.state == stateForkPicker {
				if m.forkCursor < len(m.session.Messages)-1 {
					m.forkCursor++
				}
				skipTextarea = true
			}
		case tea.KeyCtrlV:
			// Intercept Ctrl+V so we can probe the system clipboard
			// for image bytes. Terminals only deliver UTF-8 text via
			// bracketed paste, so we have to read the clipboard
			// ourselves via platform shell-outs. The read is async
			// (clipboard tools can take hundreds of milliseconds) and
			// a clipboardImageMsg is delivered back to Update with the
			// result or errNoImage.
			if m.state == stateIdle || m.state == stateCommandMenu {
				skipTextarea = true
				cmds = append(cmds, m.readClipboardImageCmd())
			}
		case tea.KeyRunes:
			if m.state == stateSessionPicker && len(msg.Runes) == 1 && msg.Runes[0] == 'd' {
				if len(m.sessions) > 0 {
					target := m.sessions[m.sessionCursor]
					name := displaySessionName(target)
					targetID := target.ID
					cmds = append(cmds, m.openConfirmModal(
						"Delete session",
						fmt.Sprintf("Delete %q? This cannot be undone.", name),
						stateSessionPicker,
						func() tea.Cmd { return m.deleteSession(targetID) },
					))
				}
				skipTextarea = true
			}
			if m.state == stateConfirmModal && len(msg.Runes) == 1 {
				switch msg.Runes[0] {
				case 'y', 'Y':
					cmds = append(cmds, m.confirmYes())
					skipTextarea = true
				case 'n', 'N':
					m.closeConfirmModal()
					skipTextarea = true
				}
			}
		case tea.KeyEnter:
			if msg.Alt {
				break // pass Alt+Enter to textarea for newline insertion
			}
			skipTextarea = true // consume plain Enter, don't let textarea add a newline
			switch m.state {
			case stateRunning, stateToolCall, stateModelLoading, stateSessionLoading:
				// ignore Enter while busy
			case stateCommandMenu:
				if len(m.cmdFiltered) > 0 {
					selected := m.cmdFiltered[m.cmdCursor]
					m.textarea.Reset()
					m.cmdFiltered = nil
					m.cmdCursor = 0
					cmds = append(cmds, m.executeCommand(selected))
				}
			case stateModelPicker:
				if len(m.models) > 0 {
					selected := m.models[m.modelCursor]
					m.state = stateModelLoading
					cmds = append(cmds, m.switchModel(selected))
				}
			case stateSessionPicker:
				if len(m.sessions) > 0 {
					selected := m.sessions[m.sessionCursor]
					// If the user hit Enter on the already-loaded
					// session, there's no need to do any work —
					// just close the picker.
					if selected.ID == m.session.ID {
						m.sessions = nil
						m.sessionCursor = 0
						m.state = stateIdle
						break
					}
					m.state = stateSessionLoading
					cmds = append(cmds, m.loadSession(selected.ID))
				}
			case stateForkPicker:
				if len(m.session.Messages) > 0 {
					msgs := m.session.Messages
					idx := m.forkCursor
					if idx < 0 {
						idx = 0
					}
					if idx >= len(msgs) {
						idx = len(msgs) - 1
					}
					target := msgs[idx]
					cmds = append(cmds, m.forkAt(target.ID, idx+1))
					m.state = stateSessionLoading
				}
			case stateTextModal:
				cmds = append(cmds, m.submitModal())
			case stateConfirmModal:
				cmds = append(cmds, m.confirmYes())
			default:
				input := strings.TrimSpace(m.textarea.Value())
				if input == "" {
					break
				}
				m.textarea.Reset()
				blocks := parseInputBlocks(input, m.pendingImages)
				m.appendOutput(m.userStyle.Render("You: ") + renderUserInputPreview(blocks) + "\n\n")
				m.session.AppendMessage(conversation.Message{
					Role:   conversation.RoleUser,
					Blocks: blocks,
				})
				// Clear pending images on send. nextImageIdx is NOT
				// reset — the counter is monotonic for the Model's
				// lifetime so stale placeholders in the scrollback
				// never collide with a fresh paste.
				m.pendingImages = map[int]pendingImage{}
				m.state = stateRunning
				cmds = append(cmds, m.runAgent())
			}
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
		if m.state == stateRunning || m.state == stateToolCall || m.state == stateModelLoading || m.state == stateSessionLoading {
			var cmd tea.Cmd
			m.spinner, cmd = m.spinner.Update(msg)
			cmds = append(cmds, cmd)
		}

	case clipboardImageMsg:
		// If the clipboard probe failed or had no image, fall back to
		// the default text-paste path so Ctrl+V keeps working
		// identically to before when the clipboard holds text. Any
		// other error is also treated as fallback — an unusable
		// clipboard probe should never block the user from pasting
		// text.
		if msg.err != nil || len(msg.data) == 0 {
			cmds = append(cmds, textarea.Paste)
			break
		}
		m.nextImageIdx++
		idx := m.nextImageIdx
		m.pendingImages[idx] = pendingImage{mime: msg.mime, data: msg.data}
		m.textarea.InsertString(fmt.Sprintf("[Image #%d]", idx))
		m.resizeTextarea()

	case agentEventMsg:
		cmds = append(cmds, m.handleAgentEvent(msg.event))

	case agentDoneMsg:
		m.state = stateIdle
		m.appendOutput("\n\n")
		if m.store != nil {
			m.store.Save(context.Background(), m.session)
		}

	case modelsLoadedMsg:
		if msg.err != nil {
			m.err = msg.err
			m.state = stateError
			m.appendOutput(m.errorStyle.Render("Error loading models: "+msg.err.Error()) + "\n\n")
			break
		}
		m.models = msg.models
		m.modelRecentCount = msg.recentCount
		m.modelCursor = 0
		if len(m.models) == 0 {
			m.appendOutput(m.toolStyle.Render("No models available — authenticate a provider first.") + "\n\n")
			m.state = stateIdle
			break
		}
		m.state = stateModelPicker

	case modelSwitchedMsg:
		if msg.err != nil {
			m.err = msg.err
			m.state = stateError
			m.appendOutput(m.errorStyle.Render("Error switching model: "+msg.err.Error()) + "\n\n")
			break
		}
		m.agent.SetProvider(msg.provider)
		m.agent.SetModel(msg.info.Model)
		// Refresh the cached model/context info so the status line
		// reflects the new selection on the next View(). The ModelInfo
		// from the picker carries ContextWindow either from upstream
		// metadata or the hardcoded table; if it is still zero (e.g.
		// because the picker was fed an older config), ask the
		// fallback table one last time.
		m.currentModel = msg.info.Model
		if cw := msg.info.ContextWindow; cw > 0 {
			m.contextWindow = cw
		} else {
			m.contextWindow = provider.ContextWindow(msg.info.Model)
		}
		// Persist the selection to the recent-models list so it
		// surfaces at the top of the picker next time. Best effort:
		// an error here should not block the switch.
		if m.modelLister != nil {
			if err := m.modelLister.TrackRecentModel(context.Background(), msg.info); err != nil {
				m.appendOutput(m.errorStyle.Render("Warning: failed to record recent model: "+err.Error()) + "\n")
			}
		}
		m.appendOutput(m.toolStyle.Render("Switched to "+msg.info.String()) + "\n\n")
		m.state = stateIdle

	case sessionsLoadedMsg:
		if msg.err != nil {
			m.err = msg.err
			m.appendOutput(m.errorStyle.Render("✗ /sessions: "+msg.err.Error()) + "\n\n")
			m.state = stateIdle
			break
		}
		m.sessions = msg.sessions
		// Default the cursor to the currently-loaded session so
		// Enter on the freshly-opened picker is a no-op rather than
		// a jarring switch to a random entry.
		m.sessionCursor = 0
		for i, s := range m.sessions {
			if s.ID == m.session.ID {
				m.sessionCursor = i
				break
			}
		}
		m.sessionVisibleOffset = 0
		if len(m.sessions) == 0 {
			m.appendOutput(m.toolStyle.Render("No saved sessions yet.") + "\n\n")
			m.state = stateIdle
			break
		}
		m.state = stateSessionPicker

	case sessionLoadedMsg:
		if msg.err != nil {
			m.err = msg.err
			m.appendOutput(m.errorStyle.Render("✗ /sessions: "+msg.err.Error()) + "\n\n")
			m.state = stateIdle
			break
		}
		m.session = msg.session
		m.output.Reset()
		m.output.WriteString(RenderConversation(msg.session.Messages))
		m.refreshViewport()
		m.viewport.GotoBottom()
		m.sessions = nil
		m.sessionCursor = 0
		m.state = stateIdle
		m.appendOutput(m.toolStyle.Render(fmt.Sprintf("✓ loaded session %q (%d messages)", displaySessionName(msg.session), len(msg.session.Messages))) + "\n\n")

	case sessionDeletedMsg:
		if msg.err != nil {
			m.appendOutput(m.errorStyle.Render("✗ /sessions: "+msg.err.Error()) + "\n\n")
			break
		}
		// Drop the deleted row from the open picker and fix the
		// cursor so it doesn't fall off the end.
		filtered := m.sessions[:0]
		for _, s := range m.sessions {
			if s.ID != msg.deletedID {
				filtered = append(filtered, s)
			}
		}
		m.sessions = filtered
		if m.sessionCursor >= len(m.sessions) {
			m.sessionCursor = len(m.sessions) - 1
		}
		if m.sessionCursor < 0 {
			m.sessionCursor = 0
		}
		if msg.deletedSelf {
			// Reset the loaded session so the status bar and
			// viewport stop referencing the now-gone session.
			m.executeNewSession()
		}
		if len(m.sessions) == 0 {
			m.state = stateIdle
			m.appendOutput(m.toolStyle.Render("✓ deleted — no saved sessions remain") + "\n\n")
		} else {
			m.appendOutput(m.toolStyle.Render("✓ session deleted") + "\n")
		}

	case sessionForkedMsg:
		if msg.err != nil {
			m.appendOutput(m.errorStyle.Render("✗ /fork: "+msg.err.Error()) + "\n\n")
			m.state = stateIdle
			break
		}
		// Reload the forked session through Load so we pull the
		// canonical message chain (with fresh IDs and hydrated
		// block rows) from the store rather than trusting whatever
		// the in-memory return shape of Fork was.
		m.session = msg.session
		// Fork() returns the session with its message chain
		// already populated from the store walk, so we can render
		// directly without a second round-trip.
		m.output.Reset()
		m.output.WriteString(RenderConversation(msg.session.Messages))
		m.refreshViewport()
		m.viewport.GotoBottom()
		m.state = stateIdle
		m.appendOutput(m.toolStyle.Render(fmt.Sprintf("✓ forked from message [%d] · %d messages copied", msg.atIndex, len(msg.session.Messages))) + "\n\n")
	}

	var cmd tea.Cmd

	// While the text modal is open, keystrokes go to the modal input
	// instead of the textarea. Enter/Esc are consumed above; anything
	// else (characters, cursor movement, backspace) flows into the
	// textinput. This keeps the modal visually and logically isolated
	// from the rest of the UI.
	if m.state == stateTextModal {
		if !skipTextarea {
			m.modalInput, cmd = m.modalInput.Update(msg)
			cmds = append(cmds, cmd)
		}
		m.viewport, cmd = m.viewport.Update(msg)
		cmds = append(cmds, cmd)
		return m, tea.Batch(cmds...)
	}

	// While the confirm modal is open, keystrokes are fully consumed
	// by the y/n/Esc/Enter handlers above. Short-circuit the textarea
	// path so a stray character can't leak into the input buffer.
	if m.state == stateConfirmModal {
		m.viewport, cmd = m.viewport.Update(msg)
		cmds = append(cmds, cmd)
		return m, tea.Batch(cmds...)
	}

	if !skipTextarea &&
		m.state != stateModelPicker &&
		m.state != stateModelLoading &&
		m.state != stateSessionPicker &&
		m.state != stateSessionLoading &&
		m.state != stateForkPicker {
		m.textarea, cmd = m.textarea.Update(msg)
		cmds = append(cmds, cmd)

		// After the textarea consumes the keypress, decide whether to
		// enter or leave the slash command menu based on the new value.
		if _, ok := msg.(tea.KeyMsg); ok {
			val := m.textarea.Value()
			switch {
			case strings.HasPrefix(val, "/") && (m.state == stateIdle || m.state == stateCommandMenu):
				m.cmdFiltered = filterCommands(val)
				if m.cmdCursor >= len(m.cmdFiltered) {
					m.cmdCursor = 0
				}
				m.state = stateCommandMenu
			case m.state == stateCommandMenu:
				m.cmdFiltered = nil
				m.cmdCursor = 0
				m.state = stateIdle
			}
		}
	}
	m.resizeTextarea()

	m.viewport, cmd = m.viewport.Update(msg)
	cmds = append(cmds, cmd)

	return m, tea.Batch(cmds...)
}

// View implements tea.Model.
func (m *Model) View() string {
	// The text modal is rendered full-screen and bypasses the normal
	// viewport/textarea chrome so the user's focus is 100% on the
	// input.
	if m.state == stateTextModal {
		return m.renderModal()
	}
	if m.state == stateConfirmModal {
		return m.renderConfirmModal()
	}

	var status string
	switch m.state {
	case stateRunning:
		status = m.spinner.View() + " thinking..."
	case stateToolCall:
		status = m.spinner.View() + " running tool..."
	case stateModelLoading:
		status = m.spinner.View() + " loading models..."
	case stateSessionLoading:
		status = m.spinner.View() + " loading sessions..."
	case stateCommandMenu:
		status = lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Render("● command")
	case stateModelPicker:
		status = lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Render("● select a model")
	case stateSessionPicker:
		status = lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Render("● select a session")
	case stateForkPicker:
		status = lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Render("● select fork point")
	case stateError:
		status = m.errorStyle.Render("● error: " + m.err.Error())
	default:
		status = lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Render("● ready")
	}

	suffix := formatModelStatus(m.currentModel, m.session.LastUsage, m.contextWindow)
	if suffix != "" {
		suffix = statusSuffixStyle.Render(suffix)
	}
	statusLine := padBetween(status, suffix, m.width)

	out := m.viewport.View() + "\n" + statusLine + "\n"
	if menu := m.renderMenu(); menu != "" {
		out += menu + "\n"
	}
	out += m.textarea.View()
	return out
}

// renderMenu returns the popup menu lines for the current state, or
// an empty string if no menu should be shown.
func (m *Model) renderMenu() string {
	switch m.state {
	case stateCommandMenu:
		return m.renderCommandMenu()
	case stateModelPicker:
		return m.renderModelPicker()
	case stateSessionPicker:
		return m.renderSessionPicker()
	case stateForkPicker:
		return m.renderForkPicker()
	}
	return ""
}

func (m *Model) renderCommandMenu() string {
	if len(m.cmdFiltered) == 0 {
		return m.menuStyle.Render("  (no matching commands)")
	}
	var b strings.Builder
	for i, c := range m.cmdFiltered {
		line := fmt.Sprintf("  %s  %s", c.Name, c.Description)
		if i == m.cmdCursor {
			b.WriteString(m.menuCursorStyle.Render("> " + c.Name + "  " + c.Description))
		} else {
			b.WriteString(m.menuStyle.Render(line))
		}
		if i < len(m.cmdFiltered)-1 {
			b.WriteString("\n")
		}
	}
	return b.String()
}

func (m *Model) renderModelPicker() string {
	if len(m.models) == 0 {
		return m.menuStyle.Render("  (no models)")
	}
	// Compute the visible window so the cursor stays in view.
	start := 0
	end := len(m.models)
	if end > maxModelPickerVisible {
		// Centre cursor when possible.
		half := maxModelPickerVisible / 2
		start = m.modelCursor - half
		if start < 0 {
			start = 0
		}
		end = start + maxModelPickerVisible
		if end > len(m.models) {
			end = len(m.models)
			start = end - maxModelPickerVisible
		}
	}
	var b strings.Builder
	for i := start; i < end; i++ {
		// Insert a divider where the recent section ends and the
		// full catalogue begins, but only when both sides are
		// visible in the current window — drawing it as the very
		// first line would just be noise.
		if m.modelRecentCount > 0 && i == m.modelRecentCount && i > start {
			b.WriteString(m.menuStyle.Render("  ── all models ──"))
			b.WriteString("\n")
		}
		info := m.models[i]
		if i == m.modelCursor {
			b.WriteString(m.menuCursorStyle.Render("> " + info.String()))
		} else {
			b.WriteString(m.menuStyle.Render("  " + info.String()))
		}
		if i < end-1 {
			b.WriteString("\n")
		}
	}
	if end-start < len(m.models) {
		b.WriteString("\n")
		b.WriteString(m.menuStyle.Render(fmt.Sprintf("  (%d/%d)", m.modelCursor+1, len(m.models))))
	}
	return b.String()
}

// menuHeight returns the number of vertical lines the menu currently
// consumes, including the trailing newline added by View().
func (m *Model) menuHeight() int {
	menu := m.renderMenu()
	if menu == "" {
		return 0
	}
	return strings.Count(menu, "\n") + 2 // +1 for the line itself, +1 for the trailing newline
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
		m.viewport.Height = m.height - h - chromeHeight - m.menuHeight()
		if m.viewport.Height < 1 {
			m.viewport.Height = 1
		}
	}
}

// executeCommand dispatches the user's selection from the command menu.
// Returns a Cmd if the command needs to fire an async operation.
func (m *Model) executeCommand(c Command) tea.Cmd {
	switch c.ID {
	case CommandModels:
		if m.modelLister == nil {
			m.appendOutput(m.errorStyle.Render("Error: model switching is not configured (pass tui.WithModelLister)") + "\n\n")
			m.state = stateIdle
			return nil
		}
		m.state = stateModelLoading
		return m.loadModels()
	case CommandNew:
		m.executeNewSession()
		m.state = stateIdle
		return nil
	case CommandHelp:
		m.executeHelp()
		m.state = stateIdle
		return nil
	case CommandSessions:
		return m.openSessionPicker()
	case CommandRename:
		return m.openRenameModal()
	case CommandFork:
		return m.openForkPicker()
	case CommandDelete:
		name := displaySessionName(m.session)
		return m.openConfirmModal(
			"Delete session",
			fmt.Sprintf("Delete %q? This cannot be undone.", name),
			stateIdle,
			func() tea.Cmd { return m.executeDelete() },
		)
	case CommandExport:
		return m.openExportModal()
	}
	m.state = stateIdle
	return nil
}

// executeNewSession discards the current session and starts fresh.
// A fresh session resets the image placeholder counter back to 1
// because the old scrollback is cleared at the same time, so there's
// nothing to collide with.
func (m *Model) executeNewSession() {
	m.session = session.New()
	m.output.Reset()
	m.pendingImages = map[int]pendingImage{}
	m.nextImageIdx = 0
	m.refreshViewport()
	m.appendOutput(m.toolStyle.Render("New session started.") + "\n\n")
}

// loadModels asynchronously fetches available models from the
// configured ModelLister and prepends the user's recently used
// models. Recents that are no longer present in the full list (e.g.
// the provider was logged out, or the model was filtered out) are
// dropped silently.
func (m *Model) loadModels() tea.Cmd {
	lister := m.modelLister
	return func() tea.Msg {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		all, err := lister.ListAllModels(ctx)
		if err != nil {
			return modelsLoadedMsg{err: err}
		}
		// Recents are best-effort: an error fetching them is not
		// fatal — we still return the full list.
		recents, _ := lister.RecentModels(ctx)

		present := make(map[string]int, len(all))
		for i, info := range all {
			present[info.String()] = i
		}

		merged := make([]profile.ModelInfo, 0, len(all)+len(recents))
		seen := make(map[string]bool, len(recents))
		for _, r := range recents {
			key := r.String()
			if seen[key] {
				continue
			}
			idx, ok := present[key]
			if !ok {
				continue
			}
			// Use the entry from the live list so the Endpoint
			// reflects current upstream metadata, not what was
			// frozen in config when the user last picked it.
			merged = append(merged, all[idx])
			seen[key] = true
		}
		recentCount := len(merged)
		for _, info := range all {
			if seen[info.String()] {
				continue
			}
			merged = append(merged, info)
		}

		return modelsLoadedMsg{models: merged, recentCount: recentCount}
	}
}

// switchModel asynchronously builds a provider for the selected model
// without mutating the agent — the Update loop applies the result so
// the swap happens on the main goroutine. The selected ModelInfo
// carries the endpoint metadata populated by ListAllModels, so
// responses-only Copilot models (e.g. gpt-5.4-mini) are routed to
// /responses rather than /chat/completions.
func (m *Model) switchModel(info profile.ModelInfo) tea.Cmd {
	lister := m.modelLister
	return func() tea.Msg {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		p, err := lister.LoadProviderForModel(ctx, info)
		if err != nil {
			return modelSwitchedMsg{info: info, err: err}
		}
		return modelSwitchedMsg{provider: p, info: info}
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

		thinkingStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Faint(true)
		currentBlock := conversation.BlockText

		for ev := range events {
			switch ev.Type {
			case agent.EventBlockStart:
				currentBlock = ev.BlockType
				switch ev.BlockType {
				case conversation.BlockThinking:
					m.appendOutput(thinkingStyle.Render("\nthinking: "))
				case conversation.BlockText:
					// plain text streams inline with no marker.
				}
			case agent.EventBlockDelta:
				switch ev.BlockType {
				case conversation.BlockText:
					m.appendOutput(ev.Content)
				case conversation.BlockThinking:
					m.appendOutput(thinkingStyle.Render(ev.Content))
				}
			case agent.EventBlockEnd:
				if currentBlock == conversation.BlockThinking {
					m.appendOutput("\n")
				}
			case agent.EventToolCall:
				m.appendOutput(m.toolStyle.Render("⚡ "+ev.ToolCall.Name) + "\n")
			case agent.EventToolResult:
				m.appendOutput(m.toolStyle.Render("  → "+truncate(ev.ToolResult, 200)) + "\n")
			case agent.EventUsage:
				if ev.Usage != nil {
					usage := *ev.Usage
					m.session.LastUsage = &usage
				}
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
	case agent.EventBlockDelta:
		m.state = stateRunning
		if ev.BlockType == conversation.BlockText {
			m.appendOutput(ev.Content)
		}
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

// renderUserInputPreview produces the echo line drawn into the
// scrollback when the user sends a message. It walks the same blocks
// that were just appended to the session so image blocks render as
// `[image: mime, N bytes]` alongside the surrounding text.
func renderUserInputPreview(blocks []conversation.Block) string {
	var b strings.Builder
	for i, blk := range blocks {
		switch blk.Type {
		case conversation.BlockText:
			if blk.Text == "" {
				continue
			}
			if i > 0 && b.Len() > 0 {
				b.WriteString(" ")
			}
			b.WriteString(blk.Text)
		case conversation.BlockImage:
			if b.Len() > 0 {
				b.WriteString(" ")
			}
			b.WriteString(renderImagePlaceholder(blk))
		}
	}
	return b.String()
}
