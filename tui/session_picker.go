package tui

import (
	"context"
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/session"
)

// maxSessionPickerVisible bounds the rendered session picker height so
// long session lists scroll rather than overflow the viewport.
const maxSessionPickerVisible = 10

// sessionsLoadedMsg carries the result of an async store.List call.
type sessionsLoadedMsg struct {
	sessions []*session.Session
	err      error
}

// sessionLoadedMsg carries the result of an async store.Load call.
type sessionLoadedMsg struct {
	session *session.Session
	err     error
}

// sessionDeletedMsg carries the result of an async store.Delete call.
// deletedID is the session that was just removed; deletedSelf is true
// if it happened to be the session currently loaded in the TUI, which
// forces a reset to a fresh session.
type sessionDeletedMsg struct {
	deletedID   string
	deletedSelf bool
	err         error
}

// sessionForkedMsg carries the result of an async forker.Fork call.
// atIndex is the 1-based index of the message the user forked at,
// used to format the feedback line.
type sessionForkedMsg struct {
	session *session.Session
	atIndex int
	err     error
}

// loadSessions kicks off an async List call against the session store.
// The current in-memory session is flushed first so a freshly /new'd
// session still shows up in the picker alongside previously-saved
// sessions. Each listed session is then hydrated via Load so the
// picker can show real message counts; SessionStore.List returns
// metadata only, so without this follow-up every row would read
// "0 msgs". The result is delivered back via sessionsLoadedMsg.
func (m *Model) loadSessions() tea.Cmd {
	store := m.store
	sess := m.session
	return func() tea.Msg {
		if store == nil {
			return sessionsLoadedMsg{err: fmt.Errorf("no session store configured")}
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := store.Save(ctx, sess); err != nil {
			return sessionsLoadedMsg{err: err}
		}
		list, err := store.List(ctx)
		if err != nil {
			return sessionsLoadedMsg{err: err}
		}
		// N+1 hydration — acceptable because picker loads are
		// user-triggered and session counts in typical embedder
		// usage are small. If this becomes a bottleneck, the
		// store can expose a dedicated count method and the TUI
		// can feature-detect it via an interface like
		// SessionForker / SessionExporter.
		for _, s := range list {
			if len(s.Messages) > 0 {
				continue
			}
			full, err := store.Load(ctx, s.ID)
			if err != nil {
				continue
			}
			s.Messages = full.Messages
		}
		return sessionsLoadedMsg{sessions: list}
	}
}

// loadSession kicks off an async Load for the chosen session ID. The
// result — full conversation history included — is delivered via
// sessionLoadedMsg and applied on the main goroutine so we never race
// with an in-flight agent run.
func (m *Model) loadSession(id string) tea.Cmd {
	store := m.store
	return func() tea.Msg {
		if store == nil {
			return sessionLoadedMsg{err: fmt.Errorf("no session store configured")}
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		sess, err := store.Load(ctx, id)
		if err != nil {
			return sessionLoadedMsg{err: err}
		}
		return sessionLoadedMsg{session: sess}
	}
}

// deleteSession kicks off an async Delete for the chosen session ID.
// If the session being deleted is the one currently loaded, the Update
// handler resets to a fresh session.
func (m *Model) deleteSession(id string) tea.Cmd {
	store := m.store
	self := id == m.session.ID
	return func() tea.Msg {
		if store == nil {
			return sessionDeletedMsg{deletedID: id, deletedSelf: self, err: fmt.Errorf("no session store configured")}
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := store.Delete(ctx, id); err != nil {
			return sessionDeletedMsg{deletedID: id, deletedSelf: self, err: err}
		}
		return sessionDeletedMsg{deletedID: id, deletedSelf: self}
	}
}

// forkAt kicks off an async Fork of the current session at the chosen
// message ID. The current in-memory session is flushed to the store
// first so Fork sees the latest message chain — important for a
// freshly /new'd session that has never been saved. The freshly forked
// session is delivered via sessionForkedMsg and swapped in on the main
// goroutine.
func (m *Model) forkAt(atMessageID string, atIndex int) tea.Cmd {
	forker := m.forker
	store := m.store
	sess := m.session
	srcID := m.session.ID
	return func() tea.Msg {
		if forker == nil {
			return sessionForkedMsg{err: fmt.Errorf("store does not support branching (need SQLite)")}
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if store != nil {
			if err := store.Save(ctx, sess); err != nil {
				return sessionForkedMsg{err: err}
			}
		}
		forked, err := forker.Fork(ctx, srcID, atMessageID, "")
		if err != nil {
			return sessionForkedMsg{err: err}
		}
		return sessionForkedMsg{session: forked, atIndex: atIndex}
	}
}

// openSessionPicker transitions the TUI into the session-loading state
// and kicks off the list load. The picker appears once the sessions
// have arrived via sessionsLoadedMsg.
func (m *Model) openSessionPicker() tea.Cmd {
	if m.store == nil {
		m.appendOutput(m.errorStyle.Render("✗ /sessions: no session store configured") + "\n\n")
		return nil
	}
	m.state = stateSessionLoading
	return m.loadSessions()
}

// openForkPicker switches to the fork picker state, pre-pointing the
// cursor at the last message in the current session so "Enter" forks
// from the leaf by default.
func (m *Model) openForkPicker() tea.Cmd {
	if m.forker == nil {
		m.appendOutput(m.errorStyle.Render("✗ /fork: store does not support branching (need SQLite)") + "\n\n")
		return nil
	}
	if len(m.session.Messages) == 0 {
		m.appendOutput(m.errorStyle.Render("✗ /fork: current session has no messages yet") + "\n\n")
		return nil
	}
	m.forkCursor = len(m.session.Messages) - 1
	m.state = stateForkPicker
	return nil
}

// renderSessionPicker draws the session list — left-to-right: marker
// dot, session name, message count, relative updated timestamp — with
// columns aligned via lipgloss width padding.
func (m *Model) renderSessionPicker() string {
	if len(m.sessions) == 0 {
		return m.menuStyle.Render("  (no sessions)")
	}

	// First pass: compute max name width so counts and timestamps
	// align across rows regardless of name length.
	rows := make([]sessionRow, 0, len(m.sessions))
	maxName := 0
	maxCount := 0
	for _, s := range m.sessions {
		name := displaySessionName(s)
		if w := lipgloss.Width(name); w > maxName {
			maxName = w
		}
		count := fmt.Sprintf("%d msgs", len(s.Messages))
		if w := lipgloss.Width(count); w > maxCount {
			maxCount = w
		}
		rows = append(rows, sessionRow{
			id:    s.ID,
			name:  name,
			count: count,
			when:  formatRelative(s.Updated),
		})
	}

	start, end := windowBounds(m.sessionCursor, len(rows), maxSessionPickerVisible)
	m.sessionVisibleOffset = start

	var b strings.Builder
	header := fmt.Sprintf("  Sessions (%d)", len(rows))
	b.WriteString(m.menuStyle.Render(header))
	b.WriteString("\n\n")

	for i := start; i < end; i++ {
		r := rows[i]
		marker := "○"
		if r.id == m.session.ID {
			marker = "●"
		}
		namePadded := padRight(r.name, maxName)
		countPadded := padRight(r.count, maxCount)
		line := fmt.Sprintf("%s %s   %s   %s", marker, namePadded, countPadded, r.when)
		if i == m.sessionCursor {
			b.WriteString(m.menuCursorStyle.Render("> " + line))
		} else {
			b.WriteString(m.menuStyle.Render("  " + line))
		}
		b.WriteString("\n")
	}

	b.WriteString("\n")
	hint := fmt.Sprintf("  ↑↓ move · enter load · d delete · esc cancel            (%d/%d)", m.sessionCursor+1, len(rows))
	b.WriteString(m.menuStyle.Render(hint))
	return b.String()
}

// renderForkPicker draws the linear chain of messages in the current
// session so the user can pick a fork point.
func (m *Model) renderForkPicker() string {
	msgs := m.session.Messages
	if len(msgs) == 0 {
		return m.menuStyle.Render("  (no messages in current session)")
	}

	// Compute index column width from the message count so the role
	// and preview columns line up regardless of session length.
	idxWidth := lipgloss.Width(fmt.Sprintf("[%d]", len(msgs)))
	roleWidth := 0
	previews := make([]string, len(msgs))
	for i, msg := range msgs {
		role := string(msg.Role)
		if w := lipgloss.Width(role); w > roleWidth {
			roleWidth = w
		}
		previews[i] = truncateLine(firstTextPreview(msg), 60)
	}

	start, end := windowBounds(m.forkCursor, len(msgs), maxSessionPickerVisible)

	var b strings.Builder
	header := fmt.Sprintf("  Fork from which message?  (current session: %s)", displaySessionName(m.session))
	b.WriteString(m.menuStyle.Render(header))
	b.WriteString("\n\n")

	for i := start; i < end; i++ {
		msg := msgs[i]
		idxCol := padRight(fmt.Sprintf("[%d]", i+1), idxWidth)
		roleCol := padRight(string(msg.Role), roleWidth)
		line := fmt.Sprintf("%s %s  %s", idxCol, roleCol, previews[i])
		if i == len(msgs)-1 {
			line += "  (leaf)"
		}
		if i == m.forkCursor {
			b.WriteString(m.menuCursorStyle.Render("> " + line))
		} else {
			b.WriteString(m.menuStyle.Render("  " + line))
		}
		b.WriteString("\n")
	}

	b.WriteString("\n")
	hint := fmt.Sprintf("  ↑↓ move · enter fork here · esc cancel                    (%d/%d)", m.forkCursor+1, len(msgs))
	b.WriteString(m.menuStyle.Render(hint))
	return b.String()
}

// sessionRow is a precomputed row for the session picker so the render
// loop can operate on already-formatted strings.
type sessionRow struct {
	id    string
	name  string
	count string
	when  string
}

// windowBounds returns [start, end) for a cursor-following window of
// up to visible rows, keeping the cursor in view.
func windowBounds(cursor, total, visible int) (int, int) {
	if total <= visible {
		return 0, total
	}
	half := visible / 2
	start := cursor - half
	if start < 0 {
		start = 0
	}
	end := start + visible
	if end > total {
		end = total
		start = end - visible
	}
	return start, end
}

// padRight pads s with spaces on the right up to visual width w.
// lipgloss.Width is used to measure so multi-byte runes align as the
// terminal actually renders them.
func padRight(s string, w int) string {
	cur := lipgloss.Width(s)
	if cur >= w {
		return s
	}
	return s + strings.Repeat(" ", w-cur)
}

// firstTextPreview returns the first text-bearing block's content, as
// a best-effort single-line preview. Assistant thinking blocks are
// preferred over nothing when a turn is thinking-only (rare but
// possible).
func firstTextPreview(msg conversation.Message) string {
	if t := msg.TextContent(); strings.TrimSpace(t) != "" {
		return t
	}
	if t := msg.ThinkingText(); strings.TrimSpace(t) != "" {
		return t
	}
	for _, b := range msg.Blocks {
		switch b.Type {
		case conversation.BlockToolUse:
			return fmt.Sprintf("⚡ %s(%s)", b.ToolName, b.ToolArgsJSON)
		case conversation.BlockToolResult:
			return b.Text
		case conversation.BlockImage:
			return fmt.Sprintf("[image: %s]", b.MimeType)
		}
	}
	return ""
}

// truncateLine collapses newlines and trims s to at most max display
// cells, appending an ellipsis if the original was longer. lipgloss
// width is used so wide runes are counted correctly.
func truncateLine(s string, max int) string {
	s = strings.ReplaceAll(s, "\r\n", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\t", " ")
	// Collapse runs of whitespace so a multi-paragraph preview isn't
	// mostly spaces.
	for strings.Contains(s, "  ") {
		s = strings.ReplaceAll(s, "  ", " ")
	}
	if lipgloss.Width(s) <= max {
		return s
	}
	// Walk runes and cut as soon as the visible width would exceed
	// max-1 (reserving one cell for the ellipsis character).
	budget := max - 1
	if budget < 1 {
		budget = 1
	}
	var b strings.Builder
	w := 0
	for _, r := range s {
		rw := lipgloss.Width(string(r))
		if w+rw > budget {
			break
		}
		b.WriteRune(r)
		w += rw
	}
	b.WriteRune('…')
	return b.String()
}

// formatRelative returns a human-friendly "ago" string for t. Falls
// back to a yyyy-mm-dd style date for anything older than ~2 weeks.
// An unset (zero) time renders as "—".
func formatRelative(t time.Time) string {
	if t.IsZero() {
		return "—"
	}
	d := time.Since(t)
	if d < 0 {
		d = 0
	}
	switch {
	case d < 45*time.Second:
		return "just now"
	case d < time.Hour:
		return fmt.Sprintf("%dm ago", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%dh ago", int(d.Hours()))
	case d < 48*time.Hour:
		return "yesterday"
	case d < 14*24*time.Hour:
		return fmt.Sprintf("%dd ago", int(d.Hours()/24))
	default:
		return t.Format("Jan 2")
	}
}
