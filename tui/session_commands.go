package tui

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/stack-bound/stackllm/session"
)

// SessionForker is the capability a SessionStore opts into when it can
// branch a session at a chosen message. SQLiteStore implements this via
// Fork; in-memory and other lightweight stores may not.
//
// The interface is defined here (the consumer side) rather than in the
// session package so tui doesn't need to depend on SQLiteStore directly
// — per the "interfaces defined where consumed" convention.
type SessionForker interface {
	Fork(ctx context.Context, srcSessionID, atMessageID, newName string) (*session.Session, error)
}

// SessionExporter is the capability a SessionStore opts into when it
// can serialise a session to JSONL. Matches SQLiteStore.ExportJSONL.
type SessionExporter interface {
	ExportJSONL(ctx context.Context, sessionID string, w io.Writer) error
}

// executeHelp renders the command help block inline in the viewport.
// The content is generated from the commands slice so the list never
// drifts from what the menu offers.
func (m *Model) executeHelp() {
	var b strings.Builder
	b.WriteString("Commands:\n")
	nameWidth := 0
	for _, c := range commands {
		if n := len(c.Name); n > nameWidth {
			nameWidth = n
		}
	}
	for _, c := range commands {
		b.WriteString(fmt.Sprintf("  %-*s  %s\n", nameWidth, c.Name, c.Description))
	}
	m.appendOutput(m.toolStyle.Render(b.String()) + "\n")
}

// executeDelete deletes the current session from the store and starts
// a fresh one in its place. If the store delete fails the error is
// surfaced inline and the current session is left untouched so the
// user can retry.
func (m *Model) executeDelete() tea.Cmd {
	if m.store == nil {
		m.appendOutput(m.errorStyle.Render("✗ /delete: no session store configured") + "\n\n")
		return nil
	}
	id := m.session.ID
	name := displaySessionName(m.session)
	if err := m.store.Delete(context.Background(), id); err != nil {
		m.appendOutput(m.errorStyle.Render("✗ /delete: "+err.Error()) + "\n\n")
		return nil
	}
	m.executeNewSession()
	m.appendOutput(m.toolStyle.Render(fmt.Sprintf("✓ deleted %q", name)) + "\n\n")
	return nil
}

// submitRename applies a new name to the current session and persists
// it. Called from the modal submit path. The name is only committed to
// the live session after the store Save succeeds — otherwise a Save
// failure would leave the status bar showing a name that never made
// it to disk.
func (m *Model) submitRename(name string) {
	name = strings.TrimSpace(name)
	if name == "" {
		m.appendOutput(m.errorStyle.Render("✗ /rename: name cannot be empty") + "\n\n")
		return
	}
	if m.store != nil {
		prev := m.session.Name
		m.session.Name = name
		if err := m.store.Save(context.Background(), m.session); err != nil {
			m.session.Name = prev
			m.appendOutput(m.errorStyle.Render("✗ /rename: "+err.Error()) + "\n\n")
			return
		}
	} else {
		m.session.Name = name
	}
	m.appendOutput(m.toolStyle.Render(fmt.Sprintf("✓ renamed to %q", name)) + "\n\n")
}

// submitExport writes the current session out to a JSONL file at the
// given path. The parent directory is created if missing. Leading `~/`
// is expanded to the user's home directory so the default that the
// modal prefills with is directly usable.
func (m *Model) submitExport(path string) {
	path = strings.TrimSpace(path)
	if path == "" {
		m.appendOutput(m.errorStyle.Render("✗ /export: path cannot be empty") + "\n\n")
		return
	}
	if m.exporter == nil {
		m.appendOutput(m.errorStyle.Render("✗ /export: store does not support export (need SQLite)") + "\n\n")
		return
	}
	resolved, err := expandHome(path)
	if err != nil {
		m.appendOutput(m.errorStyle.Render("✗ /export: "+err.Error()) + "\n\n")
		return
	}
	// Flush the latest in-memory state to the store first so the
	// export reflects messages the user may have typed since the
	// last agent turn completed (and so a freshly /new'd session
	// that has never been saved still has a row to export).
	if m.store != nil {
		if err := m.store.Save(context.Background(), m.session); err != nil {
			m.appendOutput(m.errorStyle.Render("✗ /export: "+err.Error()) + "\n\n")
			return
		}
	}
	if dir := filepath.Dir(resolved); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			m.appendOutput(m.errorStyle.Render("✗ /export: "+err.Error()) + "\n\n")
			return
		}
	}
	f, err := os.Create(resolved)
	if err != nil {
		m.appendOutput(m.errorStyle.Render("✗ /export: "+err.Error()) + "\n\n")
		return
	}
	defer f.Close()
	if err := m.exporter.ExportJSONL(context.Background(), m.session.ID, f); err != nil {
		m.appendOutput(m.errorStyle.Render("✗ /export: "+err.Error()) + "\n\n")
		return
	}
	m.appendOutput(m.toolStyle.Render(fmt.Sprintf("✓ exported %d messages to %s", len(m.session.Messages), resolved)) + "\n\n")
}

// expandHome resolves a leading `~/` to the caller's home directory.
// Anything else is returned unchanged.
func expandHome(path string) (string, error) {
	if !strings.HasPrefix(path, "~/") && path != "~" {
		return path, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	if path == "~" {
		return home, nil
	}
	return filepath.Join(home, path[2:]), nil
}

// displaySessionName returns a short label for a session: its Name if
// set, otherwise the first 8 characters of the ID.
func displaySessionName(s *session.Session) string {
	if s == nil {
		return ""
	}
	if s.Name != "" {
		return s.Name
	}
	if len(s.ID) >= 8 {
		return s.ID[:8]
	}
	return s.ID
}
