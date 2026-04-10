package tui

import "strings"

// Command identifies a slash command available from the textarea.
type Command struct {
	ID          string
	Name        string // user-facing name including the leading slash, e.g. "/models"
	Description string
}

// Command identifiers.
const (
	CommandModels = "models"
	CommandNew    = "new"
)

// commands is the canonical list of slash commands surfaced in the popup
// menu. Order is preserved when nothing has been typed after "/".
var commands = []Command{
	{ID: CommandModels, Name: "/models", Description: "Switch model or provider"},
	{ID: CommandNew, Name: "/new", Description: "Start a fresh session"},
}

// filterCommands returns the commands whose Name contains query as a
// case-insensitive substring. The leading slash on query is tolerated so
// callers can pass the textarea value directly. An empty query returns
// every command in registry order.
func filterCommands(query string) []Command {
	q := strings.ToLower(strings.TrimSpace(query))
	if q == "" || q == "/" {
		out := make([]Command, len(commands))
		copy(out, commands)
		return out
	}
	var out []Command
	for _, c := range commands {
		if strings.Contains(strings.ToLower(c.Name), q) {
			out = append(out, c)
		}
	}
	return out
}
