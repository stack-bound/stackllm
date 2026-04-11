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
	CommandHelp     = "help"
	CommandSessions = "sessions"
	CommandNew      = "new"
	CommandRename   = "rename"
	CommandFork     = "fork"
	CommandDelete   = "delete"
	CommandExport   = "export"
	CommandModels   = "models"
)

// commands is the canonical list of slash commands surfaced in the popup
// menu. Order is preserved when nothing has been typed after "/".
var commands = []Command{
	{ID: CommandHelp, Name: "/help", Description: "Show available commands"},
	{ID: CommandSessions, Name: "/sessions", Description: "Browse and switch sessions"},
	{ID: CommandNew, Name: "/new", Description: "Start a fresh session"},
	{ID: CommandRename, Name: "/rename", Description: "Rename the current session"},
	{ID: CommandFork, Name: "/fork", Description: "Fork the current session at a chosen message"},
	{ID: CommandDelete, Name: "/delete", Description: "Delete the current session"},
	{ID: CommandExport, Name: "/export", Description: "Save the current session as JSONL"},
	{ID: CommandModels, Name: "/models", Description: "Switch model or provider"},
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
