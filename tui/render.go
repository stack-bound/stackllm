package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"

	"github.com/stack-bound/stackllm/conversation"
)

// RenderMessage formats a single message for display.
func RenderMessage(msg conversation.Message) string {
	var b strings.Builder

	switch msg.Role {
	case conversation.RoleSystem:
		style := lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Italic(true)
		b.WriteString(style.Render("System: "+msg.Content) + "\n")

	case conversation.RoleUser:
		style := lipgloss.NewStyle().Foreground(lipgloss.Color("12")).Bold(true)
		b.WriteString(style.Render("You: ") + msg.Content + "\n")

	case conversation.RoleAssistant:
		style := lipgloss.NewStyle().Foreground(lipgloss.Color("10"))
		b.WriteString(style.Render("Assistant: ") + msg.Content + "\n")
		for _, tc := range msg.ToolCalls {
			toolStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("11"))
			b.WriteString(toolStyle.Render(fmt.Sprintf("  ⚡ %s(%s)", tc.Name, truncateArgs(tc.Arguments, 100))) + "\n")
		}

	case conversation.RoleTool:
		style := lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Italic(true)
		b.WriteString(style.Render(fmt.Sprintf("  → [%s] %s", msg.ToolCallID, truncate(msg.Content, 200))) + "\n")
	}

	return b.String()
}

// RenderConversation formats a full conversation for display.
func RenderConversation(msgs []conversation.Message) string {
	var b strings.Builder
	for _, msg := range msgs {
		b.WriteString(RenderMessage(msg))
	}
	return b.String()
}

func truncateArgs(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}
