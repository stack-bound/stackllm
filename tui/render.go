package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"

	"github.com/stack-bound/stackllm/conversation"
)

// RenderMessage formats a single message for display by walking its
// blocks in the order they were produced. Thinking blocks are rendered
// dimmed; tool_use blocks show a compact tool_name(args) line;
// tool_result blocks show a preview; images render as placeholders.
func RenderMessage(msg conversation.Message) string {
	var b strings.Builder

	switch msg.Role {
	case conversation.RoleSystem:
		style := lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Italic(true)
		b.WriteString(style.Render("System: "+msg.TextContent()) + "\n")

	case conversation.RoleUser:
		style := lipgloss.NewStyle().Foreground(lipgloss.Color("12")).Bold(true)
		b.WriteString(style.Render("You: "))
		renderBlocksForUser(&b, msg)
		b.WriteString("\n")

	case conversation.RoleAssistant:
		style := lipgloss.NewStyle().Foreground(lipgloss.Color("10"))
		b.WriteString(style.Render("Assistant: "))
		renderBlocksForAssistant(&b, msg)
		b.WriteString("\n")

	case conversation.RoleTool:
		style := lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Italic(true)
		for _, blk := range msg.Blocks {
			if blk.Type != conversation.BlockToolResult {
				continue
			}
			b.WriteString(style.Render(fmt.Sprintf("  → [%s] %s", blk.ToolCallID, truncate(blk.Text, 200))) + "\n")
		}
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

// renderBlocksForUser handles the subset of block types a user turn
// may contain: text and images.
func renderBlocksForUser(b *strings.Builder, msg conversation.Message) {
	for _, blk := range msg.Blocks {
		switch blk.Type {
		case conversation.BlockText:
			b.WriteString(blk.Text)
		case conversation.BlockImage:
			b.WriteString(renderImagePlaceholder(blk))
		}
	}
}

// renderBlocksForAssistant walks the full interleaved sequence an
// assistant turn can carry: thinking, text, tool_use, redacted_thinking.
func renderBlocksForAssistant(b *strings.Builder, msg conversation.Message) {
	thinkingStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Faint(true)
	toolStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("11"))

	first := true
	for _, blk := range msg.Blocks {
		switch blk.Type {
		case conversation.BlockText:
			if !first {
				b.WriteString("\n")
			}
			b.WriteString(blk.Text)
			first = false
		case conversation.BlockThinking:
			if !first {
				b.WriteString("\n")
			}
			b.WriteString(thinkingStyle.Render("  thinking: " + truncate(blk.Text, 200)))
			first = false
		case conversation.BlockRedactedThinking:
			if !first {
				b.WriteString("\n")
			}
			b.WriteString(thinkingStyle.Render(fmt.Sprintf("  [redacted thinking, %d bytes]", len(blk.RedactedData))))
			first = false
		case conversation.BlockToolUse:
			if !first {
				b.WriteString("\n")
			}
			b.WriteString(toolStyle.Render(fmt.Sprintf("  ⚡ %s(%s)", blk.ToolName, truncateArgs(blk.ToolArgsJSON, 100))))
			first = false
		case conversation.BlockImage:
			if !first {
				b.WriteString("\n")
			}
			b.WriteString(renderImagePlaceholder(blk))
			first = false
		}
	}
}

func renderImagePlaceholder(blk conversation.Block) string {
	mime := blk.MimeType
	if mime == "" {
		mime = "image"
	}
	if blk.ImageURL != "" {
		return fmt.Sprintf("[image: %s @ %s]", mime, blk.ImageURL)
	}
	return fmt.Sprintf("[image: %s, %d bytes]", mime, len(blk.ImageData))
}

func truncateArgs(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}
