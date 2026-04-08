package tui

import (
	"context"
	"fmt"

	"github.com/charmbracelet/lipgloss"

	"github.com/stack-bound/stackllm/agent"
	"github.com/stack-bound/stackllm/conversation"
)

// AuthHooks returns agent.Hooks configured to render auth prompts and tool call
// status in the TUI. Pass these to agent.New via WithHooks when using the TUI adapter.
func AuthHooks() agent.Hooks {
	return agent.Hooks{
		OnToken: func(_ context.Context, delta string) {
			fmt.Print(delta)
		},
		OnToolCall: func(_ context.Context, call conversation.ToolCall) {
			style := lipgloss.NewStyle().Foreground(lipgloss.Color("11"))
			fmt.Println(style.Render("⚡ " + call.Name))
		},
		OnToolResult: func(_ context.Context, call conversation.ToolCall, result string, err error) {
			style := lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Italic(true)
			if err != nil {
				fmt.Println(style.Render("  ✗ " + err.Error()))
			} else {
				fmt.Println(style.Render("  → " + truncate(result, 200)))
			}
		},
		AfterComplete: func(_ context.Context, _ []conversation.Message) {
			fmt.Println()
		},
	}
}

// DeviceCodePrompt renders a styled device code auth prompt for the terminal.
func DeviceCodePrompt(userCode, verifyURL string) string {
	boxStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("12")).
		Padding(1, 2)

	codeStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("11")).
		Bold(true)

	urlStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("12")).
		Underline(true)

	content := fmt.Sprintf(
		"Enter code: %s\n\nOpen: %s",
		codeStyle.Render(userCode),
		urlStyle.Render(verifyURL),
	)

	return boxStyle.Render(content)
}

// WebFlowPrompt renders a styled prompt for web-based auth flows.
func WebFlowPrompt(authURL string) string {
	boxStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("12")).
		Padding(1, 2)

	urlStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("12")).
		Underline(true)

	content := fmt.Sprintf(
		"Open this URL to authenticate:\n\n%s",
		urlStyle.Render(authURL),
	)

	return boxStyle.Render(content)
}
