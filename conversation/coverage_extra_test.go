package conversation

import (
	"strings"
	"testing"
)

func TestBuilder_ToolResultBlock(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		callID  string
		content string
		isErr   bool
	}{
		{name: "success result", callID: "call_1", content: "file contents", isErr: false},
		{name: "error result", callID: "call_2", content: "Error: no such file", isErr: true},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			msgs := NewBuilder().
				ToolResult("call_0", "first").
				ToolResultBlock(tt.callID, tt.content, tt.isErr).
				Build()

			if len(msgs) != 1 {
				t.Fatalf("got %d messages, want 1", len(msgs))
			}
			blocks := msgs[0].Blocks
			if len(blocks) != 2 {
				t.Fatalf("got %d blocks, want 2", len(blocks))
			}
			blk := blocks[1]
			if blk.Type != BlockToolResult {
				t.Errorf("block type = %q, want %q", blk.Type, BlockToolResult)
			}
			if blk.ToolCallID != tt.callID {
				t.Errorf("ToolCallID = %q, want %q", blk.ToolCallID, tt.callID)
			}
			if blk.Text != tt.content {
				t.Errorf("Text = %q, want %q", blk.Text, tt.content)
			}
			if blk.ToolIsError != tt.isErr {
				t.Errorf("ToolIsError = %v, want %v", blk.ToolIsError, tt.isErr)
			}
			if blk.ID == "" {
				t.Error("block ID should be assigned")
			}
		})
	}
}

func TestBuilder_ImageURL(t *testing.T) {
	t.Parallel()

	msgs := NewBuilder().
		User("look at this").
		ImageURL("image/jpeg", "https://example.com/cat.jpg").
		Build()

	if len(msgs) != 1 {
		t.Fatalf("got %d messages, want 1", len(msgs))
	}
	blocks := msgs[0].Blocks
	if len(blocks) != 2 {
		t.Fatalf("got %d blocks, want 2", len(blocks))
	}
	blk := blocks[1]
	if blk.Type != BlockImage {
		t.Errorf("block type = %q, want %q", blk.Type, BlockImage)
	}
	if blk.MimeType != "image/jpeg" {
		t.Errorf("MimeType = %q, want image/jpeg", blk.MimeType)
	}
	if blk.ImageURL != "https://example.com/cat.jpg" {
		t.Errorf("ImageURL = %q", blk.ImageURL)
	}
	if len(blk.ImageData) != 0 {
		t.Errorf("ImageData should be empty for URL images, got %d bytes", len(blk.ImageData))
	}
	if blk.ID == "" {
		t.Error("block ID should be assigned")
	}
}

func TestBuilder_BlockBeforeMessagePanics(t *testing.T) {
	t.Parallel()

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic when appending a block before any message")
		}
		msg, ok := r.(string)
		if !ok || !strings.Contains(msg, "before any message") {
			t.Errorf("unexpected panic value: %v", r)
		}
	}()
	NewBuilder().Text("orphan")
}

// TestTokenBudget_CountsToolUseBlocks verifies the default estimator
// charges for tool_use name+arguments, not just Text. A message whose
// only content is a large tool_use block must be dropped when it blows
// the budget and kept when it fits.
func TestTokenBudget_CountsToolUseBlocks(t *testing.T) {
	t.Parallel()

	// ~100 estimated tokens: 400 chars of args / 4.
	bigArgs := strings.Repeat("x", 400)
	msgs := []Message{
		{
			Role: RoleAssistant,
			Blocks: []Block{{
				Type:         BlockToolUse,
				ToolCallID:   "call_1",
				ToolName:     "read_file",
				ToolArgsJSON: bigArgs,
			}},
		},
	}

	// Budget below the tool_use estimate: the message must be dropped.
	// If the estimator ignored tool_use blocks the estimate would be 0
	// and the message would survive.
	got := TokenBudget(msgs, 50, nil)
	if len(got) != 0 {
		t.Errorf("TokenBudget(50) kept %d messages, want 0 (tool_use args must count)", len(got))
	}

	// Budget above the estimate: the message must survive.
	got = TokenBudget(msgs, 150, nil)
	if len(got) != 1 {
		t.Errorf("TokenBudget(150) kept %d messages, want 1", len(got))
	}
}
