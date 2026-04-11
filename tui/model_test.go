package tui

import (
	"bytes"
	"context"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/stack-bound/stackllm/agent"
	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/provider"
	"github.com/stack-bound/stackllm/session"
)

// cannedPNG is an 8-byte PNG magic header followed by a tag byte. It's
// not a valid PNG image, but it's enough for sniffImageMIME to detect
// as image/png and for round-trip tests to verify the bytes survive.
var cannedPNG = []byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A, 0x42}

func newPasteTestModel(t *testing.T, reader ClipboardReader) *Model {
	t.Helper()
	p := provider.New(provider.OllamaConfig("http://localhost", "test"))
	a := agent.New(p)
	return New(a, session.NewInMemoryStore(), WithClipboardReader(reader))
}

// runCmd executes a tea.Cmd (handling nil) and returns the resulting
// message. Used by the paste tests to drive the async clipboard read
// path without a running Bubbletea program.
func runCmd(cmd tea.Cmd) tea.Msg {
	if cmd == nil {
		return nil
	}
	return cmd()
}

// TestPasteImageInsertsPlaceholder exercises the full async flow:
// Ctrl+V → readClipboardImageCmd → clipboardImageMsg → placeholder
// insertion. It verifies that after the round-trip the textarea holds
// `[Image #1]` and the pending map carries the canned bytes under
// index 1.
func TestPasteImageInsertsPlaceholder(t *testing.T) {
	t.Parallel()

	reader := func(ctx context.Context) ([]byte, string, error) {
		return cannedPNG, "image/png", nil
	}
	m := newPasteTestModel(t, reader)

	// Size the window so resizeTextarea has something to work with.
	updated, _ := m.Update(tea.WindowSizeMsg{Width: 80, Height: 30})
	m = updated.(*Model)

	// Ctrl+V — the returned cmd should carry the clipboard probe.
	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlV})
	m = updated.(*Model)
	if cmd == nil {
		t.Fatal("Ctrl+V should produce a non-nil cmd")
	}

	// Execute the async chain until we get a clipboardImageMsg. tea
	// wraps commands in Batch, which isn't directly introspectable, so
	// we walk any batch by relying on the fact that the only non-nil
	// work we queued was readClipboardImageCmd. Its Cmd is a plain
	// function that returns a clipboardImageMsg; running it directly
	// via the model is the cleanest path.
	msg := runCmd(m.readClipboardImageCmd())
	cimg, ok := msg.(clipboardImageMsg)
	if !ok {
		t.Fatalf("expected clipboardImageMsg, got %T", msg)
	}
	if cimg.err != nil {
		t.Fatalf("unexpected clipboard error: %v", cimg.err)
	}
	if !bytes.Equal(cimg.data, cannedPNG) {
		t.Fatalf("clipboard bytes mismatch: got %v, want %v", cimg.data, cannedPNG)
	}

	// Feed the clipboardImageMsg back into Update.
	updated, _ = m.Update(cimg)
	m = updated.(*Model)

	if got := m.textarea.Value(); got != "[Image #1]" {
		t.Errorf("textarea value: got %q, want %q", got, "[Image #1]")
	}
	if m.nextImageIdx != 1 {
		t.Errorf("nextImageIdx: got %d, want 1", m.nextImageIdx)
	}
	pend, ok := m.pendingImages[1]
	if !ok {
		t.Fatal("pendingImages[1] missing")
	}
	if !bytes.Equal(pend.data, cannedPNG) {
		t.Errorf("pending data mismatch: got %v, want %v", pend.data, cannedPNG)
	}
	if pend.mime != "image/png" {
		t.Errorf("pending mime: got %q, want image/png", pend.mime)
	}
}

// TestPasteThenSendEmitsBlockImage verifies the full paste→type→send
// round trip: after pasting an image, typing descriptive text, and
// pressing Enter, the session's new user message must contain a
// BlockImage (with the canned bytes) followed by a BlockText carrying
// the description.
func TestPasteThenSendEmitsBlockImage(t *testing.T) {
	t.Parallel()

	reader := func(ctx context.Context) ([]byte, string, error) {
		return cannedPNG, "image/png", nil
	}
	m := newPasteTestModel(t, reader)

	updated, _ := m.Update(tea.WindowSizeMsg{Width: 80, Height: 30})
	m = updated.(*Model)

	// Paste the image.
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyCtrlV})
	m = updated.(*Model)
	msg := runCmd(m.readClipboardImageCmd())
	updated, _ = m.Update(msg)
	m = updated.(*Model)

	if m.textarea.Value() != "[Image #1]" {
		t.Fatalf("paste precondition failed: textarea = %q", m.textarea.Value())
	}

	// Type " describe".
	for _, r := range " describe" {
		updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
		m = updated.(*Model)
	}
	if want := "[Image #1] describe"; m.textarea.Value() != want {
		t.Fatalf("textarea after typing: got %q, want %q", m.textarea.Value(), want)
	}

	// Send.
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(*Model)

	if len(m.session.Messages) != 1 {
		t.Fatalf("session.Messages: got %d, want 1", len(m.session.Messages))
	}
	um := m.session.Messages[0]
	if um.Role != conversation.RoleUser {
		t.Errorf("message role: got %q, want %q", um.Role, conversation.RoleUser)
	}
	if len(um.Blocks) != 2 {
		t.Fatalf("message block count: got %d, want 2\n%#v", len(um.Blocks), um.Blocks)
	}
	if um.Blocks[0].Type != conversation.BlockImage {
		t.Errorf("blocks[0] type: got %q, want %q", um.Blocks[0].Type, conversation.BlockImage)
	}
	if !bytes.Equal(um.Blocks[0].ImageData, cannedPNG) {
		t.Errorf("blocks[0] image data mismatch: got %v, want %v", um.Blocks[0].ImageData, cannedPNG)
	}
	if um.Blocks[0].MimeType != "image/png" {
		t.Errorf("blocks[0] mime: got %q, want image/png", um.Blocks[0].MimeType)
	}
	if um.Blocks[1].Type != conversation.BlockText {
		t.Errorf("blocks[1] type: got %q, want %q", um.Blocks[1].Type, conversation.BlockText)
	}
	if um.Blocks[1].Text != "describe" {
		t.Errorf("blocks[1] text: got %q, want %q", um.Blocks[1].Text, "describe")
	}

	// pendingImages should be cleared on send; nextImageIdx stays.
	if len(m.pendingImages) != 0 {
		t.Errorf("pendingImages should be cleared on send, got %v", m.pendingImages)
	}
	if m.nextImageIdx != 1 {
		t.Errorf("nextImageIdx should persist after send: got %d, want 1", m.nextImageIdx)
	}
}

// TestPasteFallsBackToTextWhenNoImage verifies that when the clipboard
// holds no image (reader returns errNoImage), the Model falls back to
// the textarea's default text-paste behaviour instead of inserting a
// placeholder. We check the negative behaviour (no image side-effects)
// plus a non-nil returned cmd chain, which is where textarea.Paste
// lives.
func TestPasteFallsBackToTextWhenNoImage(t *testing.T) {
	t.Parallel()

	reader := func(ctx context.Context) ([]byte, string, error) {
		return nil, "", errNoImage
	}
	m := newPasteTestModel(t, reader)

	updated, _ := m.Update(tea.WindowSizeMsg{Width: 80, Height: 30})
	m = updated.(*Model)

	// Ctrl+V produces the probe cmd.
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyCtrlV})
	m = updated.(*Model)

	// Run the probe — it should report no image.
	msg := runCmd(m.readClipboardImageCmd())
	cimg, ok := msg.(clipboardImageMsg)
	if !ok {
		t.Fatalf("expected clipboardImageMsg, got %T", msg)
	}
	if cimg.err == nil {
		t.Fatal("expected errNoImage from fake reader")
	}

	// Feed the error back into Update. The returned cmd chain must
	// include the text-paste fallback and must NOT insert any
	// placeholder into the textarea.
	updated, cmd := m.Update(cimg)
	m = updated.(*Model)

	if strings.Contains(m.textarea.Value(), "[Image") {
		t.Errorf("textarea should not contain image placeholder on fallback, got %q", m.textarea.Value())
	}
	if len(m.pendingImages) != 0 {
		t.Errorf("pendingImages should remain empty on fallback, got %v", m.pendingImages)
	}
	if m.nextImageIdx != 0 {
		t.Errorf("nextImageIdx should remain 0 on fallback, got %d", m.nextImageIdx)
	}
	if cmd == nil {
		t.Error("expected non-nil cmd containing text-paste fallback, got nil")
	}
}

// TestPasteNewSessionResetsCounter verifies that /new clears both the
// pending map and the monotonic counter, so a fresh session starts
// numbering at 1 again.
func TestPasteNewSessionResetsCounter(t *testing.T) {
	t.Parallel()

	reader := func(ctx context.Context) ([]byte, string, error) {
		return cannedPNG, "image/png", nil
	}
	m := newPasteTestModel(t, reader)
	updated, _ := m.Update(tea.WindowSizeMsg{Width: 80, Height: 30})
	m = updated.(*Model)

	// Paste once, confirm counter advanced.
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyCtrlV})
	m = updated.(*Model)
	updated, _ = m.Update(runCmd(m.readClipboardImageCmd()))
	m = updated.(*Model)
	if m.nextImageIdx != 1 {
		t.Fatalf("nextImageIdx after first paste: got %d, want 1", m.nextImageIdx)
	}

	// /new resets.
	m.executeNewSession()

	if m.nextImageIdx != 0 {
		t.Errorf("nextImageIdx after /new: got %d, want 0", m.nextImageIdx)
	}
	if len(m.pendingImages) != 0 {
		t.Errorf("pendingImages after /new: got %v, want empty", m.pendingImages)
	}

	// Paste again — numbering should restart at 1.
	// The textarea still holds the old placeholder from the first
	// paste because /new only clears the output/session, not the
	// textarea contents. Reset it explicitly to match the command
	// handler's expected state.
	m.textarea.Reset()
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyCtrlV})
	m = updated.(*Model)
	updated, _ = m.Update(runCmd(m.readClipboardImageCmd()))
	m = updated.(*Model)

	if m.textarea.Value() != "[Image #1]" {
		t.Errorf("post-/new paste: got %q, want %q", m.textarea.Value(), "[Image #1]")
	}
	if _, ok := m.pendingImages[1]; !ok {
		t.Errorf("pendingImages[1] missing after post-/new paste")
	}
}
