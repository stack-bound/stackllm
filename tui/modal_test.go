package tui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func TestOpenRenameModal_PrefillsCurrentName(t *testing.T) {
	t.Parallel()
	m := testModel(t, newFullFakeStore())
	m.session.Name = "prefill"

	m.openRenameModal()
	if m.state != stateTextModal {
		t.Errorf("expected stateTextModal, got %v", m.state)
	}
	if m.modalKind != modalRename {
		t.Errorf("expected modalRename, got %v", m.modalKind)
	}
	if m.modalInput.Value() != "prefill" {
		t.Errorf("expected modal input to prefill with current name, got %q", m.modalInput.Value())
	}
	if !strings.Contains(m.modalTitle, "Rename") {
		t.Errorf("expected Rename in title, got %q", m.modalTitle)
	}
}

func TestRenameModal_FullFlow(t *testing.T) {
	t.Parallel()
	store := newFullFakeStore()
	m := testModel(t, store)

	// Open the rename modal — input starts empty so we can type the
	// whole name and observe the buffer accumulate.
	m.openRenameModal()
	// Clear the prefill so we test actual typing.
	m.modalInput.SetValue("")

	typed := "new name"
	for _, r := range typed {
		updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
		m = updated.(*Model)
	}

	if m.modalInput.Value() != typed {
		t.Errorf("expected modal input to contain %q, got %q", typed, m.modalInput.Value())
	}

	// Submit with Enter.
	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(*Model)

	if m.state != stateIdle {
		t.Errorf("expected stateIdle after submit, got %v", m.state)
	}
	if m.session.Name != typed {
		t.Errorf("expected session name %q, got %q", typed, m.session.Name)
	}
	// The fake store saw a Save with the new name.
	if got := store.sessions[m.session.ID]; got == nil || got.Name != typed {
		t.Errorf("expected store to have session with name %q", typed)
	}
}

func TestRenameModal_EscClosesWithoutSaving(t *testing.T) {
	t.Parallel()
	m := testModel(t, newFullFakeStore())
	m.session.Name = "original"
	m.openRenameModal()
	m.modalInput.SetValue("should not save")

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = updated.(*Model)

	if m.state != stateIdle {
		t.Errorf("expected stateIdle after Esc, got %v", m.state)
	}
	if m.session.Name != "original" {
		t.Errorf("expected name to remain unchanged, got %q", m.session.Name)
	}
}

func TestExportModal_PrefillsDefaultPath(t *testing.T) {
	t.Parallel()
	m := testModel(t, newFullFakeStore())
	m.openExportModal()
	if m.state != stateTextModal {
		t.Errorf("expected stateTextModal, got %v", m.state)
	}
	if m.modalKind != modalExport {
		t.Errorf("expected modalExport, got %v", m.modalKind)
	}
	if !strings.HasPrefix(m.modalInput.Value(), "~/stackllm-") {
		t.Errorf("expected default path prefill, got %q", m.modalInput.Value())
	}
	if !strings.HasSuffix(m.modalInput.Value(), ".jsonl") {
		t.Errorf("expected .jsonl suffix, got %q", m.modalInput.Value())
	}
}

func TestExportModal_MissingCapabilityDoesNotOpen(t *testing.T) {
	t.Parallel()
	m := testModel(t, &minimalStore{inner: newFullFakeStore()})
	m.openExportModal()
	if m.state == stateTextModal {
		t.Errorf("export modal should not open when store lacks SessionExporter")
	}
}

func TestDeleteCommand_OpensConfirmModal(t *testing.T) {
	t.Parallel()
	store := newFullFakeStore()
	m := testModel(t, store)

	// Seed the store with the current session so a successful
	// confirm has something to delete.
	m.session.Name = "to-be-deleted"
	store.Save(nil, m.session)
	oldID := m.session.ID

	// /delete should open the confirm modal, not immediately delete.
	cmd := m.executeCommand(Command{ID: CommandDelete})
	if cmd != nil {
		t.Errorf("expected openConfirmModal to return nil cmd, got one")
	}
	if m.state != stateConfirmModal {
		t.Fatalf("expected stateConfirmModal, got %v", m.state)
	}
	if !strings.Contains(m.confirmTitle, "Delete") {
		t.Errorf("expected 'Delete' in title, got %q", m.confirmTitle)
	}
	if !strings.Contains(m.confirmPrompt, "to-be-deleted") {
		t.Errorf("expected session name in prompt, got %q", m.confirmPrompt)
	}
	// The session must still exist at this point.
	if _, err := store.Load(nil, oldID); err != nil {
		t.Errorf("session should not be deleted before confirm, got err %v", err)
	}
}

func TestConfirmModal_YesRunsAction(t *testing.T) {
	t.Parallel()
	store := newFullFakeStore()
	m := testModel(t, store)
	m.session.Name = "bye"
	store.Save(nil, m.session)
	oldID := m.session.ID

	m.executeCommand(Command{ID: CommandDelete})

	// Press 'y' to confirm.
	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'y'}})
	m = updated.(*Model)

	if m.state != stateIdle {
		t.Errorf("expected stateIdle after confirm, got %v", m.state)
	}
	if _, err := store.Load(nil, oldID); err == nil {
		t.Error("expected old session to be deleted from store after confirm")
	}
	if m.session.ID == oldID {
		t.Error("expected a fresh session after delete confirm")
	}
}

func TestConfirmModal_EnterRunsAction(t *testing.T) {
	t.Parallel()
	store := newFullFakeStore()
	m := testModel(t, store)
	m.session.Name = "bye"
	store.Save(nil, m.session)
	oldID := m.session.ID

	m.executeCommand(Command{ID: CommandDelete})

	// Enter should also confirm.
	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(*Model)

	if m.state != stateIdle {
		t.Errorf("expected stateIdle after confirm, got %v", m.state)
	}
	if _, err := store.Load(nil, oldID); err == nil {
		t.Error("expected old session to be deleted after Enter confirm")
	}
}

func TestConfirmModal_NoCancels(t *testing.T) {
	t.Parallel()
	store := newFullFakeStore()
	m := testModel(t, store)
	m.session.Name = "keep me"
	store.Save(nil, m.session)
	oldID := m.session.ID

	m.executeCommand(Command{ID: CommandDelete})

	// 'n' should cancel without running the action.
	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'n'}})
	m = updated.(*Model)

	if m.state != stateIdle {
		t.Errorf("expected stateIdle after cancel, got %v", m.state)
	}
	if m.session.ID != oldID {
		t.Error("expected current session to remain unchanged after cancel")
	}
	if _, err := store.Load(nil, oldID); err != nil {
		t.Errorf("expected session to still exist after cancel, got err %v", err)
	}
}

func TestConfirmModal_EscCancels(t *testing.T) {
	t.Parallel()
	store := newFullFakeStore()
	m := testModel(t, store)
	m.session.Name = "keep me"
	store.Save(nil, m.session)
	oldID := m.session.ID

	m.executeCommand(Command{ID: CommandDelete})

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = updated.(*Model)

	if m.state != stateIdle {
		t.Errorf("expected stateIdle after Esc, got %v", m.state)
	}
	if m.session.ID != oldID {
		t.Error("expected current session to remain unchanged after Esc")
	}
}

func TestRenderConfirmModal_ContainsTitleAndHint(t *testing.T) {
	t.Parallel()
	m := testModel(t, newFullFakeStore())
	m.session.Name = "target"
	m.executeCommand(Command{ID: CommandDelete})

	out := m.renderConfirmModal()
	if !strings.Contains(out, "Delete session") {
		t.Errorf("expected title in confirm modal, got:\n%s", out)
	}
	if !strings.Contains(out, "target") {
		t.Errorf("expected session name in prompt, got:\n%s", out)
	}
	if !strings.Contains(out, "y confirm") {
		t.Errorf("expected confirm hint, got:\n%s", out)
	}
}

func TestRenderModal_ContainsTitleAndHint(t *testing.T) {
	t.Parallel()
	m := testModel(t, newFullFakeStore())
	m.openRenameModal()

	out := m.renderModal()
	if !strings.Contains(out, "Rename session") {
		t.Errorf("expected modal title, got:\n%s", out)
	}
	if !strings.Contains(out, "enter save") {
		t.Errorf("expected hint line, got:\n%s", out)
	}
}
