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
