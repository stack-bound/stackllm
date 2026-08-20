package tui

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stack-bound/stackllm/agent"
	"github.com/stack-bound/stackllm/provider"
)

func TestSubmitExport_RejectsEmptyPath(t *testing.T) {
	t.Parallel()
	m := testModel(t, newFullFakeStore())
	m.submitExport("   ")
	if !strings.Contains(m.output.String(), "path cannot be empty") {
		t.Errorf("expected empty-path error, got:\n%s", m.output.String())
	}
}

func TestSubmitExport_SaveFailureAborts(t *testing.T) {
	t.Parallel()
	m := testModel(t, &failingSaveStore{fullFakeStore: newFullFakeStore()})
	path := filepath.Join(t.TempDir(), "out.jsonl")
	m.submitExport(path)
	if !strings.Contains(m.output.String(), "✗ /export: disk full") {
		t.Errorf("expected save error surfaced, got:\n%s", m.output.String())
	}
	if _, err := os.Stat(path); err == nil {
		t.Error("export file must not be created when the pre-export save fails")
	}
}

func TestSubmitExport_CreateFailure(t *testing.T) {
	t.Parallel()
	store := newFullFakeStore()
	m := testModel(t, store)
	store.Save(context.Background(), m.session)
	// The target path is an existing directory, so os.Create must fail.
	dir := t.TempDir()
	m.submitExport(dir)
	if !strings.Contains(m.output.String(), "✗ /export:") {
		t.Errorf("expected create error surfaced, got:\n%s", m.output.String())
	}
}

func TestSubmitExport_MkdirFailure(t *testing.T) {
	t.Parallel()
	store := newFullFakeStore()
	m := testModel(t, store)
	store.Save(context.Background(), m.session)
	// Parent "directory" is a regular file, so MkdirAll must fail.
	blocker := filepath.Join(t.TempDir(), "blocker")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatalf("write blocker: %v", err)
	}
	m.submitExport(filepath.Join(blocker, "sub", "out.jsonl"))
	if !strings.Contains(m.output.String(), "✗ /export:") {
		t.Errorf("expected mkdir error surfaced, got:\n%s", m.output.String())
	}
}

// failingExportStore delegates everything to fullFakeStore but errors
// on ExportJSONL, exercising the export-write failure path.
type failingExportStore struct{ *fullFakeStore }

func (f *failingExportStore) ExportJSONL(_ context.Context, _ string, _ io.Writer) error {
	return fmt.Errorf("serialisation blew up")
}

func TestSubmitExport_ExportFailure(t *testing.T) {
	t.Parallel()
	m := testModel(t, &failingExportStore{fullFakeStore: newFullFakeStore()})
	path := filepath.Join(t.TempDir(), "out.jsonl")
	m.submitExport(path)
	if !strings.Contains(m.output.String(), "✗ /export: serialisation blew up") {
		t.Errorf("expected export error surfaced, got:\n%s", m.output.String())
	}
}

func TestExpandHome_BareTilde(t *testing.T) {
	t.Parallel()
	got, err := expandHome("~")
	if err != nil {
		t.Fatalf("expandHome(~): %v", err)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("UserHomeDir: %v", err)
	}
	if got != home {
		t.Errorf("expandHome(~) = %q, want %q", got, home)
	}
}

func TestExecuteDelete_ErrorPaths(t *testing.T) {
	t.Parallel()

	t.Run("nil store", func(t *testing.T) {
		t.Parallel()
		p := provider.New(provider.OllamaConfig("http://localhost", "test"))
		m := New(agent.New(p), nil)
		m.executeDelete()
		if !strings.Contains(m.output.String(), "✗ /delete: no session store configured") {
			t.Errorf("expected no-store error, got:\n%s", m.output.String())
		}
	})

	t.Run("store delete failure keeps session", func(t *testing.T) {
		t.Parallel()
		m := testModel(t, &failingDeleteStore{fullFakeStore: newFullFakeStore()})
		oldID := m.session.ID
		m.executeDelete()
		if m.session.ID != oldID {
			t.Error("session must be left untouched when delete fails")
		}
		if !strings.Contains(m.output.String(), "✗ /delete: database locked") {
			t.Errorf("expected delete error, got:\n%s", m.output.String())
		}
	})
}

func TestSubmitRename_NoStoreStillRenames(t *testing.T) {
	t.Parallel()
	p := provider.New(provider.OllamaConfig("http://localhost", "test"))
	m := New(agent.New(p), nil)
	m.submitRename("offline name")
	if m.session.Name != "offline name" {
		t.Errorf("expected in-memory rename without a store, got %q", m.session.Name)
	}
	if !strings.Contains(m.output.String(), `✓ renamed to "offline name"`) {
		t.Errorf("expected rename feedback, got:\n%s", m.output.String())
	}
}
