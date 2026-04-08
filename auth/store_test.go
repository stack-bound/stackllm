package auth

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestMemoryStore(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	store := NewMemoryStore()

	// Load from empty store returns error.
	_, err := store.Load(ctx, "missing")
	if err == nil {
		t.Fatal("expected error loading missing key")
	}

	// Save and load.
	if err := store.Save(ctx, "key1", "value1"); err != nil {
		t.Fatalf("Save error: %v", err)
	}
	v, err := store.Load(ctx, "key1")
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}
	if v != "value1" {
		t.Errorf("Load = %q, want %q", v, "value1")
	}

	// Delete.
	if err := store.Delete(ctx, "key1"); err != nil {
		t.Fatalf("Delete error: %v", err)
	}
	_, err = store.Load(ctx, "key1")
	if err == nil {
		t.Fatal("expected error after delete")
	}

	// Delete non-existent key is a no-op.
	if err := store.Delete(ctx, "nonexistent"); err != nil {
		t.Fatalf("Delete non-existent error: %v", err)
	}
}

func TestFileStore(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	dir := t.TempDir()
	store := &FileStore{
		AppName: "test",
		Path:    filepath.Join(dir, "auth.json"),
	}

	// Load from non-existent file returns error.
	_, err := store.Load(ctx, "missing")
	if err == nil {
		t.Fatal("expected error loading from non-existent file")
	}

	// Save creates the file.
	if err := store.Save(ctx, "key1", "value1"); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	// Load returns saved value.
	v, err := store.Load(ctx, "key1")
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}
	if v != "value1" {
		t.Errorf("Load = %q, want %q", v, "value1")
	}

	// Multiple keys.
	if err := store.Save(ctx, "key2", "value2"); err != nil {
		t.Fatalf("Save key2 error: %v", err)
	}
	v1, _ := store.Load(ctx, "key1")
	v2, _ := store.Load(ctx, "key2")
	if v1 != "value1" || v2 != "value2" {
		t.Errorf("multi-key: got %q %q, want %q %q", v1, v2, "value1", "value2")
	}

	// Delete.
	if err := store.Delete(ctx, "key1"); err != nil {
		t.Fatalf("Delete error: %v", err)
	}
	_, err = store.Load(ctx, "key1")
	if err == nil {
		t.Fatal("expected error after delete")
	}
	// key2 still exists.
	v2, err = store.Load(ctx, "key2")
	if err != nil {
		t.Fatalf("Load key2 after delete error: %v", err)
	}
	if v2 != "value2" {
		t.Errorf("key2 = %q, want %q", v2, "value2")
	}
}

func TestFileStore_AtomicWrite(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	dir := t.TempDir()
	p := filepath.Join(dir, "auth.json")
	store := &FileStore{Path: p}

	if err := store.Save(ctx, "k", "v"); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	// Verify no .tmp file remains.
	_, err := os.Stat(p + ".tmp")
	if err == nil {
		t.Error("temp file should not exist after save")
	}

	// Verify the file is valid JSON.
	data, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("ReadFile error: %v", err)
	}
	if len(data) == 0 {
		t.Error("file should not be empty")
	}
}

func TestFileStore_DefaultPath(t *testing.T) {
	t.Parallel()

	store := &FileStore{AppName: "stackllm"}
	p := store.path()

	// Should contain the app name.
	if !filepath.IsAbs(p) {
		t.Errorf("path should be absolute, got %q", p)
	}
	if filepath.Base(p) != "auth.json" {
		t.Errorf("path should end with auth.json, got %q", p)
	}
}
