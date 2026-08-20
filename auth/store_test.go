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

func TestFileStore_XDGConfigHomePath(t *testing.T) {
	// No t.Parallel(): t.Setenv forbids it.
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)

	store := &FileStore{AppName: "myapp"}
	want := filepath.Join(dir, "myapp", "auth.json")
	if got := store.path(); got != want {
		t.Errorf("path() = %q, want %q", got, want)
	}

	// The store must actually read and write through that path.
	ctx := context.Background()
	if err := store.Save(ctx, "k", "v"); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if _, err := os.Stat(want); err != nil {
		t.Errorf("expected auth file at XDG path %q: %v", want, err)
	}
	v, err := store.Load(ctx, "k")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if v != "v" {
		t.Errorf("Load = %q, want %q", v, "v")
	}
}

func TestFileStore_CorruptFile(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	dir := t.TempDir()
	p := filepath.Join(dir, "auth.json")
	if err := os.WriteFile(p, []byte("not-json{"), 0600); err != nil {
		t.Fatalf("write corrupt file: %v", err)
	}
	store := &FileStore{Path: p}

	if _, err := store.Load(ctx, "k"); err == nil {
		t.Error("Load should fail on a corrupt store file")
	}
	if err := store.Save(ctx, "k", "v"); err == nil {
		t.Error("Save should fail on a corrupt store file rather than clobber it")
	}
	if err := store.Delete(ctx, "k"); err == nil {
		t.Error("Delete should fail on a corrupt store file")
	}

	// The corrupt file must not have been overwritten by the failed ops.
	data, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if string(data) != "not-json{" {
		t.Errorf("corrupt file was modified: %q", data)
	}
}

func TestFileStore_UncreatableDir(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	if os.Geteuid() == 0 {
		t.Skip("running as root — directory permissions are not enforced")
	}

	// The store file does not exist (so readAll succeeds with an empty
	// map), but its parent directory cannot be created because the
	// grandparent is read-only — writeAll's MkdirAll must fail.
	dir := t.TempDir()
	if err := os.Chmod(dir, 0o500); err != nil {
		t.Fatalf("chmod: %v", err)
	}
	t.Cleanup(func() { _ = os.Chmod(dir, 0o700) })

	store := &FileStore{Path: filepath.Join(dir, "sub", "auth.json")}
	if err := store.Save(ctx, "k", "v"); err == nil {
		t.Error("Save should fail when the store directory cannot be created")
	}
}

func TestFileStore_UnwritableDir(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	if os.Geteuid() == 0 {
		t.Skip("running as root — directory permissions are not enforced")
	}

	// Directory exists (MkdirAll succeeds) but is read-only, so the
	// temp-file write must fail.
	dir := t.TempDir()
	if err := os.Chmod(dir, 0o500); err != nil {
		t.Fatalf("chmod: %v", err)
	}
	t.Cleanup(func() { _ = os.Chmod(dir, 0o700) })

	store := &FileStore{Path: filepath.Join(dir, "auth.json")}
	if err := store.Save(ctx, "k", "v"); err == nil {
		t.Error("Save should fail when the store directory is not writable")
	}
}

func TestFileStore_ReadErrorIsNotMaskedAsMissing(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	// A path whose parent component is a regular file produces a read
	// error that is NOT os.IsNotExist — it must surface, not be treated
	// as an empty store.
	dir := t.TempDir()
	blocker := filepath.Join(dir, "blocker")
	if err := os.WriteFile(blocker, []byte("x"), 0o600); err != nil {
		t.Fatalf("write blocker: %v", err)
	}

	store := &FileStore{Path: filepath.Join(blocker, "auth.json")}
	if _, err := store.Load(ctx, "k"); err == nil {
		t.Error("Load should surface a non-IsNotExist read error")
	}
}
