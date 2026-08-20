package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoad_InvalidJSON(t *testing.T) {
	t.Parallel()

	p := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(p, []byte("{not json"), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}

	s := &Store{Path: p}
	_, err := s.Load()
	if err == nil {
		t.Fatal("expected unmarshal error for invalid JSON")
	}
	if !strings.Contains(err.Error(), "unmarshal") {
		t.Errorf("error = %v, want unmarshal error", err)
	}
}

func TestLoad_ReadErrorNotMissing(t *testing.T) {
	t.Parallel()

	// A directory at the config path produces a read error that is not
	// os.IsNotExist, so Load must fail instead of returning a zero config.
	dir := t.TempDir()
	s := &Store{Path: dir}
	_, err := s.Load()
	if err == nil {
		t.Fatal("expected read error when config path is a directory")
	}
	if !strings.Contains(err.Error(), "read") {
		t.Errorf("error = %v, want read error", err)
	}
}

func TestSave_MkdirFails(t *testing.T) {
	t.Parallel()

	// Parent "dir" is actually a file, so MkdirAll must fail.
	tmp := t.TempDir()
	blocker := filepath.Join(tmp, "blocker")
	if err := os.WriteFile(blocker, []byte("x"), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}

	s := &Store{Path: filepath.Join(blocker, "sub", "config.json")}
	err := s.Save(&Config{DefaultProvider: "openai"})
	if err == nil {
		t.Fatal("expected error when parent path is a file")
	}
	if !strings.Contains(err.Error(), "create dir") {
		t.Errorf("error = %v, want create dir error", err)
	}
}

func TestSave_TempWriteFails(t *testing.T) {
	t.Parallel()

	// The atomic-write temp path (<path>.tmp) exists as a directory, so
	// os.WriteFile must fail.
	dir := t.TempDir()
	p := filepath.Join(dir, "config.json")
	if err := os.Mkdir(p+".tmp", 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	s := &Store{Path: p}
	err := s.Save(&Config{DefaultProvider: "openai"})
	if err == nil {
		t.Fatal("expected error when temp path is a directory")
	}
	if !strings.Contains(err.Error(), "write tmp") {
		t.Errorf("error = %v, want write tmp error", err)
	}
}

func TestSave_RenameFails(t *testing.T) {
	t.Parallel()

	// The destination exists as a non-empty directory, so the final
	// rename must fail after the temp write succeeded.
	dir := t.TempDir()
	p := filepath.Join(dir, "config.json")
	if err := os.MkdirAll(filepath.Join(p, "occupied"), 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	s := &Store{Path: p}
	err := s.Save(&Config{DefaultProvider: "openai"})
	if err == nil {
		t.Fatal("expected error when destination is a non-empty directory")
	}
	if !strings.Contains(err.Error(), "rename") {
		t.Errorf("error = %v, want rename error", err)
	}
}
