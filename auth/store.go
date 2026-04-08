package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// TokenStore persists tokens by key.
type TokenStore interface {
	Load(ctx context.Context, key string) (string, error)
	Save(ctx context.Context, key string, value string) error
	Delete(ctx context.Context, key string) error
}

// FileStore persists tokens as a JSON map at a file path.
// Default location: $XDG_CONFIG_HOME/<AppName>/auth.json
// or ~/.config/<AppName>/auth.json if XDG_CONFIG_HOME is not set.
type FileStore struct {
	AppName string
	Path    string // override; if empty, uses XDG default
}

func (f *FileStore) path() string {
	if f.Path != "" {
		return f.Path
	}
	dir := os.Getenv("XDG_CONFIG_HOME")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".config")
	}
	return filepath.Join(dir, f.AppName, "auth.json")
}

func (f *FileStore) readAll() (map[string]string, error) {
	data, err := os.ReadFile(f.path())
	if err != nil {
		if os.IsNotExist(err) {
			return make(map[string]string), nil
		}
		return nil, fmt.Errorf("auth: read store: %w", err)
	}
	m := make(map[string]string)
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("auth: unmarshal store: %w", err)
	}
	return m, nil
}

func (f *FileStore) writeAll(m map[string]string) error {
	p := f.path()
	if err := os.MkdirAll(filepath.Dir(p), 0700); err != nil {
		return fmt.Errorf("auth: create store dir: %w", err)
	}

	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("auth: marshal store: %w", err)
	}

	// Atomic write: write to temp file, then rename.
	tmp := p + ".tmp"
	if err := os.WriteFile(tmp, data, 0600); err != nil {
		return fmt.Errorf("auth: write store tmp: %w", err)
	}
	if err := os.Rename(tmp, p); err != nil {
		return fmt.Errorf("auth: rename store: %w", err)
	}
	return nil
}

// Load reads the value for key from the store file.
func (f *FileStore) Load(_ context.Context, key string) (string, error) {
	m, err := f.readAll()
	if err != nil {
		return "", err
	}
	v, ok := m[key]
	if !ok {
		return "", fmt.Errorf("auth: key %q not found", key)
	}
	return v, nil
}

// Save writes the key-value pair to the store file.
func (f *FileStore) Save(_ context.Context, key, value string) error {
	m, err := f.readAll()
	if err != nil {
		return err
	}
	m[key] = value
	return f.writeAll(m)
}

// Delete removes the key from the store file.
func (f *FileStore) Delete(_ context.Context, key string) error {
	m, err := f.readAll()
	if err != nil {
		return err
	}
	delete(m, key)
	return f.writeAll(m)
}

// MemoryStore is an in-memory TokenStore. Used in tests and when persistence
// is not needed.
type MemoryStore struct {
	mu   sync.RWMutex
	data map[string]string
}

// NewMemoryStore creates a new in-memory token store.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{data: make(map[string]string)}
}

// Load returns the value for key.
func (m *MemoryStore) Load(_ context.Context, key string) (string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	v, ok := m.data[key]
	if !ok {
		return "", fmt.Errorf("auth: key %q not found", key)
	}
	return v, nil
}

// Save stores the key-value pair.
func (m *MemoryStore) Save(_ context.Context, key, value string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.data[key] = value
	return nil
}

// Delete removes the key.
func (m *MemoryStore) Delete(_ context.Context, key string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.data, key)
	return nil
}
