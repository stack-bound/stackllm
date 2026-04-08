// Package config persists non-secret user preferences for stackllm.
//
// Secrets (API keys, OAuth tokens) live in auth.FileStore. This package
// stores provider defaults and provider-specific settings like Ollama base URLs.
//
// Default location: $XDG_CONFIG_HOME/stackllm/config.json
// or ~/.config/stackllm/config.json if XDG_CONFIG_HOME is not set.
package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// ProviderSettings holds non-secret, provider-specific configuration.
type ProviderSettings struct {
	BaseURL    string `json:"base_url,omitempty"`    // Ollama
	Endpoint   string `json:"endpoint,omitempty"`    // Azure
	Deployment string `json:"deployment,omitempty"`  // Azure
	APIVersion string `json:"api_version,omitempty"` // Azure
}

// Config holds user preferences persisted to disk.
type Config struct {
	DefaultProvider string                      `json:"default_provider,omitempty"`
	DefaultModel    string                      `json:"default_model,omitempty"`
	Providers       map[string]ProviderSettings `json:"providers,omitempty"`
}

// Store reads and writes Config to a JSON file.
type Store struct {
	AppName string // used to build XDG path; ignored if Path is set
	Path    string // override for testing
}

func (s *Store) path() string {
	if s.Path != "" {
		return s.Path
	}
	dir := os.Getenv("XDG_CONFIG_HOME")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".config")
	}
	return filepath.Join(dir, s.AppName, "config.json")
}

// Load reads the config from disk. If the file does not exist, it returns
// a zero-value Config (not an error).
func (s *Store) Load() (*Config, error) {
	data, err := os.ReadFile(s.path())
	if err != nil {
		if os.IsNotExist(err) {
			return &Config{}, nil
		}
		return nil, fmt.Errorf("config: read: %w", err)
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("config: unmarshal: %w", err)
	}
	return &cfg, nil
}

// Save writes the config to disk atomically.
func (s *Store) Save(cfg *Config) error {
	p := s.path()
	if err := os.MkdirAll(filepath.Dir(p), 0700); err != nil {
		return fmt.Errorf("config: create dir: %w", err)
	}

	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return fmt.Errorf("config: marshal: %w", err)
	}

	// Atomic write: write to temp file, then rename.
	tmp := p + ".tmp"
	if err := os.WriteFile(tmp, data, 0600); err != nil {
		return fmt.Errorf("config: write tmp: %w", err)
	}
	if err := os.Rename(tmp, p); err != nil {
		return fmt.Errorf("config: rename: %w", err)
	}
	return nil
}
