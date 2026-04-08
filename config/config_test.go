package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestStore_LoadMissingFile(t *testing.T) {
	t.Parallel()

	store := &Store{Path: filepath.Join(t.TempDir(), "nonexistent", "config.json")}
	cfg, err := store.Load()
	if err != nil {
		t.Fatalf("Load missing file should not error, got: %v", err)
	}
	if cfg.DefaultProvider != "" {
		t.Errorf("DefaultProvider = %q, want empty", cfg.DefaultProvider)
	}
	if cfg.DefaultModel != "" {
		t.Errorf("DefaultModel = %q, want empty", cfg.DefaultModel)
	}
	if cfg.Providers != nil {
		t.Errorf("Providers = %v, want nil", cfg.Providers)
	}
}

func TestStore_SaveAndLoad(t *testing.T) {
	t.Parallel()

	store := &Store{Path: filepath.Join(t.TempDir(), "config.json")}

	original := &Config{
		DefaultProvider: "copilot",
		DefaultModel:    "gpt-5.4",
		Providers: map[string]ProviderSettings{
			"ollama": {BaseURL: "http://localhost:11434"},
			"azure": {
				Endpoint:   "https://myinstance.openai.azure.com",
				Deployment: "gpt-4",
				APIVersion: "2024-02-01",
			},
		},
	}

	if err := store.Save(original); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	loaded, err := store.Load()
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}

	if loaded.DefaultProvider != original.DefaultProvider {
		t.Errorf("DefaultProvider = %q, want %q", loaded.DefaultProvider, original.DefaultProvider)
	}
	if loaded.DefaultModel != original.DefaultModel {
		t.Errorf("DefaultModel = %q, want %q", loaded.DefaultModel, original.DefaultModel)
	}

	if len(loaded.Providers) != 2 {
		t.Fatalf("Providers count = %d, want 2", len(loaded.Providers))
	}

	ollama := loaded.Providers["ollama"]
	if ollama.BaseURL != "http://localhost:11434" {
		t.Errorf("ollama BaseURL = %q, want %q", ollama.BaseURL, "http://localhost:11434")
	}

	azure := loaded.Providers["azure"]
	if azure.Endpoint != "https://myinstance.openai.azure.com" {
		t.Errorf("azure Endpoint = %q", azure.Endpoint)
	}
	if azure.Deployment != "gpt-4" {
		t.Errorf("azure Deployment = %q", azure.Deployment)
	}
	if azure.APIVersion != "2024-02-01" {
		t.Errorf("azure APIVersion = %q", azure.APIVersion)
	}
}

func TestStore_SaveOverwrite(t *testing.T) {
	t.Parallel()

	store := &Store{Path: filepath.Join(t.TempDir(), "config.json")}

	cfg1 := &Config{DefaultProvider: "openai", DefaultModel: "gpt-5.4"}
	if err := store.Save(cfg1); err != nil {
		t.Fatalf("Save 1 error: %v", err)
	}

	cfg2 := &Config{DefaultProvider: "copilot", DefaultModel: "gpt-5.4-mini"}
	if err := store.Save(cfg2); err != nil {
		t.Fatalf("Save 2 error: %v", err)
	}

	loaded, err := store.Load()
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}
	if loaded.DefaultProvider != "copilot" {
		t.Errorf("DefaultProvider = %q, want %q", loaded.DefaultProvider, "copilot")
	}
	if loaded.DefaultModel != "gpt-5.4-mini" {
		t.Errorf("DefaultModel = %q, want %q", loaded.DefaultModel, "gpt-5.4-mini")
	}
}

func TestStore_AtomicWrite(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	p := filepath.Join(dir, "config.json")
	store := &Store{Path: p}

	if err := store.Save(&Config{DefaultProvider: "openai"}); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	// No .tmp file should remain.
	_, err := os.Stat(p + ".tmp")
	if err == nil {
		t.Error("temp file should not exist after save")
	}

	// File should be valid JSON.
	data, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("ReadFile error: %v", err)
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("saved file is not valid JSON: %v", err)
	}
	if cfg.DefaultProvider != "openai" {
		t.Errorf("roundtrip DefaultProvider = %q, want %q", cfg.DefaultProvider, "openai")
	}
}

func TestStore_SaveCreatesDirectories(t *testing.T) {
	t.Parallel()

	p := filepath.Join(t.TempDir(), "nested", "deep", "config.json")
	store := &Store{Path: p}

	if err := store.Save(&Config{DefaultModel: "test"}); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	loaded, err := store.Load()
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}
	if loaded.DefaultModel != "test" {
		t.Errorf("DefaultModel = %q, want %q", loaded.DefaultModel, "test")
	}
}

func TestStore_DefaultPath(t *testing.T) {
	t.Parallel()

	store := &Store{AppName: "stackllm"}
	p := store.path()

	if !filepath.IsAbs(p) {
		t.Errorf("path should be absolute, got %q", p)
	}
	if filepath.Base(p) != "config.json" {
		t.Errorf("path should end with config.json, got %q", p)
	}
	if !containsSegment(p, "stackllm") {
		t.Errorf("path should contain app name, got %q", p)
	}
}

func TestStore_EmptyConfig(t *testing.T) {
	t.Parallel()

	store := &Store{Path: filepath.Join(t.TempDir(), "config.json")}

	if err := store.Save(&Config{}); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	loaded, err := store.Load()
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}
	if loaded.DefaultProvider != "" || loaded.DefaultModel != "" {
		t.Errorf("expected zero-value config, got provider=%q model=%q",
			loaded.DefaultProvider, loaded.DefaultModel)
	}
}

func TestStore_OmitsEmptyProviderSettings(t *testing.T) {
	t.Parallel()

	p := filepath.Join(t.TempDir(), "config.json")
	store := &Store{Path: p}

	cfg := &Config{
		DefaultProvider: "ollama",
		Providers: map[string]ProviderSettings{
			"ollama": {BaseURL: "http://localhost:11434"},
		},
	}
	if err := store.Save(cfg); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	// Read raw JSON and verify omitempty works.
	data, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("ReadFile error: %v", err)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal raw: %v", err)
	}

	// default_model should be omitted.
	if _, exists := raw["default_model"]; exists {
		t.Error("default_model should be omitted when empty")
	}

	// Verify ollama settings don't include empty azure fields.
	var providers map[string]map[string]json.RawMessage
	if err := json.Unmarshal(raw["providers"], &providers); err != nil {
		t.Fatalf("unmarshal providers: %v", err)
	}
	ollama := providers["ollama"]
	if _, exists := ollama["endpoint"]; exists {
		t.Error("endpoint should be omitted from ollama settings")
	}
	if _, exists := ollama["deployment"]; exists {
		t.Error("deployment should be omitted from ollama settings")
	}
}

func containsSegment(path, segment string) bool {
	dir := path
	for {
		var base string
		dir, base = filepath.Split(dir)
		if base == segment {
			return true
		}
		dir = filepath.Clean(dir)
		if dir == "." || dir == "/" {
			break
		}
	}
	return false
}
