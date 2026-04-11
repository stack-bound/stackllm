package profile

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stack-bound/stackllm/auth"
	"github.com/stack-bound/stackllm/config"
	"github.com/stack-bound/stackllm/provider"
)

// redirectTransport rewrites all request URLs to the target host,
// allowing httptest servers to intercept requests regardless of the
// original URL the provider is configured with.
type redirectTransport struct {
	target *url.URL
	inner  http.RoundTripper
}

func (rt *redirectTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	req.URL.Scheme = rt.target.Scheme
	req.URL.Host = rt.target.Host
	return rt.inner.RoundTrip(req)
}

func redirectClient(srv *httptest.Server) *http.Client {
	u, _ := url.Parse(srv.URL)
	return &http.Client{
		Transport: &redirectTransport{target: u, inner: http.DefaultTransport},
	}
}

func testManager(t *testing.T, opts ...Option) (*Manager, *auth.MemoryStore, *config.Store) {
	t.Helper()
	as := auth.NewMemoryStore()
	cs := &config.Store{Path: filepath.Join(t.TempDir(), "config.json")}
	base := []Option{WithAuthStore(as), WithConfigStore(cs)}
	return New(append(base, opts...)...), as, cs
}

// modelsServer returns an httptest.Server that responds to GET /v1/models
// with the given model IDs.
func modelsServer(t *testing.T, models ...string) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		type modelEntry struct {
			ID string `json:"id"`
		}
		data := make([]modelEntry, len(models))
		for i, m := range models {
			data[i] = modelEntry{ID: m}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"data": data})
	})
	// Also handle /models for providers whose BaseURL already includes /v1.
	mux.HandleFunc("/models", func(w http.ResponseWriter, r *http.Request) {
		type modelEntry struct {
			ID string `json:"id"`
		}
		data := make([]modelEntry, len(models))
		for i, m := range models {
			data[i] = modelEntry{ID: m}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"data": data})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func TestAvailableProviders(t *testing.T) {
	t.Parallel()
	mgr, _, _ := testManager(t)

	providers := mgr.AvailableProviders()
	expected := []string{"openai", "copilot", "gemini", "ollama"}
	if len(providers) != len(expected) {
		t.Fatalf("got %d providers, want %d", len(providers), len(expected))
	}
	for i, p := range providers {
		if p != expected[i] {
			t.Errorf("providers[%d] = %q, want %q", i, p, expected[i])
		}
	}

	// Returned slice should be a copy, not the internal slice.
	providers[0] = "mutated"
	providers2 := mgr.AvailableProviders()
	if providers2[0] != "openai" {
		t.Error("AvailableProviders should return a copy")
	}
}

func TestLoginOpenAI(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, as, _ := testManager(t, WithCallbacks(Callbacks{
		OnPromptKey: func(name string) (string, error) {
			if name != "openai" {
				return "", fmt.Errorf("unexpected provider %q", name)
			}
			return "sk-test-key-123", nil
		},
	}))

	if err := mgr.Login(ctx, ProviderOpenAI); err != nil {
		t.Fatalf("Login error: %v", err)
	}

	// Verify the key was persisted.
	key, err := as.Load(ctx, keyOpenAI)
	if err != nil {
		t.Fatalf("Load key error: %v", err)
	}
	if key != "sk-test-key-123" {
		t.Errorf("stored key = %q, want %q", key, "sk-test-key-123")
	}
}

func TestLoginGemini(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, as, _ := testManager(t, WithCallbacks(Callbacks{
		OnPromptKey: func(name string) (string, error) {
			if name != "gemini" {
				return "", fmt.Errorf("unexpected provider %q", name)
			}
			return "AIza-gemini-key", nil
		},
	}))

	if err := mgr.Login(ctx, ProviderGemini); err != nil {
		t.Fatalf("Login error: %v", err)
	}

	key, err := as.Load(ctx, keyGemini)
	if err != nil {
		t.Fatalf("Load key error: %v", err)
	}
	if key != "AIza-gemini-key" {
		t.Errorf("stored key = %q, want %q", key, "AIza-gemini-key")
	}
}

func TestLoginOllama(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, _, cs := testManager(t, WithCallbacks(Callbacks{
		OnPromptURL: func(name, defaultURL string) (string, error) {
			if name != "ollama" {
				return "", fmt.Errorf("unexpected provider %q", name)
			}
			if defaultURL != "http://localhost:11434" {
				return "", fmt.Errorf("unexpected default URL %q", defaultURL)
			}
			return "http://myollama:8080", nil
		},
	}))

	if err := mgr.Login(ctx, ProviderOllama); err != nil {
		t.Fatalf("Login error: %v", err)
	}

	cfg, err := cs.Load()
	if err != nil {
		t.Fatalf("Load config error: %v", err)
	}
	if cfg.Providers == nil {
		t.Fatal("Providers map is nil after ollama login")
	}
	settings := cfg.Providers[ProviderOllama]
	if settings.BaseURL != "http://myollama:8080" {
		t.Errorf("ollama BaseURL = %q, want %q", settings.BaseURL, "http://myollama:8080")
	}
}

func TestLoginOllamaDefaultURL(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, _, cs := testManager(t, WithCallbacks(Callbacks{
		OnPromptURL: func(name, defaultURL string) (string, error) {
			return "", nil // empty means use default
		},
	}))

	if err := mgr.Login(ctx, ProviderOllama); err != nil {
		t.Fatalf("Login error: %v", err)
	}

	cfg, _ := cs.Load()
	settings := cfg.Providers[ProviderOllama]
	if settings.BaseURL != "http://localhost:11434" {
		t.Errorf("ollama BaseURL = %q, want default", settings.BaseURL)
	}
}

func TestLoginCopilot(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	// Mock the GitHub device flow endpoints.
	callCount := 0
	mux := http.NewServeMux()

	// Phase 1a: device code request.
	mux.HandleFunc("/login/device/code", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"device_code":      "test-device-code",
			"user_code":        "TEST-1234",
			"verification_uri": "https://github.com/login/device",
			"interval":         0,
			"expires_in":       900,
		})
	})

	// Phase 1b: poll for token.
	mux.HandleFunc("/login/oauth/access_token", func(w http.ResponseWriter, r *http.Request) {
		callCount++
		w.Header().Set("Content-Type", "application/json")
		// Return token immediately on first poll.
		json.NewEncoder(w).Encode(map[string]any{
			"access_token": "gho_test_github_token",
		})
	})

	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	var deviceCodeReceived, deviceURLReceived string
	var successCalled bool
	mgr, as, _ := testManager(t,
		WithHTTPClient(redirectClient(srv)),
		WithPollInterval(time.Millisecond),
		WithCallbacks(Callbacks{
			OnDeviceCode: func(userCode, verifyURL string) {
				deviceCodeReceived = userCode
				deviceURLReceived = verifyURL
			},
			OnPolling: func() {},
			OnSuccess: func() {
				successCalled = true
			},
		}),
	)

	if err := mgr.Login(ctx, ProviderCopilot); err != nil {
		t.Fatalf("Login error: %v", err)
	}

	// Verify the GitHub token was persisted.
	token, err := as.Load(ctx, keyCopilotGitHub)
	if err != nil {
		t.Fatalf("Load copilot token error: %v", err)
	}
	if token != "gho_test_github_token" {
		t.Errorf("stored token = %q, want %q", token, "gho_test_github_token")
	}

	// Verify callbacks were invoked.
	if deviceCodeReceived != "TEST-1234" {
		t.Errorf("device code = %q, want %q", deviceCodeReceived, "TEST-1234")
	}
	if deviceURLReceived == "" {
		t.Error("device URL should have been received")
	}
	if !successCalled {
		t.Error("OnSuccess should have been called")
	}
}

func TestLoadProvider_Copilot(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	// Mock the Copilot token exchange endpoint.
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// Respond to the token exchange with a valid Copilot token.
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"token":      "cop_test_token",
			"expires_at": 9999999999,
		})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	mgr, as, _ := testManager(t, WithHTTPClient(redirectClient(srv)))

	// Pre-populate the GitHub token (Phase 1 already done).
	as.Save(ctx, keyCopilotGitHub, "gho_test_github_token")

	p, err := mgr.LoadProvider(ctx, ProviderCopilot, "gpt-5.4")
	if err != nil {
		t.Fatalf("LoadProvider error: %v", err)
	}
	if p == nil {
		t.Fatal("LoadProvider returned nil")
	}
}

func TestLoginEmptyKey(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, _, _ := testManager(t, WithCallbacks(Callbacks{
		OnPromptKey: func(name string) (string, error) {
			return "", nil // empty key
		},
	}))

	err := mgr.Login(ctx, ProviderOpenAI)
	if err == nil {
		t.Fatal("expected error for empty key")
	}
}

func TestLoginMissingCallback(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, _, _ := testManager(t) // no callbacks

	if err := mgr.Login(ctx, ProviderOpenAI); err == nil {
		t.Error("expected error without OnPromptKey callback")
	}
	if err := mgr.Login(ctx, ProviderOllama); err == nil {
		t.Error("expected error without OnPromptURL callback")
	}
}

func TestLoginUnknownProvider(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, _, _ := testManager(t)
	err := mgr.Login(ctx, "nonexistent")
	if err == nil {
		t.Fatal("expected error for unknown provider")
	}
}

func TestLogoutOpenAI(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, as, _ := testManager(t)
	as.Save(ctx, keyOpenAI, "sk-to-delete")

	if err := mgr.Logout(ctx, ProviderOpenAI); err != nil {
		t.Fatalf("Logout error: %v", err)
	}

	_, err := as.Load(ctx, keyOpenAI)
	if err == nil {
		t.Error("key should be deleted after logout")
	}
}

func TestLogoutGemini(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, as, _ := testManager(t)
	as.Save(ctx, keyGemini, "AIza-to-delete")

	if err := mgr.Logout(ctx, ProviderGemini); err != nil {
		t.Fatalf("Logout error: %v", err)
	}

	_, err := as.Load(ctx, keyGemini)
	if err == nil {
		t.Error("key should be deleted after logout")
	}
}

func TestLogoutCopilot(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, as, _ := testManager(t)
	as.Save(ctx, keyCopilotGitHub, "gho_copilot_token")

	if err := mgr.Logout(ctx, ProviderCopilot); err != nil {
		t.Fatalf("Logout error: %v", err)
	}

	_, err := as.Load(ctx, keyCopilotGitHub)
	if err == nil {
		t.Error("copilot token should be deleted after logout")
	}
}

func TestLogoutOllama(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, _, cs := testManager(t)

	// Set up ollama config first.
	cs.Save(&config.Config{
		Providers: map[string]config.ProviderSettings{
			"ollama": {BaseURL: "http://localhost:11434"},
		},
	})

	if err := mgr.Logout(ctx, ProviderOllama); err != nil {
		t.Fatalf("Logout error: %v", err)
	}

	cfg, _ := cs.Load()
	if _, exists := cfg.Providers["ollama"]; exists {
		t.Error("ollama settings should be removed after logout")
	}
}

func TestStatus(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, as, cs := testManager(t)

	// Initially nothing is authenticated.
	statuses, err := mgr.Status(ctx)
	if err != nil {
		t.Fatalf("Status error: %v", err)
	}
	for _, s := range statuses {
		if s.Authenticated {
			t.Errorf("%s should not be authenticated", s.Name)
		}
		if s.IsDefault {
			t.Errorf("%s should not be default", s.Name)
		}
	}

	// Authenticate openai and set as default.
	as.Save(ctx, keyOpenAI, "sk-key")
	cs.Save(&config.Config{DefaultProvider: "openai"})

	statuses, err = mgr.Status(ctx)
	if err != nil {
		t.Fatalf("Status error: %v", err)
	}

	found := false
	for _, s := range statuses {
		if s.Name == "openai" {
			found = true
			if !s.Authenticated {
				t.Error("openai should be authenticated")
			}
			if !s.IsDefault {
				t.Error("openai should be default")
			}
		} else if s.Authenticated {
			t.Errorf("%s should not be authenticated", s.Name)
		}
	}
	if !found {
		t.Error("openai not in status list")
	}
}

func TestStatusOllamaAuthenticated(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, _, cs := testManager(t)
	cs.Save(&config.Config{
		Providers: map[string]config.ProviderSettings{
			"ollama": {BaseURL: "http://localhost:11434"},
		},
	})

	statuses, err := mgr.Status(ctx)
	if err != nil {
		t.Fatalf("Status error: %v", err)
	}

	for _, s := range statuses {
		if s.Name == "ollama" {
			if !s.Authenticated {
				t.Error("ollama should be authenticated when base_url is set")
			}
			return
		}
	}
	t.Error("ollama not in status list")
}

func TestSetDefault(t *testing.T) {
	t.Parallel()

	mgr, _, cs := testManager(t)

	if err := mgr.SetDefault("copilot/gpt-5.4"); err != nil {
		t.Fatalf("SetDefault error: %v", err)
	}

	cfg, err := cs.Load()
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}
	if cfg.DefaultProvider != "copilot" {
		t.Errorf("DefaultProvider = %q, want %q", cfg.DefaultProvider, "copilot")
	}
	if cfg.DefaultModel != "gpt-5.4" {
		t.Errorf("DefaultModel = %q, want %q", cfg.DefaultModel, "gpt-5.4")
	}
}

func TestSetDefaultOverwrite(t *testing.T) {
	t.Parallel()

	mgr, _, cs := testManager(t)

	mgr.SetDefault("openai/gpt-5.4")
	mgr.SetDefault("ollama/llama3")

	cfg, _ := cs.Load()
	if cfg.DefaultProvider != "ollama" || cfg.DefaultModel != "llama3" {
		t.Errorf("got %s/%s, want ollama/llama3", cfg.DefaultProvider, cfg.DefaultModel)
	}
}

func TestSetDefaultInvalidFormat(t *testing.T) {
	t.Parallel()

	mgr, _, _ := testManager(t)

	cases := []string{
		"",
		"noslash",
		"/nomodel",
		"noprovider/",
		"unknown/model",
	}
	for _, tc := range cases {
		if err := mgr.SetDefault(tc); err == nil {
			t.Errorf("SetDefault(%q) should error", tc)
		}
	}
}

func TestLoadProvider_OpenAI(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	srv := modelsServer(t, "gpt-5.4", "gpt-5.4-mini")

	mgr, as, _ := testManager(t, WithHTTPClient(redirectClient(srv)))
	as.Save(ctx, keyOpenAI, "sk-test")

	p, err := mgr.LoadProvider(ctx, ProviderOpenAI, "gpt-5.4")
	if err != nil {
		t.Fatalf("LoadProvider error: %v", err)
	}
	if p == nil {
		t.Fatal("LoadProvider returned nil")
	}
}

func TestLoadProvider_OpenAI_NotAuthenticated(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, _, _ := testManager(t)
	_, err := mgr.LoadProvider(ctx, ProviderOpenAI, "gpt-5.4")
	if err == nil {
		t.Fatal("expected error when not authenticated")
	}
}

func TestLoadProvider_Gemini(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, as, _ := testManager(t)
	as.Save(ctx, keyGemini, "AIza-test")

	p, err := mgr.LoadProvider(ctx, ProviderGemini, "gemini-pro")
	if err != nil {
		t.Fatalf("LoadProvider error: %v", err)
	}
	if p == nil {
		t.Fatal("LoadProvider returned nil")
	}
}

func TestLoadProvider_Ollama(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, _, cs := testManager(t)
	cs.Save(&config.Config{
		Providers: map[string]config.ProviderSettings{
			"ollama": {BaseURL: "http://myollama:8080"},
		},
	})

	p, err := mgr.LoadProvider(ctx, ProviderOllama, "llama3")
	if err != nil {
		t.Fatalf("LoadProvider error: %v", err)
	}
	if p == nil {
		t.Fatal("LoadProvider returned nil")
	}
}

func TestLoadProvider_OllamaDefaultURL(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	// No ollama config set — should use default URL.
	mgr, _, _ := testManager(t)
	p, err := mgr.LoadProvider(ctx, ProviderOllama, "llama3")
	if err != nil {
		t.Fatalf("LoadProvider error: %v", err)
	}
	if p == nil {
		t.Fatal("LoadProvider returned nil")
	}
}

func TestLoadDefault(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, as, cs := testManager(t)
	as.Save(ctx, keyOpenAI, "sk-default-test")
	cs.Save(&config.Config{
		DefaultProvider: "openai",
		DefaultModel:    "gpt-5.4",
	})

	p, err := mgr.LoadDefault(ctx)
	if err != nil {
		t.Fatalf("LoadDefault error: %v", err)
	}
	if p == nil {
		t.Fatal("LoadDefault returned nil")
	}
}

func TestLoadDefault_NoDefault(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, _, _ := testManager(t)
	_, err := mgr.LoadDefault(ctx)
	if err == nil {
		t.Fatal("expected error when no default set")
	}
}

func TestLoadDefault_PartialDefault(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, _, cs := testManager(t)

	// Only provider, no model.
	cs.Save(&config.Config{DefaultProvider: "openai"})
	_, err := mgr.LoadDefault(ctx)
	if err == nil {
		t.Error("expected error when default model is empty")
	}

	// Only model, no provider.
	cs.Save(&config.Config{DefaultModel: "gpt-5.4"})
	_, err = mgr.LoadDefault(ctx)
	if err == nil {
		t.Error("expected error when default provider is empty")
	}
}

func TestListModels(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	srv := modelsServer(t, "gpt-5.4", "gpt-5.4-mini", "gpt-4o")
	mgr, as, _ := testManager(t, WithHTTPClient(redirectClient(srv)))

	// Must be authenticated to list models.
	as.Save(ctx, keyOpenAI, "sk-test")

	models, err := mgr.ListModels(ctx, ProviderOpenAI)
	if err != nil {
		t.Fatalf("ListModels error: %v", err)
	}

	if len(models) != 3 {
		t.Fatalf("got %d models, want 3", len(models))
	}
	expected := map[string]bool{"gpt-5.4": true, "gpt-5.4-mini": true, "gpt-4o": true}
	for _, m := range models {
		if !expected[m] {
			t.Errorf("unexpected model %q", m)
		}
	}
}

func TestListAllModels(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	srv := modelsServer(t, "model-a", "model-b")
	mgr, as, cs := testManager(t, WithHTTPClient(redirectClient(srv)))

	// Authenticate two providers.
	as.Save(ctx, keyOpenAI, "sk-test")
	as.Save(ctx, keyGemini, "AIza-test")

	// Set ollama config so it's also "authenticated".
	cs.Save(&config.Config{
		Providers: map[string]config.ProviderSettings{
			"ollama": {BaseURL: "http://localhost:11434"},
		},
	})

	allModels, err := mgr.ListAllModels(ctx)
	if err != nil {
		t.Fatalf("ListAllModels error: %v", err)
	}

	// Should have models from openai, gemini, and ollama (3 providers × 2 models).
	if len(allModels) < 4 {
		t.Fatalf("got %d models, want at least 4 (from multiple providers)", len(allModels))
	}

	// Verify sorted by provider then model.
	for i := 1; i < len(allModels); i++ {
		prev := allModels[i-1]
		curr := allModels[i]
		if prev.Provider > curr.Provider {
			t.Errorf("not sorted: %s came before %s", prev, curr)
		} else if prev.Provider == curr.Provider && prev.Model > curr.Model {
			t.Errorf("not sorted within provider: %s came before %s", prev, curr)
		}
	}

	// Verify ModelInfo.String() format.
	for _, m := range allModels {
		s := m.String()
		if s != m.Provider+"/"+m.Model {
			t.Errorf("String() = %q, want %q", s, m.Provider+"/"+m.Model)
		}
	}
}

func TestListAllModels_NoAuthenticated(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, _, _ := testManager(t)
	models, err := mgr.ListAllModels(ctx)
	if err != nil {
		t.Fatalf("ListAllModels error: %v", err)
	}
	if len(models) != 0 {
		t.Errorf("expected 0 models, got %d", len(models))
	}
}

func TestLoginLogoutStatusRoundTrip(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, _, _ := testManager(t, WithCallbacks(Callbacks{
		OnPromptKey: func(name string) (string, error) {
			return "test-key-" + name, nil
		},
		OnPromptURL: func(name, defaultURL string) (string, error) {
			return defaultURL, nil
		},
	}))

	// Login openai.
	if err := mgr.Login(ctx, ProviderOpenAI); err != nil {
		t.Fatalf("Login openai: %v", err)
	}

	// Login ollama.
	if err := mgr.Login(ctx, ProviderOllama); err != nil {
		t.Fatalf("Login ollama: %v", err)
	}

	// Check status.
	statuses, _ := mgr.Status(ctx)
	authCount := 0
	for _, s := range statuses {
		if s.Authenticated {
			authCount++
			if s.Name != ProviderOpenAI && s.Name != ProviderOllama {
				t.Errorf("unexpected authenticated provider: %s", s.Name)
			}
		}
	}
	if authCount != 2 {
		t.Errorf("expected 2 authenticated, got %d", authCount)
	}

	// Logout openai.
	if err := mgr.Logout(ctx, ProviderOpenAI); err != nil {
		t.Fatalf("Logout openai: %v", err)
	}

	// Check status again.
	statuses, _ = mgr.Status(ctx)
	authCount = 0
	for _, s := range statuses {
		if s.Authenticated {
			authCount++
			if s.Name != ProviderOllama {
				t.Errorf("unexpected authenticated provider after logout: %s", s.Name)
			}
		}
	}
	if authCount != 1 {
		t.Errorf("expected 1 authenticated after logout, got %d", authCount)
	}
}

func TestSetDefaultAndLoadDefault(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, as, _ := testManager(t)

	// Authenticate and set default.
	as.Save(ctx, keyGemini, "AIza-key")
	if err := mgr.SetDefault("gemini/gemini-2.0-flash"); err != nil {
		t.Fatalf("SetDefault error: %v", err)
	}

	p, err := mgr.LoadDefault(ctx)
	if err != nil {
		t.Fatalf("LoadDefault error: %v", err)
	}
	if p == nil {
		t.Fatal("LoadDefault returned nil provider")
	}
}

// modelEntry mirrors the Copilot /models response shape including
// supported_endpoints, capabilities.type, and model_picker_enabled,
// used by the endpoint routing and filter tests below.
type modelEntry struct {
	ID                 string   `json:"id"`
	SupportedEndpoints []string `json:"supported_endpoints,omitempty"`
	ModelPickerEnabled *bool    `json:"model_picker_enabled,omitempty"`
	Capabilities       struct {
		Type string `json:"type,omitempty"`
	} `json:"capabilities,omitempty"`
}

// richModelsServer responds to /models with a richer payload that
// mirrors what Copilot actually returns, including supported_endpoints
// and capabilities.type so the routing logic can be exercised end to
// end. Path is exact-matched on /models so it works for any provider's
// BaseURL.
func richModelsServer(t *testing.T, entries ...modelEntry) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	handler := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"data": entries})
	}
	mux.HandleFunc("/models", handler)
	mux.HandleFunc("/v1/models", handler)
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func TestEndpointForMeta(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		meta provider.ModelMeta
		want string
	}{
		{
			name: "no metadata defaults to chat completions",
			meta: provider.ModelMeta{ID: "gpt-4o"},
			want: provider.EndpointChatCompletions,
		},
		{
			name: "explicit chat completions",
			meta: provider.ModelMeta{
				ID:                 "claude-sonnet-4.5",
				SupportedEndpoints: []string{"/chat/completions", "/v1/messages"},
			},
			want: provider.EndpointChatCompletions,
		},
		{
			name: "both endpoints prefer chat completions for compat",
			meta: provider.ModelMeta{
				ID:                 "gpt-5.4",
				SupportedEndpoints: []string{"/chat/completions", "/responses"},
			},
			want: provider.EndpointChatCompletions,
		},
		{
			name: "responses-only routes to /responses",
			meta: provider.ModelMeta{
				ID:                 "gpt-5.4-mini",
				SupportedEndpoints: []string{"/responses", "ws:/responses"},
			},
			want: provider.EndpointResponses,
		},
		{
			name: "unknown endpoint set falls back to chat completions",
			meta: provider.ModelMeta{
				ID:                 "weird",
				SupportedEndpoints: []string{"/v1/messages"},
			},
			want: provider.EndpointChatCompletions,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := endpointForMeta(tt.meta); got != tt.want {
				t.Errorf("endpointForMeta() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestListAllModels_PopulatesEndpoint(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	srv := richModelsServer(t,
		modelEntry{ID: "gpt-5.4", SupportedEndpoints: []string{"/chat/completions", "/responses"}},
		modelEntry{ID: "gpt-5.4-mini", SupportedEndpoints: []string{"/responses"}},
		modelEntry{ID: "gpt-4o"}, // no metadata, legacy
		func() modelEntry {
			e := modelEntry{ID: "text-embedding-3-small"}
			e.Capabilities.Type = "embeddings"
			return e
		}(),
	)

	mgr, as, _ := testManager(t, WithHTTPClient(redirectClient(srv)))
	as.Save(ctx, keyOpenAI, "sk-test")

	all, err := mgr.ListAllModels(ctx)
	if err != nil {
		t.Fatalf("ListAllModels: %v", err)
	}

	// Embeddings should be filtered out.
	byModel := make(map[string]ModelInfo)
	for _, info := range all {
		byModel[info.Model] = info
	}
	if _, ok := byModel["text-embedding-3-small"]; ok {
		t.Error("embeddings model should be filtered out of ListAllModels")
	}

	// Endpoint metadata should be populated correctly.
	cases := []struct {
		model string
		want  string
	}{
		{"gpt-5.4", provider.EndpointChatCompletions},
		{"gpt-5.4-mini", provider.EndpointResponses},
		{"gpt-4o", provider.EndpointChatCompletions},
	}
	for _, c := range cases {
		got, ok := byModel[c.model]
		if !ok {
			t.Errorf("model %q missing from ListAllModels", c.model)
			continue
		}
		if got.Endpoint != c.want {
			t.Errorf("ListAllModels()[%q].Endpoint = %q, want %q", c.model, got.Endpoint, c.want)
		}
	}
}

func TestLoadProviderForModel_RoutesResponses(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	// Capture the request path to verify routing.
	var gotPath string
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "text/event-stream")
		// Send a minimal /responses-shaped stream that terminates cleanly.
		_, _ = w.Write([]byte("event: response.completed\ndata: {\"response\":{\"status\":\"completed\"}}\n\n"))
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	mgr, as, _ := testManager(t, WithHTTPClient(redirectClient(srv)))
	as.Save(ctx, keyOpenAI, "sk-test")

	info := ModelInfo{
		Provider: ProviderOpenAI,
		Model:    "gpt-5.4-mini",
		Endpoint: provider.EndpointResponses,
	}
	p, err := mgr.LoadProviderForModel(ctx, info)
	if err != nil {
		t.Fatalf("LoadProviderForModel: %v", err)
	}

	// Drive a Complete call so we can observe the routed URL path.
	events, err := p.Complete(ctx, provider.Request{Stream: true})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	for range events {
	}

	if !strings.HasSuffix(gotPath, "/responses") {
		t.Errorf("request path = %q, want suffix /responses", gotPath)
	}
}

func TestSetDefaultModelAndLoadDefault_PreservesEndpoint(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	// Capture the request path on LoadDefault → Complete to verify
	// that the persisted endpoint round-trips through config.
	var gotPath string
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: response.completed\ndata: {\"response\":{\"status\":\"completed\"}}\n\n"))
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	mgr, as, _ := testManager(t, WithHTTPClient(redirectClient(srv)))
	as.Save(ctx, keyOpenAI, "sk-test")

	if err := mgr.SetDefaultModel(ModelInfo{
		Provider: ProviderOpenAI,
		Model:    "gpt-5.4-mini",
		Endpoint: provider.EndpointResponses,
	}); err != nil {
		t.Fatalf("SetDefaultModel: %v", err)
	}

	p, err := mgr.LoadDefault(ctx)
	if err != nil {
		t.Fatalf("LoadDefault: %v", err)
	}

	events, err := p.Complete(ctx, provider.Request{Stream: true})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	for range events {
	}

	if !strings.HasSuffix(gotPath, "/responses") {
		t.Errorf("LoadDefault did not preserve endpoint: path = %q, want suffix /responses", gotPath)
	}
}

func TestSetDefault_ClearsEndpoint(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, as, cs := testManager(t)
	as.Save(ctx, keyOpenAI, "sk-test")

	// Pre-populate config with a non-default endpoint to make sure
	// SetDefault explicitly clears it.
	cs.Save(&config.Config{
		DefaultProvider: "openai",
		DefaultModel:    "gpt-5.4-mini",
		DefaultEndpoint: provider.EndpointResponses,
	})

	if err := mgr.SetDefault("openai/gpt-5.4"); err != nil {
		t.Fatalf("SetDefault: %v", err)
	}

	cfg, _ := cs.Load()
	if cfg.DefaultEndpoint != "" {
		t.Errorf("DefaultEndpoint = %q after SetDefault, want empty", cfg.DefaultEndpoint)
	}
	if cfg.DefaultModel != "gpt-5.4" {
		t.Errorf("DefaultModel = %q, want gpt-5.4", cfg.DefaultModel)
	}
}

func TestListAllModels_FiltersEmbeddings(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	srv := richModelsServer(t,
		modelEntry{ID: "gpt-4o"},
		func() modelEntry {
			e := modelEntry{ID: "text-embedding-3-small"}
			e.Capabilities.Type = "embeddings"
			return e
		}(),
	)

	mgr, as, _ := testManager(t, WithHTTPClient(redirectClient(srv)))
	as.Save(ctx, keyOpenAI, "sk-test")

	models, err := mgr.ListModels(ctx, ProviderOpenAI)
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	for _, m := range models {
		if m == "text-embedding-3-small" {
			t.Error("embedding model should be filtered out of ListModels")
		}
	}
	if len(models) != 1 || models[0] != "gpt-4o" {
		t.Errorf("ListModels = %v, want [gpt-4o]", models)
	}
}

// TestListModels_FiltersDisabledPickerModels verifies the picker
// excludes upstream entries marked model_picker_enabled=false. This
// is the field GitHub Copilot's own UI uses to hide routers, pinned
// legacy versions, and internal load-test models — surfacing them in
// our /models picker creates noise (e.g. "copilot/accounts/msft/
// routers/...") that the user can't actually use as chat models.
func TestListModels_FiltersDisabledPickerModels(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	enabled := true
	disabled := false
	srv := richModelsServer(t,
		modelEntry{ID: "gpt-4o", ModelPickerEnabled: &enabled},
		modelEntry{ID: "claude-opus-4.6", ModelPickerEnabled: &enabled},
		// Upstream-disabled router — must be filtered out.
		modelEntry{ID: "accounts/msft/routers/abcd1234", ModelPickerEnabled: &disabled},
		// Pinned legacy version — must be filtered out.
		modelEntry{ID: "gpt-4o-2024-08-06", ModelPickerEnabled: &disabled},
		// Field absent — treat as enabled (matches OpenAI/Gemini/Ollama).
		modelEntry{ID: "gpt-4-no-meta"},
	)

	mgr, as, _ := testManager(t, WithHTTPClient(redirectClient(srv)))
	as.Save(ctx, keyOpenAI, "sk-test")

	models, err := mgr.ListModels(ctx, ProviderOpenAI)
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}

	got := make(map[string]bool, len(models))
	for _, m := range models {
		got[m] = true
	}

	wantPresent := []string{"gpt-4o", "claude-opus-4.6", "gpt-4-no-meta"}
	for _, want := range wantPresent {
		if !got[want] {
			t.Errorf("expected %q in ListModels, got %v", want, models)
		}
	}

	wantAbsent := []string{"accounts/msft/routers/abcd1234", "gpt-4o-2024-08-06"}
	for _, dont := range wantAbsent {
		if got[dont] {
			t.Errorf("expected %q to be filtered out, got %v", dont, models)
		}
	}
	if len(models) != len(wantPresent) {
		t.Errorf("ListModels returned %d models (%v), want %d", len(models), models, len(wantPresent))
	}
}

// TestRecentModels_RoundTrip verifies TrackRecentModel persists the
// selection and RecentModels returns it most-recent-first. The
// Endpoint must round-trip so a /responses-only model retains its
// routing on the next session.
func TestRecentModels_RoundTrip(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, _, _ := testManager(t)

	// Empty initially.
	recents, err := mgr.RecentModels(ctx)
	if err != nil {
		t.Fatalf("RecentModels (empty): %v", err)
	}
	if len(recents) != 0 {
		t.Errorf("expected 0 initial recents, got %d", len(recents))
	}

	// Track three models in order.
	tracks := []ModelInfo{
		{Provider: "openai", Model: "gpt-4o"},
		{Provider: "copilot", Model: "claude-opus-4.6"},
		{Provider: "openai", Model: "gpt-5.4-mini", Endpoint: provider.EndpointResponses},
	}
	for _, info := range tracks {
		if err := mgr.TrackRecentModel(ctx, info); err != nil {
			t.Fatalf("TrackRecentModel(%s): %v", info, err)
		}
	}

	recents, err = mgr.RecentModels(ctx)
	if err != nil {
		t.Fatalf("RecentModels: %v", err)
	}

	// Most-recent-first order: gpt-5.4-mini, claude-opus-4.6, gpt-4o.
	// RecentModels backfills ContextWindow from provider's hardcoded
	// table so the UI can show context usage for recents without a
	// live /models query — mirror that in the expected values by
	// calling the same function rather than hardcoding token counts.
	wantOrder := []ModelInfo{
		{Provider: "openai", Model: "gpt-5.4-mini", Endpoint: provider.EndpointResponses, ContextWindow: provider.ContextWindow("gpt-5.4-mini")},
		{Provider: "copilot", Model: "claude-opus-4.6", ContextWindow: provider.ContextWindow("claude-opus-4.6")},
		{Provider: "openai", Model: "gpt-4o", ContextWindow: provider.ContextWindow("gpt-4o")},
	}
	if len(recents) != len(wantOrder) {
		t.Fatalf("got %d recents, want %d (%v)", len(recents), len(wantOrder), recents)
	}
	for i, want := range wantOrder {
		if recents[i] != want {
			t.Errorf("recents[%d] = %+v, want %+v", i, recents[i], want)
		}
	}
}

// TestTrackRecentModel_DedupesAndUnbounded verifies that re-selecting
// an existing model promotes it to the front (rather than inserting a
// duplicate) and that the list grows without an upper bound — every
// model the user has ever picked is retained in MRU order.
func TestTrackRecentModel_DedupesAndUnbounded(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, _, _ := testManager(t)

	// Push 12 unique models — far more than the prior 5-entry cap.
	const n = 12
	for i := 0; i < n; i++ {
		if err := mgr.TrackRecentModel(ctx, ModelInfo{
			Provider: "openai",
			Model:    fmt.Sprintf("model-%d", i),
		}); err != nil {
			t.Fatalf("TrackRecentModel: %v", err)
		}
	}

	recents, _ := mgr.RecentModels(ctx)
	if len(recents) != n {
		t.Fatalf("got %d recents, want all %d (no cap)", len(recents), n)
	}
	// Most-recent-first: model-11 .. model-0.
	for i, info := range recents {
		want := fmt.Sprintf("model-%d", n-1-i)
		if info.Model != want {
			t.Errorf("recents[%d].Model = %q, want %q", i, info.Model, want)
		}
	}

	// Re-tracking an existing model should promote it to the front
	// without inserting a duplicate or changing the total count.
	if err := mgr.TrackRecentModel(ctx, ModelInfo{Provider: "openai", Model: "model-3"}); err != nil {
		t.Fatalf("TrackRecentModel re-add: %v", err)
	}
	recents, _ = mgr.RecentModels(ctx)
	if len(recents) != n {
		t.Errorf("after re-track: got %d recents, want %d", len(recents), n)
	}
	if recents[0].Model != "model-3" {
		t.Errorf("re-tracked model not at front: got %q", recents[0].Model)
	}
	seen := make(map[string]bool)
	for _, r := range recents {
		key := r.Provider + "/" + r.Model
		if seen[key] {
			t.Errorf("duplicate entry %q after re-track", key)
		}
		seen[key] = true
	}

	// Re-tracking with a new endpoint should overwrite the prior endpoint.
	if err := mgr.TrackRecentModel(ctx, ModelInfo{
		Provider: "openai",
		Model:    "model-3",
		Endpoint: provider.EndpointResponses,
	}); err != nil {
		t.Fatalf("TrackRecentModel update endpoint: %v", err)
	}
	recents, _ = mgr.RecentModels(ctx)
	if recents[0].Endpoint != provider.EndpointResponses {
		t.Errorf("updated endpoint not preserved: got %q", recents[0].Endpoint)
	}
}

// TestTrackRecentModel_Validation verifies empty provider/model is rejected.
func TestTrackRecentModel_Validation(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, _, _ := testManager(t)

	if err := mgr.TrackRecentModel(ctx, ModelInfo{Model: "x"}); err == nil {
		t.Error("expected error for empty provider")
	}
	if err := mgr.TrackRecentModel(ctx, ModelInfo{Provider: "openai"}); err == nil {
		t.Error("expected error for empty model")
	}
}
