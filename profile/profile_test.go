package profile

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"time"
	"net/url"
	"path/filepath"
	"testing"

	"github.com/stack-bound/stackllm/auth"
	"github.com/stack-bound/stackllm/config"
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
