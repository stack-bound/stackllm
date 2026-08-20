package profile

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stack-bound/stackllm/auth"
	"github.com/stack-bound/stackllm/config"
)

// failingAuthStore wraps a MemoryStore but fails Save/Delete for
// selected keys, letting tests reach specific store-error branches.
type failingAuthStore struct {
	inner      *auth.MemoryStore
	failSave   map[string]bool
	failDelete map[string]bool
}

func newFailingAuthStore() *failingAuthStore {
	return &failingAuthStore{
		inner:      auth.NewMemoryStore(),
		failSave:   map[string]bool{},
		failDelete: map[string]bool{},
	}
}

func (f *failingAuthStore) Load(ctx context.Context, key string) (string, error) {
	return f.inner.Load(ctx, key)
}

func (f *failingAuthStore) Save(ctx context.Context, key, value string) error {
	if f.failSave[key] {
		return fmt.Errorf("failing store: save %s refused", key)
	}
	return f.inner.Save(ctx, key, value)
}

func (f *failingAuthStore) Delete(ctx context.Context, key string) error {
	if f.failDelete[key] {
		return fmt.Errorf("failing store: delete %s refused", key)
	}
	return f.inner.Delete(ctx, key)
}

// corruptConfigManager returns a manager whose config file contains
// invalid JSON, so every configStore.Load call fails.
func corruptConfigManager(t *testing.T) *Manager {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte("{corrupt"), 0o600); err != nil {
		t.Fatalf("write corrupt config: %v", err)
	}
	return New(WithAuthStore(auth.NewMemoryStore()), WithConfigStore(&config.Store{Path: path}))
}

// unsavableConfigManager returns a manager whose config can be read
// but not written (the atomic-write temp path is occupied by a
// directory).
func unsavableConfigManager(t *testing.T) *Manager {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.Mkdir(path+".tmp", 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	return New(WithAuthStore(auth.NewMemoryStore()), WithConfigStore(&config.Store{Path: path}))
}

// --- codex device flow ---

// codexDeviceServer fakes the three endpoints the codex device flow
// hits: usercode issue, poll, and token exchange.
func codexDeviceServer(t *testing.T) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/api/accounts/deviceauth/usercode", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"device_auth_id": "dev-auth-1",
			"user_code":      "CODEX-99",
			"interval":       0,
			"expires_in":     60,
		})
	})
	mux.HandleFunc("/api/accounts/deviceauth/token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"authorization_code": "auth-code-1",
			"code_verifier":      "verifier-1",
		})
	})
	mux.HandleFunc("/oauth/token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "codex-access",
			"refresh_token": "codex-refresh",
			"expires_in":    3600,
		})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func TestLoginOpenAICodexDevice_EndToEnd(t *testing.T) {
	t.Parallel()

	srv := codexDeviceServer(t)

	var gotUserCode, gotVerifyURL string
	var successCalls atomic.Int32

	as := auth.NewMemoryStore()
	cs := &config.Store{Path: filepath.Join(t.TempDir(), "config.json")}
	mgr := New(
		WithAuthStore(as),
		WithConfigStore(cs),
		WithHTTPClient(redirectClient(srv)),
		WithPollInterval(time.Millisecond),
		WithCallbacks(Callbacks{
			OnDeviceCode: func(code, verifyURL string) {
				gotUserCode = code
				gotVerifyURL = verifyURL
			},
			OnSuccess: func() { successCalls.Add(1) },
		}),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := mgr.LoginOpenAICodexDevice(ctx); err != nil {
		t.Fatalf("LoginOpenAICodexDevice: %v", err)
	}

	if gotUserCode != "CODEX-99" {
		t.Errorf("OnDeviceCode user code = %q, want CODEX-99", gotUserCode)
	}
	if gotVerifyURL == "" {
		t.Error("OnDeviceCode verify URL should be populated")
	}
	if successCalls.Load() != 1 {
		t.Errorf("OnSuccess calls = %d, want 1", successCalls.Load())
	}

	// The persisted record must round-trip with the token AND its expiry.
	raw, err := as.Load(ctx, auth.CodexStoreKey)
	if err != nil {
		t.Fatalf("codex record not persisted: %v", err)
	}
	var rec struct {
		AccessToken  string    `json:"access_token"`
		RefreshToken string    `json:"refresh_token"`
		ExpiresAt    time.Time `json:"expires_at"`
	}
	if err := json.Unmarshal([]byte(raw), &rec); err != nil {
		t.Fatalf("decode record: %v", err)
	}
	if rec.AccessToken != "codex-access" {
		t.Errorf("access token = %q, want codex-access", rec.AccessToken)
	}
	if rec.RefreshToken != "codex-refresh" {
		t.Errorf("refresh token = %q, want codex-refresh", rec.RefreshToken)
	}
	if rec.ExpiresAt.IsZero() || !rec.ExpiresAt.After(time.Now()) {
		t.Errorf("expires_at = %v, want a future expiry honouring expires_in", rec.ExpiresAt)
	}

	// A codex record counts as OpenAI authentication.
	statuses, err := mgr.Status(ctx)
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	for _, s := range statuses {
		if s.Name == ProviderOpenAI && !s.Authenticated {
			t.Error("openai should report authenticated after codex login")
		}
	}
}

func TestBeginOpenAICodexDeviceLogin_WaitCompletes(t *testing.T) {
	t.Parallel()

	srv := codexDeviceServer(t)
	as := auth.NewMemoryStore()
	mgr := New(
		WithAuthStore(as),
		WithConfigStore(&config.Store{Path: filepath.Join(t.TempDir(), "config.json")}),
		WithHTTPClient(redirectClient(srv)),
		WithPollInterval(time.Millisecond),
	)

	// context.Background() has no deadline, exercising runDeviceFlow's
	// internal wait timeout branch.
	flow, err := mgr.BeginOpenAICodexDeviceLogin(context.Background())
	if err != nil {
		t.Fatalf("BeginOpenAICodexDeviceLogin: %v", err)
	}
	if flow.UserCode() != "CODEX-99" {
		t.Errorf("UserCode = %q, want CODEX-99", flow.UserCode())
	}

	waitCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := flow.Wait(waitCtx); err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if flow.State() != DeviceFlowAuthenticated {
		t.Fatalf("state after Wait = %q, err = %v", flow.State(), flow.Err())
	}
	if _, err := as.Load(context.Background(), auth.CodexStoreKey); err != nil {
		t.Errorf("codex record not persisted after background flow: %v", err)
	}

	// Wait on a finished flow returns immediately.
	if err := flow.Wait(context.Background()); err != nil {
		t.Errorf("second Wait = %v, want nil", err)
	}
}

func TestDeviceFlow_WaitHonoursContextCancellation(t *testing.T) {
	t.Parallel()

	// Poll endpoint never authorises, so the flow stays pending.
	mux := http.NewServeMux()
	mux.HandleFunc("/login/device/code", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"device_code":      "dev-code",
			"user_code":        "WAIT-1",
			"verification_uri": "https://github.com/login/device",
			"interval":         0,
			"expires_in":       300,
		})
	})
	mux.HandleFunc("/login/oauth/access_token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"error": "authorization_pending"})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	mgr, _, _ := testManager(t,
		WithHTTPClient(redirectClient(srv)),
		WithPollInterval(5*time.Millisecond),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	flow, err := mgr.BeginCopilotLogin(ctx)
	if err != nil {
		t.Fatalf("BeginCopilotLogin: %v", err)
	}

	cancelled, cancelNow := context.WithCancel(context.Background())
	cancelNow()
	if err := flow.Wait(cancelled); err != context.Canceled {
		t.Errorf("Wait with cancelled ctx = %v, want context.Canceled", err)
	}

	// Clean up the background goroutine.
	flow.Cancel()
	cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cleanupCancel()
	if err := flow.Wait(cleanupCtx); err != nil {
		t.Errorf("Wait after Cancel = %v", err)
	}
}

func TestLoginOpenAICodexWeb_EndToEnd(t *testing.T) {
	t.Parallel()

	// The codex web flow binds the fixed whitelisted port 1455. Skip
	// when the environment already occupies it.
	probe, err := net.Listen("tcp", ":1455")
	if err != nil {
		t.Skipf("port 1455 unavailable: %v", err)
	}
	probe.Close()

	mux := http.NewServeMux()
	mux.HandleFunc("/oauth/token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "codex-web-access",
			"refresh_token": "codex-web-refresh",
			"expires_in":    1800,
		})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	var successCalls atomic.Int32
	as := auth.NewMemoryStore()
	mgr := New(
		WithAuthStore(as),
		WithConfigStore(&config.Store{Path: filepath.Join(t.TempDir(), "config.json")}),
		WithHTTPClient(redirectClient(srv)),
		WithCallbacks(Callbacks{
			OnOpenURL: func(authURL string) {
				// Simulate the user completing authorisation in the
				// browser: extract the state and hit the local callback.
				u, err := url.Parse(authURL)
				if err != nil {
					t.Errorf("bad auth URL %q: %v", authURL, err)
					return
				}
				state := u.Query().Get("state")
				go func() {
					cb := "http://localhost:1455/auth/callback?state=" + url.QueryEscape(state) + "&code=web-auth-code"
					resp, err := http.Get(cb)
					if err != nil {
						t.Errorf("callback GET: %v", err)
						return
					}
					resp.Body.Close()
				}()
			},
			OnSuccess: func() { successCalls.Add(1) },
		}),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := mgr.LoginOpenAICodexWeb(ctx); err != nil {
		t.Fatalf("LoginOpenAICodexWeb: %v", err)
	}
	if successCalls.Load() != 1 {
		t.Errorf("OnSuccess calls = %d, want 1", successCalls.Load())
	}

	raw, err := as.Load(ctx, auth.CodexStoreKey)
	if err != nil {
		t.Fatalf("codex record not persisted: %v", err)
	}
	var rec struct {
		AccessToken string    `json:"access_token"`
		ExpiresAt   time.Time `json:"expires_at"`
	}
	if err := json.Unmarshal([]byte(raw), &rec); err != nil {
		t.Fatalf("decode record: %v", err)
	}
	if rec.AccessToken != "codex-web-access" {
		t.Errorf("access token = %q, want codex-web-access", rec.AccessToken)
	}
	if rec.ExpiresAt.IsZero() || !rec.ExpiresAt.After(time.Now()) {
		t.Errorf("expires_at = %v, want future expiry", rec.ExpiresAt)
	}
}

// --- error branches across profile.go ---

func TestLoginAPIKey_PromptError(t *testing.T) {
	t.Parallel()
	mgr, _, _ := testManager(t, WithCallbacks(Callbacks{
		OnPromptKey: func(string) (string, error) { return "", fmt.Errorf("user aborted") },
	}))
	err := mgr.Login(context.Background(), ProviderOpenAI)
	if err == nil || !strings.Contains(err.Error(), "user aborted") {
		t.Errorf("error = %v, want prompt error", err)
	}
}

func TestLoginAPIKey_SaveError(t *testing.T) {
	t.Parallel()
	fs := newFailingAuthStore()
	fs.failSave[keyOpenAI] = true
	mgr := New(
		WithAuthStore(fs),
		WithConfigStore(&config.Store{Path: filepath.Join(t.TempDir(), "config.json")}),
		WithCallbacks(Callbacks{OnPromptKey: func(string) (string, error) { return "sk-x", nil }}),
	)
	err := mgr.Login(context.Background(), ProviderOpenAI)
	if err == nil || !strings.Contains(err.Error(), "save openai key") {
		t.Errorf("error = %v, want save error", err)
	}
}

func TestLoginOllama_ErrorBranches(t *testing.T) {
	t.Parallel()

	t.Run("prompt error", func(t *testing.T) {
		t.Parallel()
		mgr, _, _ := testManager(t, WithCallbacks(Callbacks{
			OnPromptURL: func(string, string) (string, error) { return "", fmt.Errorf("no url") },
		}))
		if err := mgr.Login(context.Background(), ProviderOllama); err == nil || !strings.Contains(err.Error(), "no url") {
			t.Errorf("error = %v, want prompt error", err)
		}
	})

	t.Run("config load error", func(t *testing.T) {
		t.Parallel()
		mgr := corruptConfigManager(t)
		mgr.callbacks = Callbacks{OnPromptURL: func(string, string) (string, error) { return "http://x", nil }}
		if err := mgr.Login(context.Background(), ProviderOllama); err == nil || !strings.Contains(err.Error(), "load config") {
			t.Errorf("error = %v, want load config error", err)
		}
	})

	t.Run("config save error", func(t *testing.T) {
		t.Parallel()
		mgr := unsavableConfigManager(t)
		mgr.callbacks = Callbacks{OnPromptURL: func(string, string) (string, error) { return "http://x", nil }}
		if err := mgr.Login(context.Background(), ProviderOllama); err == nil || !strings.Contains(err.Error(), "save ollama config") {
			t.Errorf("error = %v, want save error", err)
		}
	})
}

func TestLogout_ErrorBranches(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	t.Run("unknown provider", func(t *testing.T) {
		t.Parallel()
		mgr, _, _ := testManager(t)
		if err := mgr.Logout(ctx, "nope"); err == nil || !strings.Contains(err.Error(), "unknown provider") {
			t.Errorf("error = %v, want unknown provider", err)
		}
	})

	t.Run("openai api key delete fails", func(t *testing.T) {
		t.Parallel()
		fs := newFailingAuthStore()
		fs.failDelete[keyOpenAI] = true
		mgr := New(WithAuthStore(fs), WithConfigStore(&config.Store{Path: filepath.Join(t.TempDir(), "c.json")}))
		if err := mgr.Logout(ctx, ProviderOpenAI); err == nil {
			t.Error("expected error when api-key delete fails")
		}
	})

	t.Run("openai device record delete fails", func(t *testing.T) {
		t.Parallel()
		fs := newFailingAuthStore()
		fs.failDelete["openai_token"] = true
		mgr := New(WithAuthStore(fs), WithConfigStore(&config.Store{Path: filepath.Join(t.TempDir(), "c.json")}))
		if err := mgr.Logout(ctx, ProviderOpenAI); err == nil {
			t.Error("expected error when device record delete fails")
		}
	})

	t.Run("openai web record delete fails", func(t *testing.T) {
		t.Parallel()
		fs := newFailingAuthStore()
		fs.failDelete["openai_web_token"] = true
		mgr := New(WithAuthStore(fs), WithConfigStore(&config.Store{Path: filepath.Join(t.TempDir(), "c.json")}))
		if err := mgr.Logout(ctx, ProviderOpenAI); err == nil {
			t.Error("expected error when web record delete fails")
		}
	})

	t.Run("ollama config load fails", func(t *testing.T) {
		t.Parallel()
		mgr := corruptConfigManager(t)
		if err := mgr.Logout(ctx, ProviderOllama); err == nil || !strings.Contains(err.Error(), "load config") {
			t.Errorf("error = %v, want load config error", err)
		}
	})
}

func TestStatus_ConfigLoadError(t *testing.T) {
	t.Parallel()
	mgr := corruptConfigManager(t)
	if _, err := mgr.Status(context.Background()); err == nil || !strings.Contains(err.Error(), "load config") {
		t.Errorf("error = %v, want load config error", err)
	}
	// ListAllModels goes through Status and must propagate the error.
	if _, err := mgr.ListAllModels(context.Background()); err == nil {
		t.Error("ListAllModels should propagate Status error")
	}
}

func TestListModels_NotAuthenticated(t *testing.T) {
	t.Parallel()
	mgr, _, _ := testManager(t)
	_, err := mgr.ListModels(context.Background(), ProviderGemini)
	if err == nil || !strings.Contains(err.Error(), "not authenticated") {
		t.Errorf("error = %v, want not-authenticated error", err)
	}
}

func TestRecentModels_Branches(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	t.Run("config load error", func(t *testing.T) {
		t.Parallel()
		mgr := corruptConfigManager(t)
		if _, err := mgr.RecentModels(ctx); err == nil {
			t.Error("expected load error")
		}
	})

	t.Run("skips malformed entries", func(t *testing.T) {
		t.Parallel()
		cs := &config.Store{Path: filepath.Join(t.TempDir(), "config.json")}
		if err := cs.Save(&config.Config{RecentModels: []config.RecentModel{
			{Provider: "", Model: "orphan"},
			{Provider: "openai", Model: ""},
			{Provider: "openai", Model: "gpt-4o"},
		}}); err != nil {
			t.Fatalf("seed config: %v", err)
		}
		mgr := New(WithAuthStore(auth.NewMemoryStore()), WithConfigStore(cs))
		got, err := mgr.RecentModels(ctx)
		if err != nil {
			t.Fatalf("RecentModels: %v", err)
		}
		if len(got) != 1 || got[0].Provider != "openai" || got[0].Model != "gpt-4o" {
			t.Errorf("RecentModels = %+v, want single valid entry", got)
		}
	})
}

func TestTrackRecentModel_ErrorBranches(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	t.Run("missing provider or model", func(t *testing.T) {
		t.Parallel()
		mgr, _, _ := testManager(t)
		if err := mgr.TrackRecentModel(ctx, ModelInfo{}); err == nil {
			t.Error("expected validation error")
		}
	})

	t.Run("config load error", func(t *testing.T) {
		t.Parallel()
		mgr := corruptConfigManager(t)
		if err := mgr.TrackRecentModel(ctx, ModelInfo{Provider: "openai", Model: "m"}); err == nil {
			t.Error("expected load error")
		}
	})

	t.Run("config save error", func(t *testing.T) {
		t.Parallel()
		mgr := unsavableConfigManager(t)
		if err := mgr.TrackRecentModel(ctx, ModelInfo{Provider: "openai", Model: "m"}); err == nil {
			t.Error("expected save error")
		}
	})
}

func TestSetDefaultModel_ErrorBranches(t *testing.T) {
	t.Parallel()

	t.Run("missing fields", func(t *testing.T) {
		t.Parallel()
		mgr, _, _ := testManager(t)
		if err := mgr.SetDefaultModel(ModelInfo{}); err == nil {
			t.Error("expected validation error")
		}
	})

	t.Run("config load error", func(t *testing.T) {
		t.Parallel()
		mgr := corruptConfigManager(t)
		if err := mgr.SetDefaultModel(ModelInfo{Provider: "openai", Model: "m"}); err == nil {
			t.Error("expected load error")
		}
	})

	t.Run("config save error", func(t *testing.T) {
		t.Parallel()
		mgr := unsavableConfigManager(t)
		if err := mgr.SetDefaultModel(ModelInfo{Provider: "openai", Model: "m"}); err == nil {
			t.Error("expected save error")
		}
	})
}

func TestLoadDefault_ConfigLoadError(t *testing.T) {
	t.Parallel()
	mgr := corruptConfigManager(t)
	if _, err := mgr.LoadDefault(context.Background()); err == nil {
		t.Error("expected load error")
	}
}

func TestBuildProvider_ErrorBranches(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	t.Run("gemini not authenticated", func(t *testing.T) {
		t.Parallel()
		mgr, _, _ := testManager(t)
		if _, err := mgr.LoadProvider(ctx, ProviderGemini, "gemini-pro"); err == nil || !strings.Contains(err.Error(), "not authenticated") {
			t.Errorf("error = %v, want not-authenticated", err)
		}
	})

	t.Run("ollama config load error", func(t *testing.T) {
		t.Parallel()
		mgr := corruptConfigManager(t)
		if _, err := mgr.LoadProvider(ctx, ProviderOllama, "llama3"); err == nil {
			t.Error("expected load error")
		}
	})

	t.Run("unknown provider", func(t *testing.T) {
		t.Parallel()
		mgr, _, _ := testManager(t)
		if _, err := mgr.LoadProvider(ctx, "mystery", "m"); err == nil || !strings.Contains(err.Error(), "unknown provider") {
			t.Errorf("error = %v, want unknown provider", err)
		}
	})
}
