package profile

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestSaveAPIKey_OpenAI(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, as, _ := testManager(t)
	if err := mgr.SaveAPIKey(ctx, ProviderOpenAI, "sk-web-key"); err != nil {
		t.Fatalf("SaveAPIKey error: %v", err)
	}

	got, err := as.Load(ctx, keyOpenAI)
	if err != nil {
		t.Fatalf("Load openai key: %v", err)
	}
	if got != "sk-web-key" {
		t.Errorf("stored key = %q, want sk-web-key", got)
	}
}

func TestSaveAPIKey_Gemini(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, as, _ := testManager(t)
	if err := mgr.SaveAPIKey(ctx, ProviderGemini, "AIza-web-key"); err != nil {
		t.Fatalf("SaveAPIKey error: %v", err)
	}

	got, err := as.Load(ctx, keyGemini)
	if err != nil {
		t.Fatalf("Load gemini key: %v", err)
	}
	if got != "AIza-web-key" {
		t.Errorf("stored key = %q, want AIza-web-key", got)
	}
}

func TestSaveAPIKey_EmptyRejected(t *testing.T) {
	t.Parallel()
	mgr, _, _ := testManager(t)
	if err := mgr.SaveAPIKey(context.Background(), ProviderOpenAI, ""); err == nil {
		t.Fatal("expected error for empty key")
	}
}

func TestSaveAPIKey_UnsupportedProvider(t *testing.T) {
	t.Parallel()
	mgr, _, _ := testManager(t)
	if err := mgr.SaveAPIKey(context.Background(), ProviderCopilot, "anything"); err == nil {
		t.Fatal("expected error for copilot (not an API-key provider)")
	}
	if err := mgr.SaveAPIKey(context.Background(), ProviderOllama, "anything"); err == nil {
		t.Fatal("expected error for ollama (not an API-key provider)")
	}
}

func TestSaveOllamaURL(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mgr, _, cs := testManager(t)
	if err := mgr.SaveOllamaURL(ctx, "http://ollama.internal:11434"); err != nil {
		t.Fatalf("SaveOllamaURL error: %v", err)
	}

	cfg, err := cs.Load()
	if err != nil {
		t.Fatalf("Load config: %v", err)
	}
	settings, ok := cfg.Providers[ProviderOllama]
	if !ok {
		t.Fatal("ollama settings missing")
	}
	if settings.BaseURL != "http://ollama.internal:11434" {
		t.Errorf("BaseURL = %q", settings.BaseURL)
	}
}

func TestSaveOllamaURL_EmptyFallsBackToDefault(t *testing.T) {
	t.Parallel()
	mgr, _, cs := testManager(t)
	if err := mgr.SaveOllamaURL(context.Background(), ""); err != nil {
		t.Fatalf("SaveOllamaURL error: %v", err)
	}
	cfg, _ := cs.Load()
	if cfg.Providers[ProviderOllama].BaseURL != "http://localhost:11434" {
		t.Errorf("BaseURL = %q, want default", cfg.Providers[ProviderOllama].BaseURL)
	}
}

func TestDefault_Unset(t *testing.T) {
	t.Parallel()
	mgr, _, _ := testManager(t)
	_, ok, err := mgr.Default(context.Background())
	if err != nil {
		t.Fatalf("Default error: %v", err)
	}
	if ok {
		t.Error("Default should report not-set for fresh config")
	}
}

func TestDefault_SetAndRead(t *testing.T) {
	t.Parallel()
	mgr, _, _ := testManager(t)
	info := ModelInfo{Provider: ProviderCopilot, Model: "gpt-5.4", Endpoint: "/responses"}
	if err := mgr.SetDefaultModel(info); err != nil {
		t.Fatalf("SetDefaultModel: %v", err)
	}
	got, ok, err := mgr.Default(context.Background())
	if err != nil {
		t.Fatalf("Default error: %v", err)
	}
	if !ok {
		t.Fatal("Default should be set")
	}
	if got.Provider != info.Provider || got.Model != info.Model || got.Endpoint != info.Endpoint {
		t.Errorf("Default = %+v, want %+v", got, info)
	}
}

func TestBeginCopilotLogin_ReturnsDeviceCodeBeforeAuthorisation(t *testing.T) {
	t.Parallel()

	// Gate authorisation on a channel so the test observes the
	// "pending" state after the device code is issued but before the
	// user completes auth.
	authorise := make(chan struct{})
	var pollCount atomic.Int32

	mux := http.NewServeMux()
	mux.HandleFunc("/login/device/code", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"device_code":      "dev-code",
			"user_code":        "WEB-1234",
			"verification_uri": "https://github.com/login/device",
			"interval":         0,
			"expires_in":       60,
		})
	})
	mux.HandleFunc("/login/oauth/access_token", func(w http.ResponseWriter, r *http.Request) {
		pollCount.Add(1)
		w.Header().Set("Content-Type", "application/json")
		select {
		case <-authorise:
			json.NewEncoder(w).Encode(map[string]any{"access_token": "gho_web_token"})
		default:
			json.NewEncoder(w).Encode(map[string]any{"error": "authorization_pending"})
		}
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	mgr, as, _ := testManager(t,
		WithHTTPClient(redirectClient(srv)),
		WithPollInterval(5*time.Millisecond),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	flow, err := mgr.BeginCopilotLogin(ctx)
	if err != nil {
		t.Fatalf("BeginCopilotLogin: %v", err)
	}
	if flow.UserCode() != "WEB-1234" {
		t.Errorf("UserCode = %q, want WEB-1234", flow.UserCode())
	}
	if flow.VerifyURL() == "" {
		t.Error("VerifyURL should be populated")
	}
	if s := flow.State(); s != DeviceFlowPending {
		t.Errorf("initial state = %q, want pending", s)
	}

	// Release the poll loop and wait for completion.
	close(authorise)

	deadline := time.Now().Add(2 * time.Second)
	for flow.State() == DeviceFlowPending {
		if time.Now().After(deadline) {
			t.Fatal("device flow did not complete within deadline")
		}
		time.Sleep(10 * time.Millisecond)
	}

	if flow.State() != DeviceFlowAuthenticated {
		t.Fatalf("final state = %q, err = %v", flow.State(), flow.Err())
	}
	if flow.Err() != nil {
		t.Fatalf("Err after success = %v", flow.Err())
	}

	token, err := as.Load(context.Background(), keyCopilotGitHub)
	if err != nil {
		t.Fatalf("Load github token: %v", err)
	}
	if token != "gho_web_token" {
		t.Errorf("stored token = %q, want gho_web_token", token)
	}
	if pollCount.Load() < 1 {
		t.Error("expected at least one poll")
	}
}

func TestBeginCopilotLogin_PropagatesErrors(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	mux.HandleFunc("/login/device/code", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	mgr, _, _ := testManager(t, WithHTTPClient(redirectClient(srv)))

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	flow, err := mgr.BeginCopilotLogin(ctx)
	if err == nil {
		t.Fatal("expected error when device code request fails")
	}
	if flow == nil {
		t.Fatal("flow should still be returned so caller can inspect state")
	}
	if flow.State() != DeviceFlowError {
		t.Errorf("state = %q, want error", flow.State())
	}
}

func TestBeginOpenAIDeviceLogin_MirrorsTokenIntoKeySlot(t *testing.T) {
	t.Parallel()

	authorise := make(chan struct{})
	mux := http.NewServeMux()
	mux.HandleFunc("/oauth/device/code", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"device_code":              "oai-dev",
			"user_code":                "OAI-WEB-9",
			"verification_uri":         "https://auth0.openai.com/activate",
			"verification_uri_complete": "https://auth0.openai.com/activate?user_code=OAI-WEB-9",
			"interval":                 0,
			"expires_in":               60,
		})
	})
	mux.HandleFunc("/oauth/token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		select {
		case <-authorise:
			json.NewEncoder(w).Encode(map[string]any{
				"access_token": "oai-access-token",
				"refresh_token": "oai-refresh",
				"expires_in":   3600,
			})
		default:
			json.NewEncoder(w).Encode(map[string]any{"error": "authorization_pending"})
		}
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	mgr, as, _ := testManager(t,
		WithHTTPClient(redirectClient(srv)),
		WithPollInterval(5*time.Millisecond),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	flow, err := mgr.BeginOpenAIDeviceLogin(ctx, "my-client-id")
	if err != nil {
		t.Fatalf("BeginOpenAIDeviceLogin: %v", err)
	}
	if flow.UserCode() != "OAI-WEB-9" {
		t.Errorf("UserCode = %q, want OAI-WEB-9", flow.UserCode())
	}
	// Verification URIs with a completion parameter should be preferred
	// so the UI can hand the user a one-click link.
	if flow.VerifyURL() == "" {
		t.Error("VerifyURL should be populated")
	}

	close(authorise)

	deadline := time.Now().Add(2 * time.Second)
	for flow.State() == DeviceFlowPending {
		if time.Now().After(deadline) {
			t.Fatal("flow did not complete")
		}
		time.Sleep(10 * time.Millisecond)
	}

	if flow.State() != DeviceFlowAuthenticated {
		t.Fatalf("final state = %q, err = %v", flow.State(), flow.Err())
	}

	// The access token must have been mirrored into the API-key slot
	// so the existing LoadProvider code path picks it up.
	key, err := as.Load(context.Background(), keyOpenAI)
	if err != nil {
		t.Fatalf("load openai api key: %v", err)
	}
	if key != "oai-access-token" {
		t.Errorf("mirrored key = %q, want oai-access-token", key)
	}
}

func TestLogout_OpenAI_ClearsOAuthRecord(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	// Seed both slots: the API key (as if via SaveAPIKey) and the
	// OAuth device-flow token record (as if via BeginOpenAIDeviceLogin).
	mgr, as, _ := testManager(t)
	if err := as.Save(ctx, keyOpenAI, "sk-mirror"); err != nil {
		t.Fatalf("seed api key: %v", err)
	}
	if err := as.Save(ctx, "openai_token", `{"access_token":"oai-access","refresh_token":"r"}`); err != nil {
		t.Fatalf("seed oauth record: %v", err)
	}

	if err := mgr.Logout(ctx, ProviderOpenAI); err != nil {
		t.Fatalf("Logout: %v", err)
	}

	if _, err := as.Load(ctx, keyOpenAI); err == nil {
		t.Error("API key should be cleared after logout")
	}
	if _, err := as.Load(ctx, "openai_token"); err == nil {
		t.Error("OAuth token record should be cleared after logout")
	}
}

func TestBeginOpenAIDeviceLogin_RequiresClientID(t *testing.T) {
	t.Parallel()
	mgr, _, _ := testManager(t)
	_, err := mgr.BeginOpenAIDeviceLogin(context.Background(), "")
	if err == nil {
		t.Fatal("expected error for empty client ID")
	}
}

func TestBeginCopilotLogin_CancelAbortsBackgroundPolling(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	mux.HandleFunc("/login/device/code", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"device_code":      "dev",
			"user_code":        "ABC",
			"verification_uri": "https://github.com/login/device",
			"interval":         0,
			"expires_in":       60,
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
	flow.Cancel()

	deadline := time.Now().Add(2 * time.Second)
	for flow.State() == DeviceFlowPending {
		if time.Now().After(deadline) {
			t.Fatal("flow did not terminate after Cancel")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if flow.State() != DeviceFlowError {
		t.Errorf("state after cancel = %q, want error", flow.State())
	}
}
