package profile

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stack-bound/stackllm/config"
)

func TestSaveAPIKey_StoreSaveError(t *testing.T) {
	t.Parallel()
	fs := newFailingAuthStore()
	fs.failSave[keyOpenAI] = true
	mgr := New(
		WithAuthStore(fs),
		WithConfigStore(&config.Store{Path: filepath.Join(t.TempDir(), "c.json")}),
	)
	err := mgr.SaveAPIKey(context.Background(), ProviderOpenAI, "sk-x")
	if err == nil || !strings.Contains(err.Error(), "save openai key") {
		t.Errorf("error = %v, want save error", err)
	}
}

func TestSaveOllamaURL_ConfigErrors(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	t.Run("load error", func(t *testing.T) {
		t.Parallel()
		mgr := corruptConfigManager(t)
		if err := mgr.SaveOllamaURL(ctx, "http://x"); err == nil || !strings.Contains(err.Error(), "load config") {
			t.Errorf("error = %v, want load config error", err)
		}
	})

	t.Run("save error", func(t *testing.T) {
		t.Parallel()
		mgr := unsavableConfigManager(t)
		if err := mgr.SaveOllamaURL(ctx, "http://x"); err == nil || !strings.Contains(err.Error(), "save ollama config") {
			t.Errorf("error = %v, want save error", err)
		}
	})
}

func TestDefault_ConfigLoadError(t *testing.T) {
	t.Parallel()
	mgr := corruptConfigManager(t)
	if _, _, err := mgr.Default(context.Background()); err == nil || !strings.Contains(err.Error(), "load config") {
		t.Errorf("error = %v, want load config error", err)
	}
}

func TestBeginCopilotLogin_TimesOutWaitingForDeviceCode(t *testing.T) {
	t.Parallel()

	// The device-code endpoint never responds, so no code is ever
	// issued and the caller's deadline elapses first.
	release := make(chan struct{})
	mux := http.NewServeMux()
	mux.HandleFunc("/login/device/code", func(w http.ResponseWriter, r *http.Request) {
		// Drain the body so the server's background read can detect the
		// client disconnect, then block until cancellation or teardown.
		_, _ = io.ReadAll(r.Body)
		select {
		case <-r.Context().Done():
		case <-release:
		}
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(release) }) // runs before srv.Close (LIFO)

	mgr, _, _ := testManager(t, WithHTTPClient(redirectClient(srv)))

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	flow, err := mgr.BeginCopilotLogin(ctx)
	if err == nil || !strings.Contains(err.Error(), "waiting for device code") {
		t.Fatalf("error = %v, want waiting-for-device-code timeout", err)
	}
	if flow.State() != DeviceFlowError {
		t.Errorf("state = %q, want error", flow.State())
	}
	if flow.Err() == nil {
		t.Error("flow.Err should carry the timeout cause")
	}

	// The background goroutine must exit (its context was cancelled).
	// Generous deadline: under a heavily loaded machine the goroutine's
	// aborted HTTP round-trip can take a while to unwind.
	waitCtx, waitCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer waitCancel()
	if err := flow.Wait(waitCtx); err != nil {
		t.Fatalf("Wait after timeout = %v", err)
	}
}

func TestBeginOpenAIDeviceLogin_MirrorSaveFails(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	mux.HandleFunc("/oauth/device/code", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"device_code":      "oai-dev",
			"user_code":        "MIRROR-1",
			"verification_uri": "https://auth0.openai.com/activate",
			"interval":         0,
			"expires_in":       60,
		})
	})
	mux.HandleFunc("/oauth/token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "oai-access",
			"refresh_token": "oai-refresh",
			"expires_in":    3600,
		})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	// The OAuth record itself persists fine, but mirroring the access
	// token into the API-key slot fails — the flow must surface that.
	fs := newFailingAuthStore()
	fs.failSave[keyOpenAI] = true
	mgr := New(
		WithAuthStore(fs),
		WithConfigStore(&config.Store{Path: filepath.Join(t.TempDir(), "c.json")}),
		WithHTTPClient(redirectClient(srv)),
		WithPollInterval(time.Millisecond),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	flow, err := mgr.BeginOpenAIDeviceLogin(ctx, "client-id")
	if err != nil {
		t.Fatalf("BeginOpenAIDeviceLogin: %v", err)
	}

	waitCtx, waitCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer waitCancel()
	if err := flow.Wait(waitCtx); err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if flow.State() != DeviceFlowError {
		t.Fatalf("state = %q, want error when mirror save fails", flow.State())
	}
	if flow.Err() == nil || !strings.Contains(flow.Err().Error(), "mirror key") {
		t.Errorf("flow.Err = %v, want mirror key error", flow.Err())
	}
}
