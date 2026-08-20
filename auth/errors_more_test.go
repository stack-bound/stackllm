package auth

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// failingSaveStore wraps a TokenStore but rejects every Save, for
// exercising persistence-error paths.
type failingSaveStore struct {
	inner TokenStore
}

func (f *failingSaveStore) Load(ctx context.Context, key string) (string, error) {
	return f.inner.Load(ctx, key)
}

func (f *failingSaveStore) Save(context.Context, string, string) error {
	return fmt.Errorf("save disabled")
}

func (f *failingSaveStore) Delete(ctx context.Context, key string) error {
	return f.inner.Delete(ctx, key)
}

func TestExtractChatGPTAccountID_Malformed(t *testing.T) {
	t.Parallel()

	claims := map[string]any{"chatgpt_account_id": "acc-padded"}
	claimsJSON, err := json.Marshal(claims)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	tests := []struct {
		name  string
		token string
		want  string
	}{
		{
			name:  "no dot separators",
			token: "justonesegment",
			want:  "",
		},
		{
			name: "padded std base64 payload falls back to StdEncoding",
			// StdEncoding output includes '=' padding, which
			// RawURLEncoding rejects, forcing the fallback path.
			token: "h." + base64.StdEncoding.EncodeToString(claimsJSON) + ".s",
			want:  "acc-padded",
		},
		{
			name:  "payload not base64 at all",
			token: "h.!!!not-base64!!!.s",
			want:  "",
		},
		{
			name:  "payload decodes but is not JSON",
			token: "h." + base64.RawURLEncoding.EncodeToString([]byte("not-json{")) + ".s",
			want:  "",
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := extractChatGPTAccountID(tt.token); got != tt.want {
				t.Errorf("extractChatGPTAccountID = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestExchangeCodexToken_TransportAndRequestErrors(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	t.Run("transport error", func(t *testing.T) {
		t.Parallel()
		client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return nil, fmt.Errorf("connection refused")
		})}
		_, err := exchangeCodexToken(ctx, client, "https://test.codex.local/token", url.Values{})
		if err == nil || !strings.Contains(err.Error(), "token request") {
			t.Errorf("error = %v, want token request error", err)
		}
	})

	t.Run("invalid token URL", func(t *testing.T) {
		t.Parallel()
		_, err := exchangeCodexToken(ctx, http.DefaultClient, "http://bad url\n", url.Values{})
		if err == nil || !strings.Contains(err.Error(), "build token request") {
			t.Errorf("error = %v, want build token request error", err)
		}
	})
}

func TestExchangeOpenAIToken_TransportAndRequestErrors(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	t.Run("transport error", func(t *testing.T) {
		t.Parallel()
		client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return nil, fmt.Errorf("connection refused")
		})}
		_, err := exchangeOpenAIToken(ctx, client, "https://test.openai.local/token", url.Values{})
		if err == nil || !strings.Contains(err.Error(), "token exchange") {
			t.Errorf("error = %v, want token exchange error", err)
		}
	})

	t.Run("invalid token URL", func(t *testing.T) {
		t.Parallel()
		_, err := exchangeOpenAIToken(ctx, http.DefaultClient, "http://bad url\n", url.Values{})
		if err == nil || !strings.Contains(err.Error(), "token request") {
			t.Errorf("error = %v, want token request error", err)
		}
	})
}

func TestCodexDeviceSource_TokenRefreshFailure(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	ctx := context.Background()
	rec := CodexTokenRecord{
		AccessToken:  "stale",
		RefreshToken: "refresh-dead",
		ExpiresAt:    time.Now().Add(-time.Hour),
	}
	if err := saveCodexRecord(ctx, store, rec); err != nil {
		t.Fatalf("seed: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"error": "invalid_grant"})
	}))
	defer srv.Close()

	src := NewCodexDeviceSource(CodexDeviceConfig{
		Store:      store,
		HTTPClient: srv.Client(),
		TokenURL:   srv.URL,
	})

	_, err := src.Token(ctx)
	if err == nil || !strings.Contains(err.Error(), "invalid_grant") {
		t.Errorf("Token error = %v, want invalid_grant", err)
	}
	// The stale record must remain untouched — a failed refresh must
	// not destroy the stored credential.
	stored, loadErr := LoadCodexRecord(ctx, store)
	if loadErr != nil {
		t.Fatalf("LoadCodexRecord: %v", loadErr)
	}
	if stored.RefreshToken != "refresh-dead" {
		t.Errorf("stored RefreshToken = %q, want unchanged refresh-dead", stored.RefreshToken)
	}
}

func TestCodexSources_SaveFailureSurfaces(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	idTok := "h." + base64.RawURLEncoding.EncodeToString([]byte(`{"chatgpt_account_id":"acc"}`)) + ".s"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"access_token": "tok",
			"id_token":     idTok,
			"expires_in":   3600,
		})
	}))
	t.Cleanup(srv.Close) // outlives the parallel subtests, unlike defer

	t.Run("web Complete", func(t *testing.T) {
		t.Parallel()
		src := NewCodexWebFlowSource(CodexWebFlowConfig{
			Store:      &failingSaveStore{inner: NewMemoryStore()},
			HTTPClient: srv.Client(),
			AuthURL:    "https://test.codex.local/authorize",
			TokenURL:   srv.URL,
		})
		authURL, err := src.Begin(ctx, "http://localhost:1455/auth/callback")
		if err != nil {
			t.Fatalf("Begin: %v", err)
		}
		parsed, err := url.Parse(authURL)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		err = src.Complete(ctx, parsed.Query().Get("state"), "code")
		if err == nil || !strings.Contains(err.Error(), "save") {
			t.Errorf("Complete error = %v, want save failure", err)
		}
	})

	t.Run("web Token refresh persist", func(t *testing.T) {
		t.Parallel()
		inner := NewMemoryStore()
		rec := CodexTokenRecord{
			AccessToken:  "stale",
			RefreshToken: "refresh-ok",
			ExpiresAt:    time.Now().Add(-time.Hour),
		}
		if err := saveCodexRecord(ctx, inner, rec); err != nil {
			t.Fatalf("seed: %v", err)
		}
		src := NewCodexWebFlowSource(CodexWebFlowConfig{
			Store:      &failingSaveStore{inner: inner},
			HTTPClient: srv.Client(),
			TokenURL:   srv.URL,
		})
		_, err := src.Token(ctx)
		if err == nil || !strings.Contains(err.Error(), "save") {
			t.Errorf("Token error = %v, want save failure", err)
		}
	})
}

func TestOpenAIWebFlowSource_CompleteSaveFailure(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"access_token": "tok",
			"expires_in":   3600,
		})
	}))
	defer srv.Close()

	src := NewOpenAIWebFlowSource(OpenAIWebFlowConfig{
		ClientID:   "c",
		Store:      &failingSaveStore{inner: NewMemoryStore()},
		HTTPClient: srv.Client(),
		AuthURL:    "https://test.openai.local/authorize",
		TokenURL:   srv.URL,
	})
	authURL, err := src.Begin(ctx, "http://localhost:9999/callback")
	if err != nil {
		t.Fatalf("Begin: %v", err)
	}
	parsed, err := url.Parse(authURL)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	err = src.Complete(ctx, parsed.Query().Get("state"), "code")
	if err == nil || !strings.Contains(err.Error(), "save") {
		t.Errorf("Complete error = %v, want save failure", err)
	}
}

func TestCopilotTokenSource_CachesExchangedToken(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	ctx := context.Background()
	if err := store.Save(ctx, copilotStoreKey, "ghu_valid"); err != nil {
		t.Fatalf("seed: %v", err)
	}

	var exchanges atomic.Int32
	client := &http.Client{Transport: copilotRoundTripFunc(func(req *http.Request) (*http.Response, error) {
		exchanges.Add(1)
		return jsonResponse(map[string]any{
			"token":      "copilot-cached",
			"expires_at": time.Now().Add(30 * time.Minute).Unix(),
		}), nil
	})}

	src := NewCopilotSource(CopilotConfig{
		Store:           store,
		HTTPClient:      client,
		CopilotTokenURL: "https://mcp.test/copilot_internal/v2/token",
	})

	if _, err := src.Token(ctx); err != nil {
		t.Fatalf("Token: %v", err)
	}
	tok, err := src.Token(ctx)
	if err != nil {
		t.Fatalf("Token (second): %v", err)
	}
	if tok.AccessToken != "copilot-cached" {
		t.Errorf("AccessToken = %q", tok.AccessToken)
	}
	if exchanges.Load() != 1 {
		t.Errorf("exchanges = %d, want 1 — Phase 2 token must be cached until expiry", exchanges.Load())
	}
}

func TestCopilotTokenSource_ExchangeDecodeError(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	ctx := context.Background()
	if err := store.Save(ctx, copilotStoreKey, "ghu_valid"); err != nil {
		t.Fatalf("seed: %v", err)
	}

	client := &http.Client{Transport: copilotRoundTripFunc(func(req *http.Request) (*http.Response, error) {
		resp := jsonResponse(nil)
		resp.Body = http.NoBody // undecodable body
		return resp, nil
	})}

	src := NewCopilotSource(CopilotConfig{
		Store:           store,
		HTTPClient:      client,
		CopilotTokenURL: "https://mcp.test/copilot_internal/v2/token",
	})
	if _, err := src.Token(ctx); err == nil {
		t.Error("expected decode error for empty exchange body")
	}
}

func TestCopilotTokenSource_PollTransientFailuresThenSuccess(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	var polls atomic.Int32
	client := &http.Client{Transport: copilotRoundTripFunc(func(req *http.Request) (*http.Response, error) {
		switch req.URL.String() {
		case "https://mcp.test/login/device/code":
			return jsonResponse(map[string]any{
				"device_code":      "dev-code",
				"user_code":        "ABCD-3333",
				"verification_uri": "https://github.com/login/device",
				"interval":         0,
				"expires_in":       60,
			}), nil
		case "https://mcp.test/login/oauth/access_token":
			switch polls.Add(1) {
			case 1:
				return nil, fmt.Errorf("transient network failure")
			case 2:
				resp := jsonResponse(nil)
				resp.Body = http.NoBody // undecodable → keep polling
				return resp, nil
			default:
				return jsonResponse(map[string]any{"access_token": "ghu_after_retries"}), nil
			}
		default:
			t.Errorf("unexpected URL: %s", req.URL)
			return nil, fmt.Errorf("unexpected URL")
		}
	})}

	src := NewCopilotSource(CopilotConfig{
		Store:          store,
		PollInterval:   time.Millisecond,
		HTTPClient:     client,
		DeviceCodeURL:  "https://mcp.test/login/device/code",
		AccessTokenURL: "https://mcp.test/login/oauth/access_token",
	})

	if err := src.Login(context.Background()); err != nil {
		t.Fatalf("Login: %v", err)
	}
	if polls.Load() < 3 {
		t.Errorf("polls = %d, want at least 3 (transient failures retried)", polls.Load())
	}
	ghTok, err := store.Load(context.Background(), copilotStoreKey)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if ghTok != "ghu_after_retries" {
		t.Errorf("stored GitHub token = %q, want ghu_after_retries", ghTok)
	}
}

func TestOpenAIDeviceSource_PollTransientFailuresThenSuccess(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	var polls atomic.Int32
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		switch req.URL.String() {
		case "https://test.openai.local/oauth/device/code":
			return jsonResponse(map[string]any{
				"device_code":      "dev-abc",
				"user_code":        "CODE-5555",
				"verification_uri": "https://openai.com/verify",
				"interval":         0,
				"expires_in":       60,
			}), nil
		case "https://test.openai.local/oauth/token":
			switch polls.Add(1) {
			case 1:
				return nil, fmt.Errorf("transient network failure")
			case 2:
				resp := jsonResponse(nil)
				resp.Body = http.NoBody // undecodable → keep polling
				return resp, nil
			default:
				return jsonResponse(map[string]any{
					"access_token": "tok-after-retries",
					"expires_in":   3600,
				}), nil
			}
		default:
			t.Errorf("unexpected URL: %s", req.URL)
			return nil, fmt.Errorf("unexpected URL")
		}
	})}

	src := NewOpenAIDeviceSource(OpenAIDeviceConfig{
		ClientID:      "test-client",
		Store:         store,
		PollInterval:  time.Millisecond,
		HTTPClient:    client,
		DeviceCodeURL: "https://test.openai.local/oauth/device/code",
		TokenURL:      "https://test.openai.local/oauth/token",
	})

	if err := src.Login(context.Background()); err != nil {
		t.Fatalf("Login: %v", err)
	}
	record, err := loadOpenAITokenRecord(context.Background(), store, openaiStoreKey)
	if err != nil {
		t.Fatalf("loadOpenAITokenRecord: %v", err)
	}
	if record.AccessToken != "tok-after-retries" {
		t.Errorf("stored AccessToken = %q", record.AccessToken)
	}
}

func TestCodexDeviceSource_PollTransientAndDecodeBehaviour(t *testing.T) {
	t.Parallel()

	t.Run("transport error keeps polling", func(t *testing.T) {
		t.Parallel()
		store := NewMemoryStore()
		var polls atomic.Int32
		idTok := "h." + base64.RawURLEncoding.EncodeToString([]byte(`{"chatgpt_account_id":"acc"}`)) + ".s"
		client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			switch req.URL.String() {
			case "https://test.codex.local/usercode":
				return jsonResponse(map[string]any{
					"device_auth_id": "dev-abc",
					"user_code":      "CODE-6666",
					"interval":       0,
					"expires_in":     60,
				}), nil
			case "https://test.codex.local/poll":
				if polls.Add(1) == 1 {
					return nil, fmt.Errorf("transient network failure")
				}
				return jsonResponse(map[string]any{
					"authorization_code": "auth-code",
					"code_verifier":      "verifier",
				}), nil
			case "https://test.codex.local/token":
				return jsonResponse(map[string]any{
					"access_token": "tok",
					"id_token":     idTok,
					"expires_in":   3600,
				}), nil
			default:
				t.Errorf("unexpected URL: %s", req.URL)
				return nil, fmt.Errorf("unexpected URL")
			}
		})}
		src := NewCodexDeviceSource(CodexDeviceConfig{
			Store:         store,
			PollInterval:  time.Millisecond,
			HTTPClient:    client,
			DeviceCodeURL: "https://test.codex.local/usercode",
			DevicePollURL: "https://test.codex.local/poll",
			TokenURL:      "https://test.codex.local/token",
		})
		if err := src.Login(context.Background()); err != nil {
			t.Fatalf("Login: %v", err)
		}
		if polls.Load() < 2 {
			t.Errorf("polls = %d, want at least 2", polls.Load())
		}
	})

	t.Run("undecodable poll body is fatal", func(t *testing.T) {
		t.Parallel()
		client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			switch req.URL.String() {
			case "https://test.codex.local/usercode":
				return jsonResponse(map[string]any{
					"device_auth_id": "dev-abc",
					"user_code":      "CODE-7777",
					"interval":       0,
					"expires_in":     60,
				}), nil
			case "https://test.codex.local/poll":
				resp := jsonResponse(nil)
				resp.Body = http.NoBody
				return resp, nil
			default:
				t.Errorf("unexpected URL: %s", req.URL)
				return nil, fmt.Errorf("unexpected URL")
			}
		})}
		src := NewCodexDeviceSource(CodexDeviceConfig{
			Store:         NewMemoryStore(),
			PollInterval:  time.Millisecond,
			HTTPClient:    client,
			DeviceCodeURL: "https://test.codex.local/usercode",
			DevicePollURL: "https://test.codex.local/poll",
			TokenURL:      "https://test.codex.local/token",
		})
		err := src.Login(context.Background())
		if err == nil || !strings.Contains(err.Error(), "decode poll response") {
			t.Errorf("Login error = %v, want decode poll response", err)
		}
	})

	t.Run("undecodable usercode body is fatal", func(t *testing.T) {
		t.Parallel()
		client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			resp := jsonResponse(nil)
			resp.Body = http.NoBody
			return resp, nil
		})}
		src := NewCodexDeviceSource(CodexDeviceConfig{
			Store:         NewMemoryStore(),
			HTTPClient:    client,
			DeviceCodeURL: "https://test.codex.local/usercode",
		})
		err := src.Login(context.Background())
		if err == nil || !strings.Contains(err.Error(), "decode usercode") {
			t.Errorf("Login error = %v, want decode usercode", err)
		}
	})
}

func TestOpenAIDeviceSource_LoginSaveFailure(t *testing.T) {
	t.Parallel()

	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		switch req.URL.String() {
		case "https://test.openai.local/oauth/device/code":
			return jsonResponse(map[string]any{
				"device_code":      "dev-abc",
				"user_code":        "CODE-8888",
				"verification_uri": "https://openai.com/verify",
				"interval":         0,
				"expires_in":       60,
			}), nil
		case "https://test.openai.local/oauth/token":
			return jsonResponse(map[string]any{
				"access_token": "tok",
				"expires_in":   3600,
			}), nil
		default:
			t.Errorf("unexpected URL: %s", req.URL)
			return nil, fmt.Errorf("unexpected URL")
		}
	})}

	src := NewOpenAIDeviceSource(OpenAIDeviceConfig{
		ClientID:      "test-client",
		Store:         &failingSaveStore{inner: NewMemoryStore()},
		PollInterval:  time.Millisecond,
		HTTPClient:    client,
		DeviceCodeURL: "https://test.openai.local/oauth/device/code",
		TokenURL:      "https://test.openai.local/oauth/token",
	})

	err := src.Login(context.Background())
	if err == nil || !strings.Contains(err.Error(), "save") {
		t.Errorf("Login error = %v, want save failure", err)
	}
}

func TestOpenAIWebFlowSource_TokenFallsBackToLoginWhenRefreshFails(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	ctx := context.Background()
	expired := openAITokenRecord{
		AccessToken:  "old-web",
		RefreshToken: "dead-refresh",
		ExpiresAt:    time.Now().Add(-time.Hour),
	}
	if err := saveOpenAITokenRecord(ctx, store, openaiWebStoreKey, expired); err != nil {
		t.Fatalf("seed: %v", err)
	}

	var refreshTried atomic.Int32
	tokenServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Errorf("ParseForm: %v", err)
		}
		if r.Form.Get("grant_type") == "refresh_token" {
			refreshTried.Add(1)
			http.Error(w, `{"error":"invalid_grant"}`, http.StatusBadRequest)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "relogin-web",
			"refresh_token": "relogin-web-refresh",
			"expires_in":    3600,
		})
	}))
	defer tokenServer.Close()

	port := freePort(t)
	urlCh := make(chan string, 1)
	src := NewOpenAIWebFlowSource(OpenAIWebFlowConfig{
		ClientID:   "web-client",
		Port:       port,
		Store:      store,
		OnOpenURL:  func(u string) { urlCh <- u },
		HTTPClient: tokenServer.Client(),
		AuthURL:    "https://test.openai.local/authorize",
		TokenURL:   tokenServer.URL,
	})

	type tokenResult struct {
		tok *Token
		err error
	}
	resCh := make(chan tokenResult, 1)
	go func() {
		tok, err := src.Token(ctx)
		resCh <- tokenResult{tok, err}
	}()

	var authURL string
	select {
	case authURL = <-urlCh:
	case res := <-resCh:
		t.Fatalf("Token exited before publishing auth URL: %v", res.err)
	case <-time.After(30 * time.Second):
		t.Fatal("timed out waiting for auth URL")
	}
	parsed, err := url.Parse(authURL)
	if err != nil {
		t.Fatalf("parse authURL: %v", err)
	}
	state := parsed.Query().Get("state")

	resp := driveCallback(t, fmt.Sprintf("http://localhost:%d/callback?state=%s&code=x", port, url.QueryEscape(state)))
	resp.Body.Close()

	res := <-resCh
	if res.err != nil {
		t.Fatalf("Token: %v", res.err)
	}
	if refreshTried.Load() != 1 {
		t.Errorf("refresh attempts = %d, want 1", refreshTried.Load())
	}
	if res.tok.AccessToken != "relogin-web" {
		t.Errorf("AccessToken = %q, want relogin-web", res.tok.AccessToken)
	}

	record, err := loadOpenAITokenRecord(ctx, store, openaiWebStoreKey)
	if err != nil {
		t.Fatalf("loadOpenAITokenRecord: %v", err)
	}
	if record.AccessToken != "relogin-web" || record.RefreshToken != "relogin-web-refresh" {
		t.Errorf("stored record = %+v, want re-login tokens", record)
	}
}
