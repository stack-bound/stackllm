package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestFlexInt_UnmarshalJSON(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		input   string
		want    flexInt
		wantErr bool
	}{
		{"number", "42", 42, false},
		{"quoted number", `"7"`, 7, false},
		{"null", "null", 0, false},
		{"empty string", `""`, 0, false},
		{"non-numeric string", `"abc"`, 0, true},
		{"wrong type", `{}`, 0, true},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var f flexInt
			err := json.Unmarshal([]byte(tt.input), &f)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("Unmarshal(%s): expected error", tt.input)
				}
				return
			}
			if err != nil {
				t.Fatalf("Unmarshal(%s): %v", tt.input, err)
			}
			if f != tt.want {
				t.Errorf("flexInt = %d, want %d", f, tt.want)
			}
		})
	}
}

func TestLoadCodexRecord_Errors(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	tests := []struct {
		name  string
		setup func(t *testing.T, store TokenStore)
	}{
		{
			name:  "missing key",
			setup: func(t *testing.T, store TokenStore) {},
		},
		{
			name: "invalid JSON",
			setup: func(t *testing.T, store TokenStore) {
				if err := store.Save(ctx, CodexStoreKey, "not-json{"); err != nil {
					t.Fatalf("seed: %v", err)
				}
			},
		},
		{
			name: "empty access token",
			setup: func(t *testing.T, store TokenStore) {
				if err := store.Save(ctx, CodexStoreKey, `{"refresh_token":"r"}`); err != nil {
					t.Fatalf("seed: %v", err)
				}
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			store := NewMemoryStore()
			tt.setup(t, store)
			if _, err := LoadCodexRecord(ctx, store); err == nil {
				t.Error("expected LoadCodexRecord error")
			}
		})
	}
}

func TestExchangeCodexToken_Errors(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	tests := []struct {
		name    string
		handler http.HandlerFunc
		wantErr string
	}{
		{
			name: "non-200 status",
			handler: func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, "boom", http.StatusInternalServerError)
			},
			wantErr: "status 500",
		},
		{
			name: "error field in body",
			handler: func(w http.ResponseWriter, r *http.Request) {
				json.NewEncoder(w).Encode(map[string]any{
					"error":             "access_denied",
					"error_description": "nope",
				})
			},
			wantErr: "access_denied",
		},
		{
			name: "empty access token",
			handler: func(w http.ResponseWriter, r *http.Request) {
				json.NewEncoder(w).Encode(map[string]any{"expires_in": 3600})
			},
			wantErr: "empty access token",
		},
		{
			name: "invalid JSON body",
			handler: func(w http.ResponseWriter, r *http.Request) {
				fmt.Fprint(w, "not-json{")
			},
			wantErr: "decode token response",
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(tt.handler)
			defer srv.Close()

			_, err := exchangeCodexToken(ctx, srv.Client(), srv.URL, url.Values{"grant_type": {"authorization_code"}})
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error = %q, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestCodexWebFlowConfig_Port(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		cfg  CodexWebFlowConfig
		want int
	}{
		{"default", CodexWebFlowConfig{}, codexDefaultPort},
		{"override", CodexWebFlowConfig{Port: 9321}, 9321},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := tt.cfg.port(); got != tt.want {
				t.Errorf("port() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestCodexWebFlowSource_LoginEndToEnd(t *testing.T) {
	t.Parallel()

	idTok := codexFakeIDToken(t, map[string]any{"chatgpt_account_id": "acc-login"})

	tokenServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Errorf("ParseForm: %v", err)
		}
		if r.Form.Get("grant_type") != "authorization_code" {
			t.Errorf("grant_type = %q", r.Form.Get("grant_type"))
		}
		if r.Form.Get("code") != "cb-code" {
			t.Errorf("code = %q, want cb-code", r.Form.Get("code"))
		}
		if r.Form.Get("client_id") != CodexDefaultClientID {
			t.Errorf("client_id = %q", r.Form.Get("client_id"))
		}
		if r.Form.Get("code_verifier") == "" {
			t.Error("expected code_verifier")
		}
		json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "codex-login-access",
			"refresh_token": "codex-login-refresh",
			"id_token":      idTok,
			"expires_in":    3600,
		})
	}))
	defer tokenServer.Close()

	port := freePort(t)
	urlCh := make(chan string, 1)
	var successCalled atomic.Bool
	store := NewMemoryStore()

	src := NewCodexWebFlowSource(CodexWebFlowConfig{
		Store:      store,
		Port:       port,
		OnOpenURL:  func(u string) { urlCh <- u },
		OnSuccess:  func() { successCalled.Store(true) },
		HTTPClient: tokenServer.Client(),
		AuthURL:    "https://test.codex.local/authorize",
		TokenURL:   tokenServer.URL,
	})

	if src.Record() != nil {
		t.Error("Record() should be nil before login")
	}

	loginErr := make(chan error, 1)
	go func() { loginErr <- src.Login(context.Background()) }()

	authURL := awaitAuthURL(t, urlCh, loginErr)
	parsed, err := url.Parse(authURL)
	if err != nil {
		t.Fatalf("parse authURL: %v", err)
	}
	q := parsed.Query()
	wantRedirect := fmt.Sprintf("http://localhost:%d%s", port, codexWebCallbackPath)
	if q.Get("redirect_uri") != wantRedirect {
		t.Errorf("redirect_uri = %q, want %q", q.Get("redirect_uri"), wantRedirect)
	}
	if q.Get("client_id") != CodexDefaultClientID {
		t.Errorf("client_id = %q", q.Get("client_id"))
	}
	state := q.Get("state")
	if state == "" {
		t.Fatal("missing state")
	}

	resp := driveCallback(t, fmt.Sprintf("%s?state=%s&code=cb-code", wantRedirect, url.QueryEscape(state)))
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("callback status = %d, body %s", resp.StatusCode, body)
	}
	if !strings.Contains(string(body), "Signed in") {
		t.Errorf("callback body = %q, want signed-in page", body)
	}

	if err := <-loginErr; err != nil {
		t.Fatalf("Login: %v", err)
	}
	if !successCalled.Load() {
		t.Error("expected OnSuccess callback")
	}

	// Record() must expose the exchanged credential, including the
	// account ID extracted from the id_token.
	rec := src.Record()
	if rec == nil {
		t.Fatal("Record() = nil after login")
	}
	if rec.ChatGPTAccountID != "acc-login" {
		t.Errorf("Record().ChatGPTAccountID = %q, want acc-login", rec.ChatGPTAccountID)
	}

	// Persisted round-trip with expiry honoured.
	stored, err := LoadCodexRecord(context.Background(), store)
	if err != nil {
		t.Fatalf("LoadCodexRecord: %v", err)
	}
	if stored.AccessToken != "codex-login-access" {
		t.Errorf("stored AccessToken = %q", stored.AccessToken)
	}
	if stored.ExpiresAt.Before(time.Now().Add(3500 * time.Second)) {
		t.Errorf("ExpiresAt = %v — expires_in=3600 not honoured (too early)", stored.ExpiresAt)
	}
	if stored.ExpiresAt.After(time.Now().Add(3700 * time.Second)) {
		t.Errorf("ExpiresAt = %v — expires_in=3600 not honoured (too late)", stored.ExpiresAt)
	}
}

func TestCodexWebFlowSource_LoginCallbackErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		query   func(state string) string
		wantErr string
	}{
		{
			name:    "state mismatch",
			query:   func(string) string { return "state=wrong&code=abc" },
			wantErr: "state mismatch",
		},
		{
			name: "provider error",
			query: func(state string) string {
				return "state=" + url.QueryEscape(state) + "&error=access_denied&error_description=nope"
			},
			wantErr: "access_denied",
		},
		{
			name:    "missing code",
			query:   func(state string) string { return "state=" + url.QueryEscape(state) },
			wantErr: "missing code",
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			port := freePort(t)
			urlCh := make(chan string, 1)
			src := NewCodexWebFlowSource(CodexWebFlowConfig{
				Store:     NewMemoryStore(),
				Port:      port,
				OnOpenURL: func(u string) { urlCh <- u },
				AuthURL:   "https://test.codex.local/authorize",
				TokenURL:  "https://test.codex.local/token",
			})

			loginErr := make(chan error, 1)
			go func() { loginErr <- src.Login(context.Background()) }()

			authURL := awaitAuthURL(t, urlCh, loginErr)
			parsed, err := url.Parse(authURL)
			if err != nil {
				t.Fatalf("parse authURL: %v", err)
			}
			state := parsed.Query().Get("state")

			resp := driveCallback(t, fmt.Sprintf("http://localhost:%d%s?%s", port, codexWebCallbackPath, tt.query(state)))
			resp.Body.Close()
			if resp.StatusCode != http.StatusBadRequest {
				t.Errorf("callback status = %d, want 400", resp.StatusCode)
			}

			err = <-loginErr
			if err == nil {
				t.Fatal("expected Login error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("Login error = %q, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestCodexWebFlowSource_LoginListenError(t *testing.T) {
	t.Parallel()

	l, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer l.Close()
	port := l.Addr().(*net.TCPAddr).Port

	src := NewCodexWebFlowSource(CodexWebFlowConfig{
		Store: NewMemoryStore(),
		Port:  port,
	})
	err = src.Login(context.Background())
	if err == nil {
		t.Fatal("expected listen error when port is occupied")
	}
	if !strings.Contains(err.Error(), "listen") {
		t.Errorf("error = %q, want listen error", err)
	}
}

func TestCodexWebFlowSource_LoginTimeout(t *testing.T) {
	t.Parallel()

	port := freePort(t)
	src := NewCodexWebFlowSource(CodexWebFlowConfig{
		Store: NewMemoryStore(),
		Port:  port,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	err := src.Login(ctx)
	if err == nil {
		t.Fatal("expected timeout error")
	}
	if !strings.Contains(err.Error(), "timed out") {
		t.Errorf("Login error = %q, want timeout", err)
	}
}

func TestCodexWebFlowSource_Token(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	t.Run("no stored record", func(t *testing.T) {
		t.Parallel()
		src := NewCodexWebFlowSource(CodexWebFlowConfig{Store: NewMemoryStore()})
		if _, err := src.Token(ctx); err == nil {
			t.Error("expected error with empty store")
		}
	})

	t.Run("valid stored record without network", func(t *testing.T) {
		t.Parallel()
		store := NewMemoryStore()
		rec := CodexTokenRecord{AccessToken: "stored-ok", ExpiresAt: time.Now().Add(time.Hour)}
		if err := saveCodexRecord(ctx, store, rec); err != nil {
			t.Fatalf("seed: %v", err)
		}
		client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			t.Errorf("unexpected HTTP call to %s", req.URL)
			return nil, fmt.Errorf("no network expected")
		})}
		src := NewCodexWebFlowSource(CodexWebFlowConfig{Store: store, HTTPClient: client})

		tok, err := src.Token(ctx)
		if err != nil {
			t.Fatalf("Token: %v", err)
		}
		if tok.AccessToken != "stored-ok" {
			t.Errorf("AccessToken = %q", tok.AccessToken)
		}
		// Second call must be served from the in-memory cache.
		tok2, err := src.Token(ctx)
		if err != nil {
			t.Fatalf("Token (cached): %v", err)
		}
		if tok2.AccessToken != "stored-ok" {
			t.Errorf("cached AccessToken = %q", tok2.AccessToken)
		}
		if src.Record() == nil || src.Record().AccessToken != "stored-ok" {
			t.Error("Record() should expose the loaded record")
		}
	})

	t.Run("expired without refresh token", func(t *testing.T) {
		t.Parallel()
		store := NewMemoryStore()
		rec := CodexTokenRecord{AccessToken: "stale", ExpiresAt: time.Now().Add(-time.Hour)}
		if err := saveCodexRecord(ctx, store, rec); err != nil {
			t.Fatalf("seed: %v", err)
		}
		src := NewCodexWebFlowSource(CodexWebFlowConfig{Store: store})
		_, err := src.Token(ctx)
		if err == nil {
			t.Fatal("expected error for expired token without refresh token")
		}
		if !strings.Contains(err.Error(), "re-run login") {
			t.Errorf("error = %q, want re-run login hint", err)
		}
	})

	t.Run("expired with refresh token refreshes and persists", func(t *testing.T) {
		t.Parallel()
		store := NewMemoryStore()
		rec := CodexTokenRecord{
			AccessToken:      "stale",
			RefreshToken:     "web-refresh-old",
			ChatGPTAccountID: "acc-kept",
			ExpiresAt:        time.Now().Add(-time.Hour),
		}
		if err := saveCodexRecord(ctx, store, rec); err != nil {
			t.Fatalf("seed: %v", err)
		}

		var refreshCalls atomic.Int32
		refreshServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			refreshCalls.Add(1)
			if err := r.ParseForm(); err != nil {
				t.Errorf("ParseForm: %v", err)
			}
			if r.Form.Get("grant_type") != "refresh_token" {
				t.Errorf("grant_type = %q, want refresh_token", r.Form.Get("grant_type"))
			}
			if r.Form.Get("refresh_token") != "web-refresh-old" {
				t.Errorf("refresh_token = %q", r.Form.Get("refresh_token"))
			}
			json.NewEncoder(w).Encode(map[string]any{
				"access_token": "web-fresh",
				"expires_in":   3600,
			})
		}))
		defer refreshServer.Close()

		src := NewCodexWebFlowSource(CodexWebFlowConfig{
			Store:      store,
			HTTPClient: refreshServer.Client(),
			TokenURL:   refreshServer.URL,
		})

		tok, err := src.Token(ctx)
		if err != nil {
			t.Fatalf("Token: %v", err)
		}
		if tok.AccessToken != "web-fresh" {
			t.Errorf("AccessToken = %q, want web-fresh", tok.AccessToken)
		}
		if refreshCalls.Load() != 1 {
			t.Errorf("refresh called %d times, want 1", refreshCalls.Load())
		}

		// Refresh response omitted refresh_token / id_token: originals
		// must be preserved in the persisted record.
		stored, err := LoadCodexRecord(ctx, store)
		if err != nil {
			t.Fatalf("LoadCodexRecord: %v", err)
		}
		if stored.AccessToken != "web-fresh" {
			t.Errorf("stored AccessToken = %q", stored.AccessToken)
		}
		if stored.RefreshToken != "web-refresh-old" {
			t.Errorf("stored RefreshToken = %q, want preserved web-refresh-old", stored.RefreshToken)
		}
		if stored.ChatGPTAccountID != "acc-kept" {
			t.Errorf("stored ChatGPTAccountID = %q, want preserved acc-kept", stored.ChatGPTAccountID)
		}
		if stored.ExpiresAt.Before(time.Now()) {
			t.Errorf("stored ExpiresAt = %v — want future", stored.ExpiresAt)
		}
	})
}

func TestCodexWebFlowSource_LogoutClearsRecordAndFlow(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	ctx := context.Background()
	if err := saveCodexRecord(ctx, store, CodexTokenRecord{AccessToken: "x"}); err != nil {
		t.Fatalf("seed: %v", err)
	}

	src := NewCodexWebFlowSource(CodexWebFlowConfig{Store: store})
	src.record = &CodexTokenRecord{AccessToken: "x"}
	if _, err := src.Begin(ctx, "http://localhost:1455/auth/callback"); err != nil {
		t.Fatalf("Begin: %v", err)
	}

	if err := src.Logout(ctx); err != nil {
		t.Fatalf("Logout: %v", err)
	}
	if src.Record() != nil {
		t.Error("Record() should be nil after Logout")
	}
	src.mu.Lock()
	flow := src.flow
	src.mu.Unlock()
	if flow != nil {
		t.Error("flow should be cleared on Logout")
	}
	if _, err := LoadCodexRecord(ctx, store); err == nil {
		t.Error("stored record should be deleted on Logout")
	}
}

func TestCodexDeviceSource_TokenLoadsStoredRecord(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	ctx := context.Background()
	rec := CodexTokenRecord{
		AccessToken:      "device-stored",
		ChatGPTAccountID: "acc-dev",
		ExpiresAt:        time.Now().Add(time.Hour),
	}
	if err := saveCodexRecord(ctx, store, rec); err != nil {
		t.Fatalf("seed: %v", err)
	}

	src := NewCodexDeviceSource(CodexDeviceConfig{Store: store})
	if src.Record() != nil {
		t.Error("Record() should be nil before Token loads the store")
	}

	tok, err := src.Token(ctx)
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if tok.AccessToken != "device-stored" {
		t.Errorf("AccessToken = %q", tok.AccessToken)
	}
	got := src.Record()
	if got == nil {
		t.Fatal("Record() = nil after Token")
	}
	if got.ChatGPTAccountID != "acc-dev" {
		t.Errorf("Record().ChatGPTAccountID = %q, want acc-dev", got.ChatGPTAccountID)
	}
}

func TestCodexDeviceSource_TokenExpiredNoRefresh(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	ctx := context.Background()
	rec := CodexTokenRecord{AccessToken: "stale", ExpiresAt: time.Now().Add(-time.Hour)}
	if err := saveCodexRecord(ctx, store, rec); err != nil {
		t.Fatalf("seed: %v", err)
	}

	src := NewCodexDeviceSource(CodexDeviceConfig{Store: store})
	_, err := src.Token(ctx)
	if err == nil {
		t.Fatal("expected error for expired token without refresh token")
	}
	if !strings.Contains(err.Error(), "re-run login") {
		t.Errorf("error = %q, want re-run login hint", err)
	}
}

func TestCodexDeviceSource_LoginErrors(t *testing.T) {
	t.Parallel()

	newSrc := func(store TokenStore, transport roundTripFunc) *CodexDeviceSource {
		return NewCodexDeviceSource(CodexDeviceConfig{
			Store:         store,
			PollInterval:  time.Millisecond,
			HTTPClient:    &http.Client{Transport: transport},
			DeviceCodeURL: "https://test.codex.local/usercode",
			DevicePollURL: "https://test.codex.local/poll",
			TokenURL:      "https://test.codex.local/token",
			VerifyURL:     "https://test.codex.local/codex/device",
			RedirectURI:   "https://test.codex.local/deviceauth/callback",
		})
	}

	t.Run("usercode non-200", func(t *testing.T) {
		t.Parallel()
		src := newSrc(NewMemoryStore(), func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusInternalServerError,
				Body:       io.NopCloser(strings.NewReader("server broke")),
				Header:     make(http.Header),
			}, nil
		})
		err := src.Login(context.Background())
		if err == nil || !strings.Contains(err.Error(), "status 500") {
			t.Errorf("Login error = %v, want usercode status 500", err)
		}
	})

	t.Run("usercode missing fields", func(t *testing.T) {
		t.Parallel()
		src := newSrc(NewMemoryStore(), func(req *http.Request) (*http.Response, error) {
			return jsonResponse(map[string]any{"interval": 1}), nil
		})
		err := src.Login(context.Background())
		if err == nil || !strings.Contains(err.Error(), "missing fields") {
			t.Errorf("Login error = %v, want missing fields", err)
		}
	})

	t.Run("poll fatal status", func(t *testing.T) {
		t.Parallel()
		src := newSrc(NewMemoryStore(), func(req *http.Request) (*http.Response, error) {
			switch req.URL.String() {
			case "https://test.codex.local/usercode":
				return jsonResponse(map[string]any{
					"device_auth_id": "dev-abc",
					"user_code":      "CODE-0001",
					"interval":       0,
					"expires_in":     60,
				}), nil
			case "https://test.codex.local/poll":
				return &http.Response{
					StatusCode: http.StatusInternalServerError,
					Body:       io.NopCloser(strings.NewReader("poll broke")),
					Header:     make(http.Header),
				}, nil
			default:
				t.Errorf("unexpected URL: %s", req.URL)
				return nil, fmt.Errorf("unexpected URL")
			}
		})
		err := src.Login(context.Background())
		if err == nil || !strings.Contains(err.Error(), "status 500") {
			t.Errorf("Login error = %v, want poll status 500", err)
		}
	})

	t.Run("empty authorization_code keeps polling then succeeds", func(t *testing.T) {
		t.Parallel()
		store := NewMemoryStore()
		var polls atomic.Int32
		idTok := codexFakeIDToken(t, map[string]any{"chatgpt_account_id": "acc-poll"})
		src := newSrc(store, func(req *http.Request) (*http.Response, error) {
			switch req.URL.String() {
			case "https://test.codex.local/usercode":
				return jsonResponse(map[string]any{
					"device_auth_id": "dev-abc",
					"user_code":      "CODE-0002",
					"interval":       0,
					"expires_in":     60,
				}), nil
			case "https://test.codex.local/poll":
				if polls.Add(1) == 1 {
					// 200 but no authorization_code yet — must keep polling.
					return jsonResponse(map[string]any{}), nil
				}
				return jsonResponse(map[string]any{
					"authorization_code": "auth-code",
					"code_verifier":      "verifier",
				}), nil
			case "https://test.codex.local/token":
				return jsonResponse(map[string]any{
					"access_token": "polled-access",
					"id_token":     idTok,
					"expires_in":   3600,
				}), nil
			default:
				t.Errorf("unexpected URL: %s", req.URL)
				return nil, fmt.Errorf("unexpected URL")
			}
		})
		if err := src.Login(context.Background()); err != nil {
			t.Fatalf("Login: %v", err)
		}
		if polls.Load() < 2 {
			t.Errorf("polls = %d, want at least 2", polls.Load())
		}
		rec, err := LoadCodexRecord(context.Background(), store)
		if err != nil {
			t.Fatalf("LoadCodexRecord: %v", err)
		}
		if rec.AccessToken != "polled-access" {
			t.Errorf("AccessToken = %q", rec.AccessToken)
		}
	})

	t.Run("token exchange failure surfaces", func(t *testing.T) {
		t.Parallel()
		src := newSrc(NewMemoryStore(), func(req *http.Request) (*http.Response, error) {
			switch req.URL.String() {
			case "https://test.codex.local/usercode":
				return jsonResponse(map[string]any{
					"device_auth_id": "dev-abc",
					"user_code":      "CODE-0003",
					"interval":       0,
					"expires_in":     60,
				}), nil
			case "https://test.codex.local/poll":
				return jsonResponse(map[string]any{
					"authorization_code": "auth-code",
					"code_verifier":      "verifier",
				}), nil
			case "https://test.codex.local/token":
				return jsonResponse(map[string]any{
					"error":             "invalid_grant",
					"error_description": "expired",
				}), nil
			default:
				t.Errorf("unexpected URL: %s", req.URL)
				return nil, fmt.Errorf("unexpected URL")
			}
		})
		err := src.Login(context.Background())
		if err == nil || !strings.Contains(err.Error(), "invalid_grant") {
			t.Errorf("Login error = %v, want invalid_grant", err)
		}
	})

	t.Run("context cancelled during poll wait", func(t *testing.T) {
		t.Parallel()
		src := NewCodexDeviceSource(CodexDeviceConfig{
			Store:        NewMemoryStore(),
			PollInterval: time.Hour, // force a long wait so cancellation wins
			HTTPClient: &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				return jsonResponse(map[string]any{
					"device_auth_id": "dev-abc",
					"user_code":      "CODE-0004",
					"interval":       0,
					"expires_in":     600,
				}), nil
			})},
			DeviceCodeURL: "https://test.codex.local/usercode",
			DevicePollURL: "https://test.codex.local/poll",
			TokenURL:      "https://test.codex.local/token",
		})
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
		defer cancel()
		err := src.Login(ctx)
		if err == nil || !strings.Contains(err.Error(), context.DeadlineExceeded.Error()) {
			t.Errorf("Login error = %v, want context deadline exceeded", err)
		}
	})

	t.Run("device flow timeout", func(t *testing.T) {
		t.Parallel()
		src := newSrc(NewMemoryStore(), func(req *http.Request) (*http.Response, error) {
			switch req.URL.String() {
			case "https://test.codex.local/usercode":
				return jsonResponse(map[string]any{
					"device_auth_id": "dev-abc",
					"user_code":      "CODE-0005",
					"interval":       0,
					"expires_in":     1, // 1s deadline
				}), nil
			case "https://test.codex.local/poll":
				return &http.Response{
					StatusCode: http.StatusForbidden, // still waiting, forever
					Body:       io.NopCloser(strings.NewReader("")),
					Header:     make(http.Header),
				}, nil
			default:
				t.Errorf("unexpected URL: %s", req.URL)
				return nil, fmt.Errorf("unexpected URL")
			}
		})
		err := src.Login(context.Background())
		if err == nil || !strings.Contains(err.Error(), "timed out") {
			t.Errorf("Login error = %v, want timed out", err)
		}
	})
}
