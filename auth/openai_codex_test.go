package auth

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// codexFakeIDToken builds a minimal JWT with just the middle (payload)
// segment populated. Signing is irrelevant for tests — the codex auth
// source only parses claims, never verifies.
func codexFakeIDToken(t *testing.T, claims map[string]any) string {
	t.Helper()
	body, err := json.Marshal(claims)
	if err != nil {
		t.Fatalf("marshal claims: %v", err)
	}
	return "header." + base64.RawURLEncoding.EncodeToString(body) + ".sig"
}

func TestCodexDeviceSource_FullFlow(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	var polls atomic.Int32
	var codeCalled, successCalled bool
	var capturedUserCode, capturedVerifyURL string

	idTok := codexFakeIDToken(t, map[string]any{
		"chatgpt_account_id": "acc-123",
	})

	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		switch req.URL.String() {
		case "https://test.codex.local/usercode":
			body, _ := io.ReadAll(req.Body)
			var p map[string]string
			_ = json.Unmarshal(body, &p)
			if p["client_id"] != CodexDefaultClientID {
				t.Errorf("client_id = %q, want %q", p["client_id"], CodexDefaultClientID)
			}
			return jsonResponse(map[string]any{
				"device_auth_id": "dev-abc",
				"user_code":      "WXYZ-1234",
				"interval":       0,
				"expires_in":     60,
			}), nil
		case "https://test.codex.local/poll":
			if polls.Add(1) == 1 {
				// Return 403 the first time — codex semantics: still waiting.
				return &http.Response{
					StatusCode: http.StatusForbidden,
					Body:       io.NopCloser(strings.NewReader("")),
					Header:     make(http.Header),
				}, nil
			}
			return jsonResponse(map[string]any{
				"authorization_code": "auth-code-xyz",
				"code_verifier":      "verifier-abc",
			}), nil
		case "https://test.codex.local/token":
			if err := req.ParseForm(); err != nil {
				t.Fatalf("ParseForm: %v", err)
			}
			if req.Form.Get("grant_type") != "authorization_code" {
				t.Errorf("grant_type = %q", req.Form.Get("grant_type"))
			}
			if req.Form.Get("code") != "auth-code-xyz" {
				t.Errorf("code = %q", req.Form.Get("code"))
			}
			if req.Form.Get("code_verifier") != "verifier-abc" {
				t.Errorf("code_verifier = %q", req.Form.Get("code_verifier"))
			}
			return jsonResponse(map[string]any{
				"access_token":  "codex-access",
				"refresh_token": "codex-refresh",
				"id_token":      idTok,
				"expires_in":    3600,
			}), nil
		default:
			t.Fatalf("unexpected URL: %s", req.URL.String())
			return nil, nil
		}
	})}

	src := NewCodexDeviceSource(CodexDeviceConfig{
		Store: store,
		OnCode: func(userCode, verifyURL string) {
			codeCalled = true
			capturedUserCode = userCode
			capturedVerifyURL = verifyURL
		},
		OnSuccess:     func() { successCalled = true },
		PollInterval:  time.Millisecond,
		HTTPClient:    client,
		DeviceCodeURL: "https://test.codex.local/usercode",
		DevicePollURL: "https://test.codex.local/poll",
		TokenURL:      "https://test.codex.local/token",
		VerifyURL:     "https://test.codex.local/codex/device",
		RedirectURI:   "https://test.codex.local/deviceauth/callback",
	})

	if err := src.Login(context.Background()); err != nil {
		t.Fatalf("Login error: %v", err)
	}

	if !codeCalled {
		t.Error("expected OnCode callback")
	}
	if capturedUserCode != "WXYZ-1234" {
		t.Errorf("userCode = %q", capturedUserCode)
	}
	if capturedVerifyURL != "https://test.codex.local/codex/device" {
		t.Errorf("verifyURL = %q", capturedVerifyURL)
	}
	if !successCalled {
		t.Error("expected OnSuccess callback")
	}

	rec, err := LoadCodexRecord(context.Background(), store)
	if err != nil {
		t.Fatalf("LoadCodexRecord: %v", err)
	}
	if rec.AccessToken != "codex-access" {
		t.Errorf("AccessToken = %q", rec.AccessToken)
	}
	if rec.RefreshToken != "codex-refresh" {
		t.Errorf("RefreshToken = %q", rec.RefreshToken)
	}
	if rec.ChatGPTAccountID != "acc-123" {
		t.Errorf("ChatGPTAccountID = %q, want acc-123 (extracted from id_token)", rec.ChatGPTAccountID)
	}
	if rec.ExpiresAt.IsZero() || rec.ExpiresAt.Before(time.Now()) {
		t.Errorf("ExpiresAt = %v — want future value", rec.ExpiresAt)
	}

	// Token() on the source should return the cached record without a
	// second network roundtrip.
	tok, err := src.Token(context.Background())
	if err != nil {
		t.Fatalf("Token after login: %v", err)
	}
	if tok.AccessToken != "codex-access" {
		t.Errorf("cached AccessToken = %q", tok.AccessToken)
	}
}

func TestCodexDeviceSource_RefreshesExpiredToken(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	ctx := context.Background()

	// Pre-populate store with an expired record plus account id.
	expired := CodexTokenRecord{
		AccessToken:      "stale",
		RefreshToken:     "refresh-old",
		ChatGPTAccountID: "acc-persisted",
		ExpiresAt:        time.Now().Add(-1 * time.Hour),
	}
	if err := saveCodexRecord(ctx, store, expired); err != nil {
		t.Fatalf("seed record: %v", err)
	}

	var refreshCalled atomic.Int32
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		if err := req.ParseForm(); err != nil {
			t.Fatalf("ParseForm: %v", err)
		}
		if req.Form.Get("grant_type") != "refresh_token" {
			t.Errorf("grant_type = %q, want refresh_token", req.Form.Get("grant_type"))
		}
		if req.Form.Get("refresh_token") != "refresh-old" {
			t.Errorf("refresh_token = %q", req.Form.Get("refresh_token"))
		}
		refreshCalled.Add(1)
		return jsonResponse(map[string]any{
			"access_token": "fresh",
			"expires_in":   3600,
		}), nil
	})}

	src := NewCodexDeviceSource(CodexDeviceConfig{
		Store:      store,
		HTTPClient: client,
		TokenURL:   "https://test.codex.local/token",
	})

	tok, err := src.Token(ctx)
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if refreshCalled.Load() != 1 {
		t.Errorf("refresh called %d times, want 1", refreshCalled.Load())
	}
	if tok.AccessToken != "fresh" {
		t.Errorf("AccessToken = %q", tok.AccessToken)
	}

	// Refresh response omitted both refresh_token and id_token; the
	// source must preserve the original values so the caller can keep
	// issuing requests with the correct ChatGPT-Account-Id header on
	// future refreshes.
	rec, err := LoadCodexRecord(ctx, store)
	if err != nil {
		t.Fatalf("LoadCodexRecord: %v", err)
	}
	if rec.RefreshToken != "refresh-old" {
		t.Errorf("RefreshToken = %q, want preserved refresh-old", rec.RefreshToken)
	}
	if rec.ChatGPTAccountID != "acc-persisted" {
		t.Errorf("ChatGPTAccountID = %q, want preserved acc-persisted", rec.ChatGPTAccountID)
	}
}

func TestExtractChatGPTAccountID(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name   string
		claims map[string]any
		want   string
	}{
		{
			name:   "top-level",
			claims: map[string]any{"chatgpt_account_id": "top"},
			want:   "top",
		},
		{
			name: "nested auth claim",
			claims: map[string]any{
				"https://api.openai.com/auth": map[string]any{
					"chatgpt_account_id": "nested",
				},
			},
			want: "nested",
		},
		{
			name: "falls back to first organization",
			claims: map[string]any{
				"organizations": []map[string]any{
					{"id": "org-first"},
					{"id": "org-second"},
				},
			},
			want: "org-first",
		},
		{
			name:   "no fields → empty",
			claims: map[string]any{"unrelated": "x"},
			want:   "",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			body, _ := json.Marshal(tc.claims)
			tok := "h." + base64.RawURLEncoding.EncodeToString(body) + ".s"
			if got := extractChatGPTAccountID(tok); got != tc.want {
				t.Errorf("extractChatGPTAccountID = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestCodexWebFlowSource_BeginComplete(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	ctx := context.Background()

	idTok := codexFakeIDToken(t, map[string]any{
		"organizations": []map[string]any{{"id": "org-web"}},
	})

	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		if req.URL.String() != "https://test.codex.local/token" {
			t.Fatalf("unexpected URL: %s", req.URL.String())
		}
		if err := req.ParseForm(); err != nil {
			t.Fatalf("ParseForm: %v", err)
		}
		if req.Form.Get("grant_type") != "authorization_code" {
			t.Errorf("grant_type = %q", req.Form.Get("grant_type"))
		}
		if req.Form.Get("client_id") != CodexDefaultClientID {
			t.Errorf("client_id = %q", req.Form.Get("client_id"))
		}
		if req.Form.Get("code") != "the-code" {
			t.Errorf("code = %q", req.Form.Get("code"))
		}
		if req.Form.Get("code_verifier") == "" {
			t.Errorf("code_verifier is empty")
		}
		return jsonResponse(map[string]any{
			"access_token":  "web-access",
			"refresh_token": "web-refresh",
			"id_token":      idTok,
			"expires_in":    3600,
		}), nil
	})}

	src := NewCodexWebFlowSource(CodexWebFlowConfig{
		Store:      store,
		HTTPClient: client,
		TokenURL:   "https://test.codex.local/token",
	})

	authURL, err := src.Begin(ctx, "http://localhost:1455/auth/callback")
	if err != nil {
		t.Fatalf("Begin: %v", err)
	}

	parsed, err := url.Parse(authURL)
	if err != nil {
		t.Fatalf("parse authURL: %v", err)
	}
	q := parsed.Query()
	if q.Get("client_id") != CodexDefaultClientID {
		t.Errorf("client_id param = %q", q.Get("client_id"))
	}
	if q.Get("code_challenge_method") != "S256" {
		t.Errorf("code_challenge_method = %q", q.Get("code_challenge_method"))
	}
	state := q.Get("state")
	if state == "" {
		t.Fatal("missing state param")
	}

	// Wrong state → rejected.
	if err := src.Complete(ctx, "not-the-state", "the-code"); err == nil {
		t.Error("expected state mismatch to fail")
	}

	// Correct state → record persisted with account id extracted.
	if err := src.Complete(ctx, state, "the-code"); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	rec, err := LoadCodexRecord(ctx, store)
	if err != nil {
		t.Fatalf("LoadCodexRecord: %v", err)
	}
	if rec.AccessToken != "web-access" {
		t.Errorf("AccessToken = %q", rec.AccessToken)
	}
	if rec.ChatGPTAccountID != "org-web" {
		t.Errorf("ChatGPTAccountID = %q, want org-web", rec.ChatGPTAccountID)
	}
}

func TestCodexDeviceSource_LogoutClearsRecord(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	ctx := context.Background()

	rec := CodexTokenRecord{AccessToken: "x", ExpiresAt: time.Now().Add(time.Hour)}
	if err := saveCodexRecord(ctx, store, rec); err != nil {
		t.Fatalf("seed: %v", err)
	}

	src := NewCodexDeviceSource(CodexDeviceConfig{Store: store})
	if err := src.Logout(ctx); err != nil {
		t.Fatalf("Logout: %v", err)
	}

	if _, err := LoadCodexRecord(ctx, store); err == nil {
		t.Error("expected LoadCodexRecord to fail after Logout")
	}
}

// compile-time sanity: the codex sources satisfy TokenSource. If this
// ever breaks the provider wiring will not compile either.
var (
	_ TokenSource = (*CodexDeviceSource)(nil)
	_ TokenSource = (*CodexWebFlowSource)(nil)
)
