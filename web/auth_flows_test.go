package web

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/stack-bound/stackllm/auth"
)

// startCopilotFakeGitHub returns an httptest server mimicking GitHub's
// device-code and access-token endpoints. The access-token endpoint
// answers "authorization_pending" until the authorise channel is
// closed, after which it grants ghToken.
func startCopilotFakeGitHub(t *testing.T, authorise <-chan struct{}, userCode, ghToken string) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/login/device/code", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"device_code":      "dev",
			"user_code":        userCode,
			"verification_uri": "https://github.com/login/device",
			"interval":         0,
			"expires_in":       60,
		})
	})
	mux.HandleFunc("/login/oauth/access_token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		select {
		case <-authorise:
			json.NewEncoder(w).Encode(map[string]any{"access_token": ghToken})
		default:
			json.NewEncoder(w).Encode(map[string]any{"error": "authorization_pending"})
		}
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func TestAuthRoutes_CopilotDeviceFlow_EndToEnd(t *testing.T) {
	t.Parallel()

	authorise := make(chan struct{})
	srv := startCopilotFakeGitHub(t, authorise, "AUTH-42", "gho_authflow")

	store := auth.NewMemoryStore()
	var routes *AuthRoutes
	src := auth.NewCopilotSource(auth.CopilotConfig{
		Store:          store,
		PollInterval:   5 * time.Millisecond,
		DeviceCodeURL:  srv.URL + "/login/device/code",
		AccessTokenURL: srv.URL + "/login/oauth/access_token",
		OnDeviceCode:   func(code, verifyURL string) { routes.SetDeviceCode(code, verifyURL) },
	})
	routes = NewAuthRoutes(AuthRoutesConfig{Copilot: src})

	// Kick off the flow.
	w := httptest.NewRecorder()
	routes.ServeHTTP(w, httptest.NewRequest("GET", "/auth/copilot/start", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("start status = %d body = %s", w.Code, w.Body.String())
	}

	// The device code is captured asynchronously by OnDeviceCode; poll
	// /auth/copilot/start until it surfaces. While the flow is pending
	// this exercises the "already in progress" branch, which must echo
	// the existing code rather than mint a new flow.
	deadline := time.Now().Add(2 * time.Second)
	var code, verify string
	for time.Now().Before(deadline) {
		w := httptest.NewRecorder()
		routes.ServeHTTP(w, httptest.NewRequest("GET", "/auth/copilot/start", nil))
		var p map[string]string
		if err := json.Unmarshal(w.Body.Bytes(), &p); err != nil {
			t.Fatalf("decode start payload: %v", err)
		}
		if p["user_code"] != "" {
			code, verify = p["user_code"], p["verify_url"]
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if code != "AUTH-42" {
		t.Fatalf("user_code = %q, want AUTH-42", code)
	}
	if verify != "https://github.com/login/device" {
		t.Fatalf("verify_url = %q", verify)
	}

	// Flow has not been authorised yet — status must be pending.
	w = httptest.NewRecorder()
	routes.ServeHTTP(w, httptest.NewRequest("GET", "/auth/copilot/status", nil))
	var statusPayload map[string]string
	json.Unmarshal(w.Body.Bytes(), &statusPayload)
	if statusPayload["status"] != "pending" {
		t.Fatalf("status = %q, want pending", statusPayload["status"])
	}

	// Authorise upstream and wait for the flow to complete.
	close(authorise)
	var finalStatus string
	deadline = time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		w := httptest.NewRecorder()
		routes.ServeHTTP(w, httptest.NewRequest("GET", "/auth/copilot/status", nil))
		var p map[string]string
		json.Unmarshal(w.Body.Bytes(), &p)
		if p["status"] == "authenticated" {
			finalStatus = p["status"]
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if finalStatus != "authenticated" {
		t.Fatalf("final status = %q, want authenticated", finalStatus)
	}

	// Behaviour, not structure: the GitHub token must actually have
	// been persisted so later provider calls can use it.
	tok, err := store.Load(context.Background(), "copilot_github_token")
	if err != nil {
		t.Fatalf("stored github token: %v", err)
	}
	if tok != "gho_authflow" {
		t.Errorf("stored token = %q, want gho_authflow", tok)
	}
}

func TestAuthRoutes_CopilotStart_LoginErrorSurfacedViaStatus(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	mux.HandleFunc("/login/device/code", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	store := auth.NewMemoryStore()
	var routes *AuthRoutes
	src := auth.NewCopilotSource(auth.CopilotConfig{
		Store:          store,
		PollInterval:   5 * time.Millisecond,
		DeviceCodeURL:  srv.URL + "/login/device/code",
		AccessTokenURL: srv.URL + "/login/oauth/access_token",
		OnDeviceCode:   func(code, verifyURL string) { routes.SetDeviceCode(code, verifyURL) },
	})
	routes = NewAuthRoutes(AuthRoutesConfig{Copilot: src})

	w := httptest.NewRecorder()
	routes.ServeHTTP(w, httptest.NewRequest("GET", "/auth/copilot/start", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("start status = %d", w.Code)
	}

	deadline := time.Now().Add(2 * time.Second)
	var errStatus, errMsg string
	for time.Now().Before(deadline) {
		w := httptest.NewRecorder()
		routes.ServeHTTP(w, httptest.NewRequest("GET", "/auth/copilot/status", nil))
		var p map[string]string
		json.Unmarshal(w.Body.Bytes(), &p)
		if p["status"] == "error" {
			errStatus, errMsg = p["status"], p["error"]
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if errStatus != "error" {
		t.Fatalf("status = %q, want error", errStatus)
	}
	if errMsg == "" {
		t.Error("error status must carry a non-empty error message")
	}

	// The failed flow must not have persisted any token.
	if _, err := store.Load(context.Background(), "copilot_github_token"); err == nil {
		t.Error("no token should be stored after a failed device flow")
	}
}

// beginOpenAIWebFlow drives GET /auth/openai/start and returns the
// state parameter baked into the returned authorization URL along
// with the redirect_uri the routes advertised.
func beginOpenAIWebFlow(t *testing.T, routes *AuthRoutes, target string, header http.Header) (state, redirectURI string) {
	t.Helper()
	req := httptest.NewRequest("GET", target, nil)
	for k, vs := range header {
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}
	w := httptest.NewRecorder()
	routes.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("start status = %d body = %s", w.Code, w.Body.String())
	}
	var payload map[string]string
	if err := json.NewDecoder(w.Body).Decode(&payload); err != nil {
		t.Fatalf("decode start payload: %v", err)
	}
	u, err := url.Parse(payload["redirect_url"])
	if err != nil {
		t.Fatalf("parse redirect_url %q: %v", payload["redirect_url"], err)
	}
	return u.Query().Get("state"), u.Query().Get("redirect_uri")
}

func TestAuthRoutes_OpenAICallback_Success(t *testing.T) {
	t.Parallel()

	const expiresIn = 3600
	mux := http.NewServeMux()
	mux.HandleFunc("/oauth/token", func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Errorf("parse token form: %v", err)
		}
		if got := r.Form.Get("grant_type"); got != "authorization_code" {
			t.Errorf("grant_type = %q, want authorization_code", got)
		}
		if got := r.Form.Get("code"); got != "cb-code" {
			t.Errorf("code = %q, want cb-code", got)
		}
		if r.Form.Get("code_verifier") == "" {
			t.Error("token exchange must include the PKCE code_verifier")
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "web-access",
			"refresh_token": "web-refresh",
			"expires_in":    expiresIn,
		})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	store := auth.NewMemoryStore()
	src := auth.NewOpenAIWebFlowSource(auth.OpenAIWebFlowConfig{
		ClientID: "client-id",
		Store:    store,
		TokenURL: srv.URL + "/oauth/token",
	})
	routes := NewAuthRoutes(AuthRoutesConfig{OpenAI: src})

	before := time.Now()
	state, redirectURI := beginOpenAIWebFlow(t, routes, "http://example.com/auth/openai/start", nil)
	if state == "" {
		t.Fatal("authorization URL carries no state parameter")
	}
	if redirectURI != "http://example.com/auth/openai/callback" {
		t.Fatalf("redirect_uri = %q", redirectURI)
	}

	cb := "/auth/openai/callback?state=" + url.QueryEscape(state) + "&code=cb-code"
	w := httptest.NewRecorder()
	routes.ServeHTTP(w, httptest.NewRequest("GET", cb, nil))
	if w.Code != http.StatusOK {
		t.Fatalf("callback status = %d body = %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "Authentication successful") {
		t.Errorf("callback body = %q", w.Body.String())
	}

	// The token record must round-trip through the store with its
	// expiry honoured (expires_in → absolute expires_at).
	raw, err := store.Load(context.Background(), "openai_web_token")
	if err != nil {
		t.Fatalf("load persisted token record: %v", err)
	}
	var rec struct {
		AccessToken  string    `json:"access_token"`
		RefreshToken string    `json:"refresh_token"`
		ExpiresAt    time.Time `json:"expires_at"`
	}
	if err := json.Unmarshal([]byte(raw), &rec); err != nil {
		t.Fatalf("unmarshal token record: %v", err)
	}
	if rec.AccessToken != "web-access" {
		t.Errorf("access_token = %q, want web-access", rec.AccessToken)
	}
	if rec.RefreshToken != "web-refresh" {
		t.Errorf("refresh_token = %q, want web-refresh", rec.RefreshToken)
	}
	if rec.ExpiresAt.Before(before) {
		t.Errorf("expires_at %v must not be before flow start %v", rec.ExpiresAt, before)
	}
	if max := before.Add(expiresIn*time.Second + time.Minute); rec.ExpiresAt.After(max) {
		t.Errorf("expires_at %v exceeds expires_in-derived bound %v", rec.ExpiresAt, max)
	}
}

func TestAuthRoutes_OpenAICallback_ErrorBranches(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		query      string
		wantStatus int
		wantBody   []string
	}{
		{
			name:       "upstream error param",
			query:      "?error=access_denied&error_description=user+said+no",
			wantStatus: http.StatusBadRequest,
			wantBody:   []string{"access_denied", "user said no"},
		},
		{
			name:       "missing code",
			query:      "",
			wantStatus: http.StatusBadRequest,
			wantBody:   []string{"no authorization code"},
		},
		{
			name:       "no flow in progress",
			query:      "?code=x&state=y",
			wantStatus: http.StatusBadRequest,
			wantBody:   []string{"no flow in progress"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			routes := NewAuthRoutes(AuthRoutesConfig{
				OpenAI: auth.NewOpenAIWebFlowSource(auth.OpenAIWebFlowConfig{
					ClientID: "client-id",
					Store:    auth.NewMemoryStore(),
				}),
			})
			w := httptest.NewRecorder()
			routes.ServeHTTP(w, httptest.NewRequest("GET", "/auth/openai/callback"+tt.query, nil))
			if w.Code != tt.wantStatus {
				t.Fatalf("status = %d, want %d (body %s)", w.Code, tt.wantStatus, w.Body.String())
			}
			for _, want := range tt.wantBody {
				if !strings.Contains(w.Body.String(), want) {
					t.Errorf("body %q missing %q", w.Body.String(), want)
				}
			}
		})
	}
}

func TestAuthRoutes_OpenAICallback_StateMismatch(t *testing.T) {
	t.Parallel()

	store := auth.NewMemoryStore()
	src := auth.NewOpenAIWebFlowSource(auth.OpenAIWebFlowConfig{
		ClientID: "client-id",
		Store:    store,
	})
	routes := NewAuthRoutes(AuthRoutesConfig{OpenAI: src})

	state, _ := beginOpenAIWebFlow(t, routes, "http://example.com/auth/openai/start", nil)
	if state == "" {
		t.Fatal("no state issued")
	}

	w := httptest.NewRecorder()
	routes.ServeHTTP(w, httptest.NewRequest("GET", "/auth/openai/callback?state=not-the-state&code=cb-code", nil))
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
	if !strings.Contains(w.Body.String(), "state mismatch") {
		t.Errorf("body = %q, want state mismatch error", w.Body.String())
	}
	// A rejected callback must not persist a token.
	if _, err := store.Load(context.Background(), "openai_web_token"); err == nil {
		t.Error("no token record should be stored after a state mismatch")
	}
}

func TestAuthRoutes_OpenAIRedirectURI_SchemeSelection(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		target string
		header http.Header
		want   string
	}{
		{
			name:   "plain http",
			target: "http://example.com/auth/openai/start",
			want:   "http://example.com/auth/openai/callback",
		},
		{
			name:   "tls request",
			target: "https://secure.example.com/auth/openai/start",
			want:   "https://secure.example.com/auth/openai/callback",
		},
		{
			name:   "x-forwarded-proto overrides",
			target: "http://proxied.example.com/auth/openai/start",
			header: http.Header{"X-Forwarded-Proto": []string{"https"}},
			want:   "https://proxied.example.com/auth/openai/callback",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			routes := NewAuthRoutes(AuthRoutesConfig{
				OpenAI: auth.NewOpenAIWebFlowSource(auth.OpenAIWebFlowConfig{
					ClientID: "client-id",
					Store:    auth.NewMemoryStore(),
				}),
			})
			_, redirectURI := beginOpenAIWebFlow(t, routes, tt.target, tt.header)
			if redirectURI != tt.want {
				t.Errorf("redirect_uri = %q, want %q", redirectURI, tt.want)
			}
		})
	}
}
