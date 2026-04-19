package web

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stack-bound/stackllm/auth"
	"github.com/stack-bound/stackllm/config"
	"github.com/stack-bound/stackllm/profile"
	"github.com/stack-bound/stackllm/provider"
	"github.com/stack-bound/stackllm/session"
)

// redirectTransport rewrites all request URLs to the target httptest
// server so profile.Manager's provider / auth code exercises the mock
// instead of real upstream hosts. Mirrors the helper in profile_test.go.
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
	return &http.Client{Transport: &redirectTransport{target: u, inner: http.DefaultTransport}}
}

func newTestManager(t *testing.T, opts ...profile.Option) *profile.Manager {
	t.Helper()
	base := []profile.Option{
		profile.WithAuthStore(auth.NewMemoryStore()),
		profile.WithConfigStore(&config.Store{Path: filepath.Join(t.TempDir(), "config.json")}),
	}
	return profile.New(append(base, opts...)...)
}

func TestManagedHandler_SaveOpenAIKey(t *testing.T) {
	t.Parallel()

	mgr := newTestManager(t)
	h := NewManagedHandler(mgr, session.NewInMemoryStore())

	req := httptest.NewRequest("POST", "/providers/openai/login", strings.NewReader(`{"key":"sk-web-key"}`))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", w.Code, w.Body.String())
	}

	// Verify persisted via Status — the auth path is opaque from the caller's side
	// so we probe the observable contract rather than the store key.
	sreq := httptest.NewRequest("GET", "/providers", nil)
	sw := httptest.NewRecorder()
	h.ServeHTTP(sw, sreq)

	var payload struct {
		Providers []profile.ProviderStatus `json:"providers"`
	}
	if err := json.Unmarshal(sw.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode status: %v", err)
	}
	var openaiAuthed bool
	for _, s := range payload.Providers {
		if s.Name == profile.ProviderOpenAI {
			openaiAuthed = s.Authenticated
		}
	}
	if !openaiAuthed {
		t.Error("openai should be reported as authenticated after /providers/openai/login")
	}
}

func TestManagedHandler_SaveAPIKey_EmptyRejected(t *testing.T) {
	t.Parallel()
	h := NewManagedHandler(newTestManager(t), session.NewInMemoryStore())

	req := httptest.NewRequest("POST", "/providers/openai/login", strings.NewReader(`{"key":""}`))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestManagedHandler_SaveOllamaURL(t *testing.T) {
	t.Parallel()
	h := NewManagedHandler(newTestManager(t), session.NewInMemoryStore())

	req := httptest.NewRequest("POST", "/providers/ollama/login", strings.NewReader(`{"base_url":"http://ollama:11434"}`))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d", w.Code)
	}
}

func TestManagedHandler_Logout(t *testing.T) {
	t.Parallel()

	mgr := newTestManager(t)
	h := NewManagedHandler(mgr, session.NewInMemoryStore())

	// Login first.
	req := httptest.NewRequest("POST", "/providers/openai/login", strings.NewReader(`{"key":"sk-1"}`))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	// Now log out.
	req = httptest.NewRequest("POST", "/providers/openai/logout", nil)
	w = httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("logout status = %d", w.Code)
	}

	// Status should reflect the logout.
	req = httptest.NewRequest("GET", "/providers", nil)
	w = httptest.NewRecorder()
	h.ServeHTTP(w, req)

	var payload struct {
		Providers []profile.ProviderStatus `json:"providers"`
	}
	json.Unmarshal(w.Body.Bytes(), &payload)
	for _, s := range payload.Providers {
		if s.Name == profile.ProviderOpenAI && s.Authenticated {
			t.Error("openai should not be authenticated after logout")
		}
	}
}

func TestManagedHandler_CopilotDeviceFlow(t *testing.T) {
	t.Parallel()

	authorise := make(chan struct{})
	mux := http.NewServeMux()
	mux.HandleFunc("/login/device/code", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"device_code":      "dev",
			"user_code":        "WEB-1234",
			"verification_uri": "https://github.com/login/device",
			"interval":         0,
			"expires_in":       60,
		})
	})
	mux.HandleFunc("/login/oauth/access_token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		select {
		case <-authorise:
			json.NewEncoder(w).Encode(map[string]any{"access_token": "gho_web"})
		default:
			json.NewEncoder(w).Encode(map[string]any{"error": "authorization_pending"})
		}
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	mgr := newTestManager(t,
		profile.WithHTTPClient(redirectClient(srv)),
		profile.WithPollInterval(5*time.Millisecond),
	)
	h := NewManagedHandler(mgr, session.NewInMemoryStore())

	// Start the flow.
	req := httptest.NewRequest("POST", "/providers/copilot/login", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("start status = %d body = %s", w.Code, w.Body.String())
	}
	var startPayload map[string]any
	json.Unmarshal(w.Body.Bytes(), &startPayload)
	if startPayload["user_code"] != "WEB-1234" {
		t.Fatalf("user_code = %v", startPayload["user_code"])
	}
	if startPayload["status"] != "pending" {
		t.Fatalf("status = %v, want pending", startPayload["status"])
	}

	// Status endpoint should echo the same pending info.
	req = httptest.NewRequest("GET", "/providers/copilot/status", nil)
	w = httptest.NewRecorder()
	h.ServeHTTP(w, req)
	var statusPayload map[string]any
	json.Unmarshal(w.Body.Bytes(), &statusPayload)
	if statusPayload["status"] != "pending" {
		t.Fatalf("status poll status = %v, want pending", statusPayload["status"])
	}

	// Complete the flow.
	close(authorise)

	deadline := time.Now().Add(2 * time.Second)
	var finalStatus string
	for time.Now().Before(deadline) {
		req := httptest.NewRequest("GET", "/providers/copilot/status", nil)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, req)
		var p map[string]any
		json.Unmarshal(w.Body.Bytes(), &p)
		if s, _ := p["status"].(string); s == "authenticated" {
			finalStatus = s
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if finalStatus != "authenticated" {
		t.Fatalf("final status = %q", finalStatus)
	}
}

func TestManagedHandler_OpenAIOAuth_DisabledByDefault(t *testing.T) {
	t.Parallel()
	h := NewManagedHandler(newTestManager(t), session.NewInMemoryStore())

	req := httptest.NewRequest("POST", "/providers/openai/oauth/login", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusNotFound {
		t.Errorf("start status = %d, want 404 when no client ID configured", w.Code)
	}

	req = httptest.NewRequest("GET", "/providers/openai/oauth/status", nil)
	w = httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusNotFound {
		t.Errorf("status status = %d, want 404 when no client ID configured", w.Code)
	}
}

func TestManagedHandler_OpenAIOAuth_ReadyFlag(t *testing.T) {
	t.Parallel()

	// Without client ID, the /providers payload should signal that
	// OAuth is not available so the UI can hide the button.
	h := NewManagedHandler(newTestManager(t), session.NewInMemoryStore())
	req := httptest.NewRequest("GET", "/providers", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	var payload map[string]any
	json.Unmarshal(w.Body.Bytes(), &payload)
	if payload["openai_oauth_ready"] != false {
		t.Errorf("openai_oauth_ready = %v, want false", payload["openai_oauth_ready"])
	}

	// With client ID, it should flip to true.
	h2 := NewManagedHandler(newTestManager(t), session.NewInMemoryStore(),
		WithOpenAIOAuthClientID("my-client"))
	req = httptest.NewRequest("GET", "/providers", nil)
	w = httptest.NewRecorder()
	h2.ServeHTTP(w, req)
	payload = nil
	json.Unmarshal(w.Body.Bytes(), &payload)
	if payload["openai_oauth_ready"] != true {
		t.Errorf("openai_oauth_ready = %v, want true", payload["openai_oauth_ready"])
	}
}

func TestManagedHandler_OpenAIOAuth_DeviceFlow(t *testing.T) {
	t.Parallel()

	authorise := make(chan struct{})
	mux := http.NewServeMux()
	mux.HandleFunc("/oauth/device/code", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"device_code":      "dev",
			"user_code":        "OAI-WEB-1",
			"verification_uri": "https://auth0.openai.com/activate",
			"interval":         0,
			"expires_in":       60,
		})
	})
	mux.HandleFunc("/oauth/token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		select {
		case <-authorise:
			json.NewEncoder(w).Encode(map[string]any{
				"access_token": "oai-access",
				"expires_in":   3600,
			})
		default:
			json.NewEncoder(w).Encode(map[string]any{"error": "authorization_pending"})
		}
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	mgr := newTestManager(t,
		profile.WithHTTPClient(redirectClient(srv)),
		profile.WithPollInterval(5*time.Millisecond),
	)
	h := NewManagedHandler(mgr, session.NewInMemoryStore(),
		WithOpenAIOAuthClientID("test-client"))

	req := httptest.NewRequest("POST", "/providers/openai/oauth/login", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("start status = %d body = %s", w.Code, w.Body.String())
	}
	var start map[string]any
	json.Unmarshal(w.Body.Bytes(), &start)
	if start["user_code"] != "OAI-WEB-1" {
		t.Fatalf("user_code = %v", start["user_code"])
	}
	if start["status"] != "pending" {
		t.Fatalf("status = %v", start["status"])
	}

	// Status endpoint should mirror the pending flow.
	req = httptest.NewRequest("GET", "/providers/openai/oauth/status", nil)
	w = httptest.NewRecorder()
	h.ServeHTTP(w, req)
	var poll map[string]any
	json.Unmarshal(w.Body.Bytes(), &poll)
	if poll["status"] != "pending" {
		t.Fatalf("status poll = %v", poll["status"])
	}

	close(authorise)

	deadline := time.Now().Add(2 * time.Second)
	var finalStatus string
	for time.Now().Before(deadline) {
		req := httptest.NewRequest("GET", "/providers/openai/oauth/status", nil)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, req)
		var p map[string]any
		json.Unmarshal(w.Body.Bytes(), &p)
		if s, _ := p["status"].(string); s == "authenticated" {
			finalStatus = s
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if finalStatus != "authenticated" {
		t.Fatalf("final status = %q", finalStatus)
	}

	// After success the /providers list must report openai as
	// authenticated, because BeginOpenAIDeviceLogin mirrors the
	// access token into the API-key slot.
	req = httptest.NewRequest("GET", "/providers", nil)
	w = httptest.NewRecorder()
	h.ServeHTTP(w, req)
	var listing struct {
		Providers []profile.ProviderStatus `json:"providers"`
	}
	json.Unmarshal(w.Body.Bytes(), &listing)
	authed := false
	for _, p := range listing.Providers {
		if p.Name == profile.ProviderOpenAI {
			authed = p.Authenticated
		}
	}
	if !authed {
		t.Error("openai should be reported authenticated after OAuth device flow")
	}
}

func TestManagedHandler_Logout_CancelsInFlightCopilotFlow(t *testing.T) {
	t.Parallel()

	// Reproduce the race: start a Copilot device flow, log out
	// while the goroutine is still polling, then authorise
	// upstream. Without the cancel-and-wait in logout, the goroutine
	// would store fresh credentials after the delete. With the fix,
	// the store must stay empty.
	authorise := make(chan struct{})
	mux := http.NewServeMux()
	mux.HandleFunc("/login/device/code", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"device_code":      "dev",
			"user_code":        "RACE-1",
			"verification_uri": "https://github.com/login/device",
			"interval":         0,
			"expires_in":       60,
		})
	})
	mux.HandleFunc("/login/oauth/access_token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		select {
		case <-authorise:
			json.NewEncoder(w).Encode(map[string]any{"access_token": "gho_should_not_land"})
		default:
			json.NewEncoder(w).Encode(map[string]any{"error": "authorization_pending"})
		}
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	authStore := auth.NewMemoryStore()
	mgr := profile.New(
		profile.WithAuthStore(authStore),
		profile.WithConfigStore(&config.Store{Path: filepath.Join(t.TempDir(), "c.json")}),
		profile.WithHTTPClient(redirectClient(srv)),
		profile.WithPollInterval(5*time.Millisecond),
	)
	h := NewManagedHandler(mgr, session.NewInMemoryStore())

	// Start the flow.
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/providers/copilot/login", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("start status = %d", w.Code)
	}

	// Log out while the background goroutine is still polling.
	w = httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/providers/copilot/logout", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("logout status = %d", w.Code)
	}

	// Release the upstream to grant the token. If the goroutine
	// still races past the cancel, it would save here.
	close(authorise)

	// Give the upstream plenty of time to be hit; the goroutine
	// should have exited already.
	time.Sleep(100 * time.Millisecond)

	if _, err := authStore.Load(context.Background(), "copilot_github_token"); err == nil {
		t.Error("copilot token should remain cleared after logout — background flow raced past logout")
	}
}

func TestManagedHandler_Logout_CancelsInFlightOpenAIOAuth(t *testing.T) {
	t.Parallel()

	authorise := make(chan struct{})
	mux := http.NewServeMux()
	mux.HandleFunc("/oauth/device/code", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"device_code":      "dev",
			"user_code":        "RACE-2",
			"verification_uri": "https://auth0.openai.com/activate",
			"interval":         0,
			"expires_in":       60,
		})
	})
	mux.HandleFunc("/oauth/token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		select {
		case <-authorise:
			json.NewEncoder(w).Encode(map[string]any{"access_token": "oai-should-not-land", "expires_in": 3600})
		default:
			json.NewEncoder(w).Encode(map[string]any{"error": "authorization_pending"})
		}
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	authStore := auth.NewMemoryStore()
	mgr := profile.New(
		profile.WithAuthStore(authStore),
		profile.WithConfigStore(&config.Store{Path: filepath.Join(t.TempDir(), "c.json")}),
		profile.WithHTTPClient(redirectClient(srv)),
		profile.WithPollInterval(5*time.Millisecond),
	)
	h := NewManagedHandler(mgr, session.NewInMemoryStore(),
		WithOpenAIOAuthClientID("test-client"))

	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/providers/openai/oauth/login", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("start status = %d", w.Code)
	}

	w = httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/providers/openai/logout", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("logout status = %d", w.Code)
	}

	close(authorise)
	time.Sleep(100 * time.Millisecond)

	if _, err := authStore.Load(context.Background(), "openai_api_key"); err == nil {
		t.Error("openai api-key slot should remain cleared after logout")
	}
	if _, err := authStore.Load(context.Background(), "openai_token"); err == nil {
		t.Error("openai oauth record should remain cleared after logout")
	}
}

func TestManagedHandler_CopilotStatus_NotStarted(t *testing.T) {
	t.Parallel()
	h := NewManagedHandler(newTestManager(t), session.NewInMemoryStore())

	req := httptest.NewRequest("GET", "/providers/copilot/status", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d", w.Code)
	}
	var payload map[string]string
	json.Unmarshal(w.Body.Bytes(), &payload)
	if payload["status"] != "not_started" {
		t.Errorf("status = %q, want not_started", payload["status"])
	}
}

func TestManagedHandler_ListModels(t *testing.T) {
	t.Parallel()

	// Mock provider /models endpoints for openai.
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"id": "gpt-4o"},
				{"id": "gpt-4o-mini"},
			},
		})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	mgr := newTestManager(t, profile.WithHTTPClient(redirectClient(srv)))
	h := NewManagedHandler(mgr, session.NewInMemoryStore())

	// Authenticate openai first so ListAllModels queries it.
	req := httptest.NewRequest("POST", "/providers/openai/login", strings.NewReader(`{"key":"sk-x"}`))
	h.ServeHTTP(httptest.NewRecorder(), req)

	req = httptest.NewRequest("GET", "/models", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}

	var payload struct {
		Models []map[string]any `json:"models"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(payload.Models) != 2 {
		t.Fatalf("got %d models, want 2", len(payload.Models))
	}
	// Entries should carry the composite id so UIs can render them directly.
	ids := make(map[string]bool)
	for _, m := range payload.Models {
		ids[m["id"].(string)] = true
	}
	if !ids["openai/gpt-4o"] || !ids["openai/gpt-4o-mini"] {
		t.Errorf("expected both openai/gpt-4o and openai/gpt-4o-mini, got %v", ids)
	}
}

func TestManagedHandler_DefaultGetSet(t *testing.T) {
	t.Parallel()
	h := NewManagedHandler(newTestManager(t), session.NewInMemoryStore())

	// Unset initially.
	req := httptest.NewRequest("GET", "/default", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	var payload map[string]any
	json.Unmarshal(w.Body.Bytes(), &payload)
	if payload["set"] != false {
		t.Errorf("fresh default set = %v, want false", payload["set"])
	}

	// Set it.
	req = httptest.NewRequest("POST", "/default", strings.NewReader(`{"provider":"openai","model":"gpt-4o"}`))
	w = httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("set default status = %d, body = %s", w.Code, w.Body.String())
	}

	// Read back.
	req = httptest.NewRequest("GET", "/default", nil)
	w = httptest.NewRecorder()
	h.ServeHTTP(w, req)
	json.Unmarshal(w.Body.Bytes(), &payload)
	if payload["set"] != true {
		t.Error("default should be set")
	}
	if payload["provider"] != "openai" || payload["model"] != "gpt-4o" {
		t.Errorf("default = %+v", payload)
	}
}

func TestManagedHandler_Chat_RequiresDefault(t *testing.T) {
	t.Parallel()
	h := NewManagedHandler(newTestManager(t), session.NewInMemoryStore())

	req := httptest.NewRequest("POST", "/chat", strings.NewReader(`{"message":{"role":"user","blocks":[{"type":"text","text":"hi"}]}}`))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusConflict {
		t.Fatalf("status = %d, want 409", w.Code)
	}
}

func TestManagedHandler_Chat_UsesDefault(t *testing.T) {
	t.Parallel()

	// Upstream server plays a tiny chat completions stream.
	var hits int
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		f := w.(http.Flusher)

		writeSSE := func(payload string) {
			w.Write([]byte("data: " + payload + "\n\n"))
			f.Flush()
		}
		writeSSE(`{"choices":[{"delta":{"content":"hello"}}]}`)
		writeSSE(`{"choices":[{"delta":{},"finish_reason":"stop"}]}`)
		writeSSE(`[DONE]`)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	mgr := newTestManager(t, profile.WithHTTPClient(redirectClient(srv)))
	h := NewManagedHandler(mgr, session.NewInMemoryStore())

	// Authenticate and set default.
	h.ServeHTTP(httptest.NewRecorder(),
		httptest.NewRequest("POST", "/providers/openai/login", strings.NewReader(`{"key":"sk-x"}`)))
	h.ServeHTTP(httptest.NewRecorder(),
		httptest.NewRequest("POST", "/default", strings.NewReader(`{"provider":"openai","model":"gpt-4o"}`)))

	body := `{"message":{"role":"user","blocks":[{"type":"text","text":"hi"}]}}`
	req := httptest.NewRequest("POST", "/chat", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if hits == 0 {
		t.Fatal("upstream /v1/chat/completions was never called")
	}

	// SSE should have block events + done.
	scanner := bufio.NewScanner(strings.NewReader(w.Body.String()))
	var events []string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "event: ") {
			events = append(events, strings.TrimPrefix(line, "event: "))
		}
	}
	hasDone := false
	for _, e := range events {
		if e == "done" {
			hasDone = true
		}
	}
	if !hasDone {
		t.Errorf("expected done event, got %v", events)
	}
}

func TestManagedHandler_Chat_PersistsAndLoadsSession(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		f := w.(http.Flusher)
		w.Write([]byte(`data: {"choices":[{"delta":{"content":"reply"}}]}` + "\n\n"))
		f.Flush()
		w.Write([]byte(`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}` + "\n\n"))
		f.Flush()
		w.Write([]byte("data: [DONE]\n\n"))
		f.Flush()
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	mgr := newTestManager(t, profile.WithHTTPClient(redirectClient(srv)))
	store := session.NewInMemoryStore()
	h := NewManagedHandler(mgr, store)

	h.ServeHTTP(httptest.NewRecorder(),
		httptest.NewRequest("POST", "/providers/openai/login", strings.NewReader(`{"key":"sk-x"}`)))
	h.ServeHTTP(httptest.NewRecorder(),
		httptest.NewRequest("POST", "/default", strings.NewReader(`{"provider":"openai","model":"gpt-4o"}`)))

	req := httptest.NewRequest("POST", "/chat", strings.NewReader(`{"message":{"role":"user","blocks":[{"type":"text","text":"hi"}]}}`))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	// Pull the session_id out of the done event.
	var sessionID string
	scanner := bufio.NewScanner(strings.NewReader(w.Body.String()))
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var p map[string]string
		json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &p)
		if p["session_id"] != "" {
			sessionID = p["session_id"]
		}
	}
	if sessionID == "" {
		t.Fatal("missing session_id")
	}

	sess, err := store.Load(context.Background(), sessionID)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(sess.Messages) != 2 {
		t.Fatalf("messages = %d, want 2", len(sess.Messages))
	}
	if sess.Model != "openai/gpt-4o" {
		t.Errorf("session.Model = %q, want openai/gpt-4o", sess.Model)
	}
	if sess.Messages[1].TextContent() != "reply" {
		t.Errorf("assistant text = %q", sess.Messages[1].TextContent())
	}

	// GET /sessions/{id} should round-trip.
	req = httptest.NewRequest("GET", "/sessions/"+sessionID, nil)
	w = httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("GET /sessions/{id} status = %d", w.Code)
	}
}

func TestManagedHandler_ListProviderModels_PreservesEndpoint(t *testing.T) {
	t.Parallel()

	// Upstream mock returns a Copilot-shaped /models response in
	// which one model is only reachable via /responses. The web
	// handler must carry that Endpoint through to the wire payload
	// or clients persisting the selection will silently fall back
	// to /chat/completions.
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"id": "gpt-4o", "supported_endpoints": []string{"/chat/completions"}},
				{"id": "gpt-codex", "supported_endpoints": []string{"/responses"}},
			},
		})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	mgr := newTestManager(t, profile.WithHTTPClient(redirectClient(srv)))
	h := NewManagedHandler(mgr, session.NewInMemoryStore())

	h.ServeHTTP(httptest.NewRecorder(),
		httptest.NewRequest("POST", "/providers/openai/login", strings.NewReader(`{"key":"sk-x"}`)))

	req := httptest.NewRequest("GET", "/models/openai", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}

	var payload struct {
		Models []map[string]any `json:"models"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode: %v", err)
	}
	var codexEndpoint string
	for _, m := range payload.Models {
		if m["model"] == "gpt-codex" {
			codexEndpoint, _ = m["endpoint"].(string)
		}
	}
	if codexEndpoint != "/responses" {
		t.Errorf("gpt-codex endpoint = %q, want /responses", codexEndpoint)
	}
}

func TestManagedHandler_ListProviderModels_Unauthenticated(t *testing.T) {
	t.Parallel()
	h := NewManagedHandler(newTestManager(t), session.NewInMemoryStore())

	req := httptest.NewRequest("GET", "/models/openai", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

// compile-time check: provider.Event / conversation types are correctly imported.
var _ = provider.EventTypeDone
