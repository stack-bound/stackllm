package web

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stack-bound/stackllm/agent"
	"github.com/stack-bound/stackllm/auth"
	"github.com/stack-bound/stackllm/config"
	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/profile"
	"github.com/stack-bound/stackllm/session"
)

// startChatUpstream returns an httptest server that answers
// /v1/chat/completions with a single-text-block SSE stream saying
// reply.
func startChatUpstream(t *testing.T, reply string) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		f := w.(http.Flusher)
		enc, _ := json.Marshal(map[string]any{
			"choices": []map[string]any{{"delta": map[string]any{"content": reply}}},
		})
		w.Write([]byte("data: " + string(enc) + "\n\n"))
		f.Flush()
		w.Write([]byte(`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}` + "\n\n"))
		f.Flush()
		w.Write([]byte("data: [DONE]\n\n"))
		f.Flush()
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

// loginAndSetDefault authenticates openai with a static key and sets
// openai/gpt-4o as the default via the public HTTP surface.
func loginAndSetDefault(t *testing.T, h *ManagedHandler) {
	t.Helper()
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/providers/openai/login", strings.NewReader(`{"key":"sk-x"}`)))
	if w.Code != http.StatusOK {
		t.Fatalf("login status = %d", w.Code)
	}
	w = httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/default", strings.NewReader(`{"provider":"openai","model":"gpt-4o"}`)))
	if w.Code != http.StatusOK {
		t.Fatalf("set default status = %d", w.Code)
	}
}

func TestManagedHandler_WithAgentOptions_AppliedToChatAgent(t *testing.T) {
	t.Parallel()

	srv := startChatUpstream(t, "hello from hook")
	mgr := newTestManager(t, profile.WithHTTPClient(redirectClient(srv)))

	var mu sync.Mutex
	var tokens []string
	h := NewManagedHandler(mgr, session.NewInMemoryStore(),
		WithAgentOptions(agent.WithHooks(agent.Hooks{
			OnToken: func(_ context.Context, delta string) {
				mu.Lock()
				tokens = append(tokens, delta)
				mu.Unlock()
			},
		})),
		// Deprecated no-op must still be accepted without changing
		// behaviour.
		WithOpenAIOAuthClientID("legacy-client-id"),
	)

	loginAndSetDefault(t, h)

	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/chat",
		strings.NewReader(`{"message":{"role":"user","blocks":[{"type":"text","text":"hi"}]}}`)))
	if w.Code != http.StatusOK {
		t.Fatalf("chat status = %d body = %s", w.Code, w.Body.String())
	}

	mu.Lock()
	joined := strings.Join(tokens, "")
	mu.Unlock()
	if joined != "hello from hook" {
		t.Errorf("hook saw %q, want the streamed reply — agent options were not applied", joined)
	}
}

func TestManagedHandler_DeleteSession(t *testing.T) {
	t.Parallel()

	store := session.NewInMemoryStore()
	h := NewManagedHandler(newTestManager(t), store)

	sess := session.New()
	sess.AppendMessage(conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "keep me?"}},
	})
	if err := store.Save(context.Background(), sess); err != nil {
		t.Fatalf("seed save: %v", err)
	}

	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("DELETE", "/sessions/"+sess.ID, nil))
	if w.Code != http.StatusNoContent {
		t.Fatalf("delete status = %d, want 204", w.Code)
	}

	if _, err := store.Load(context.Background(), sess.ID); err == nil {
		t.Error("session must be gone from the store after DELETE")
	}
}

func TestManagedHandler_DeleteSession_StoreErrorReturns404(t *testing.T) {
	t.Parallel()

	store := errorDeleteStore{
		SessionStore: session.NewInMemoryStore(),
		err:          errors.New("session not found"),
	}
	h := NewManagedHandler(newTestManager(t), store)

	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("DELETE", "/sessions/gone", nil))
	if w.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", w.Code)
	}
	var payload map[string]string
	json.Unmarshal(w.Body.Bytes(), &payload)
	if !strings.Contains(payload["error"], "gone") {
		t.Errorf("error payload = %q, should name the session id", payload["error"])
	}
}

func TestManagedHandler_GetSession_NotFound(t *testing.T) {
	t.Parallel()

	h := NewManagedHandler(newTestManager(t), session.NewInMemoryStore())
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("GET", "/sessions/missing", nil))
	if w.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", w.Code)
	}
}

func TestManagedHandler_SetDefault_Errors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		body       string
		wantStatus int
	}{
		{name: "invalid json", body: `{`, wantStatus: http.StatusBadRequest},
		{name: "missing model", body: `{"provider":"openai"}`, wantStatus: http.StatusBadRequest},
		{name: "unknown provider", body: `{"provider":"nope","model":"m"}`, wantStatus: http.StatusBadRequest},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			h := NewManagedHandler(newTestManager(t), session.NewInMemoryStore())
			w := httptest.NewRecorder()
			h.ServeHTTP(w, httptest.NewRequest("POST", "/default", strings.NewReader(tt.body)))
			if w.Code != tt.wantStatus {
				t.Fatalf("status = %d, want %d (body %s)", w.Code, tt.wantStatus, w.Body.String())
			}
			var payload map[string]string
			json.Unmarshal(w.Body.Bytes(), &payload)
			if payload["error"] == "" {
				t.Error("error responses must carry a JSON error message")
			}

			// A failed set must leave the default unset.
			gw := httptest.NewRecorder()
			h.ServeHTTP(gw, httptest.NewRequest("GET", "/default", nil))
			var def map[string]any
			json.Unmarshal(gw.Body.Bytes(), &def)
			if def["set"] != false {
				t.Errorf("default should remain unset after a rejected POST, got %v", def)
			}
		})
	}
}

func TestManagedHandler_SetDefault_TracksRecentModels(t *testing.T) {
	t.Parallel()

	mgr := newTestManager(t)
	h := NewManagedHandler(mgr, session.NewInMemoryStore())

	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/default",
		strings.NewReader(`{"provider":"openai","model":"gpt-4o"}`)))
	if w.Code != http.StatusOK {
		t.Fatalf("first set status = %d", w.Code)
	}
	w = httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/default",
		strings.NewReader(`{"provider":"ollama","model":"llama3","endpoint":"/responses"}`)))
	if w.Code != http.StatusOK {
		t.Fatalf("second set status = %d", w.Code)
	}

	// Setting a default must push onto the recent-models list,
	// most recent first, preserving the endpoint.
	recents, err := mgr.RecentModels(context.Background())
	if err != nil {
		t.Fatalf("RecentModels: %v", err)
	}
	if len(recents) != 2 {
		t.Fatalf("recents = %d entries, want 2: %+v", len(recents), recents)
	}
	if recents[0].Provider != "ollama" || recents[0].Model != "llama3" || recents[0].Endpoint != "/responses" {
		t.Errorf("most recent = %+v, want ollama/llama3 with /responses endpoint", recents[0])
	}
	if recents[1].Provider != "openai" || recents[1].Model != "gpt-4o" {
		t.Errorf("second recent = %+v, want openai/gpt-4o", recents[1])
	}

	// GET /default must reflect the latest choice including endpoint.
	w = httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("GET", "/default", nil))
	var def map[string]any
	json.Unmarshal(w.Body.Bytes(), &def)
	if def["provider"] != "ollama" || def["model"] != "llama3" || def["endpoint"] != "/responses" {
		t.Errorf("default = %+v", def)
	}
}

func TestManagedHandler_CopilotStart_UpstreamFailureReturns502(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	mux.HandleFunc("/login/device/code", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "gh down", http.StatusInternalServerError)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	mgr := newTestManager(t,
		profile.WithHTTPClient(redirectClient(srv)),
		profile.WithPollInterval(5*time.Millisecond),
	)
	h := NewManagedHandler(mgr, session.NewInMemoryStore())

	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/providers/copilot/login", nil))
	if w.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502 (body %s)", w.Code, w.Body.String())
	}
	var payload map[string]string
	json.Unmarshal(w.Body.Bytes(), &payload)
	if payload["error"] == "" {
		t.Error("502 must carry a JSON error message")
	}
}

func TestManagedHandler_OpenAIOAuthStart_UpstreamFailureReturns502(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	mux.HandleFunc("/api/accounts/deviceauth/usercode", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "oai down", http.StatusInternalServerError)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	mgr := newTestManager(t,
		profile.WithHTTPClient(redirectClient(srv)),
		profile.WithPollInterval(5*time.Millisecond),
	)
	h := NewManagedHandler(mgr, session.NewInMemoryStore())

	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/providers/openai/oauth/login", nil))
	if w.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502 (body %s)", w.Code, w.Body.String())
	}
}

func TestManagedHandler_CopilotStart_SecondCallReturnsExistingFlow(t *testing.T) {
	t.Parallel()

	authorise := make(chan struct{})
	srv := startCopilotFakeGitHub(t, authorise, "SAME-CODE", "gho_x")
	// Release the poll loop when the test ends so the background
	// goroutine exits promptly.
	t.Cleanup(func() { close(authorise) })

	mgr := newTestManager(t,
		profile.WithHTTPClient(redirectClient(srv)),
		profile.WithPollInterval(5*time.Millisecond),
	)
	h := NewManagedHandler(mgr, session.NewInMemoryStore())

	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/providers/copilot/login", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("first start status = %d", w.Code)
	}
	var first map[string]any
	json.Unmarshal(w.Body.Bytes(), &first)
	if first["user_code"] != "SAME-CODE" || first["status"] != "pending" {
		t.Fatalf("first payload = %v", first)
	}

	// Second login while pending must return the same flow, not mint
	// a fresh device code.
	w = httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/providers/copilot/login", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("second start status = %d", w.Code)
	}
	var second map[string]any
	json.Unmarshal(w.Body.Bytes(), &second)
	if second["user_code"] != "SAME-CODE" || second["status"] != "pending" {
		t.Fatalf("second payload = %v, want the pending flow echoed back", second)
	}
}

func TestManagedHandler_OpenAIOAuthStart_SecondCallReturnsExistingFlow(t *testing.T) {
	t.Parallel()

	authorise := make(chan struct{})
	mux := http.NewServeMux()
	mux.HandleFunc("/api/accounts/deviceauth/usercode", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"device_auth_id": "dev-pending",
			"user_code":      "OAI-SAME",
			"interval":       0,
			"expires_in":     60,
		})
	})
	mux.HandleFunc("/api/accounts/deviceauth/token", func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-authorise:
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"authorization_code": "c",
				"code_verifier":      "v",
			})
		default:
			http.Error(w, "pending", http.StatusForbidden)
		}
	})
	mux.HandleFunc("/oauth/token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"access_token": "a", "expires_in": 3600})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(authorise) })

	mgr := newTestManager(t,
		profile.WithHTTPClient(redirectClient(srv)),
		profile.WithPollInterval(5*time.Millisecond),
	)
	h := NewManagedHandler(mgr, session.NewInMemoryStore())

	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/providers/openai/oauth/login", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("first start status = %d", w.Code)
	}

	w = httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/providers/openai/oauth/login", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("second start status = %d", w.Code)
	}
	var second map[string]any
	json.Unmarshal(w.Body.Bytes(), &second)
	if second["user_code"] != "OAI-SAME" || second["status"] != "pending" {
		t.Fatalf("second payload = %v, want the pending flow echoed back", second)
	}
}

func TestManagedHandler_OpenAIOAuthStatus_NotStarted(t *testing.T) {
	t.Parallel()

	h := NewManagedHandler(newTestManager(t), session.NewInMemoryStore())
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("GET", "/providers/openai/oauth/status", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d", w.Code)
	}
	var payload map[string]string
	json.Unmarshal(w.Body.Bytes(), &payload)
	if payload["status"] != "not_started" {
		t.Errorf("status = %q, want not_started", payload["status"])
	}
}

func TestManagedHandler_Logout_UnknownProvider(t *testing.T) {
	t.Parallel()

	h := NewManagedHandler(newTestManager(t), session.NewInMemoryStore())
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/providers/bogus/logout", nil))
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
	var payload map[string]string
	json.Unmarshal(w.Body.Bytes(), &payload)
	if !strings.Contains(payload["error"], "bogus") {
		t.Errorf("error = %q, should name the unknown provider", payload["error"])
	}
}

func TestManagedHandler_Chat_RequestValidation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		body string
	}{
		{name: "invalid json", body: `not json`},
		{name: "empty message", body: `{"message":{"role":"user","blocks":[{"type":"text","text":"   "}]}}`},
		{name: "null message", body: `{"message":null}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			h := NewManagedHandler(newTestManager(t), session.NewInMemoryStore())
			w := httptest.NewRecorder()
			h.ServeHTTP(w, httptest.NewRequest("POST", "/chat", strings.NewReader(tt.body)))
			if w.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400 (body %s)", w.Code, w.Body.String())
			}
		})
	}
}

func TestManagedHandler_Chat_ProviderLoadFailureReturns502(t *testing.T) {
	t.Parallel()

	// Default set but no credentials stored: building the provider
	// must fail and surface as 502.
	h := NewManagedHandler(newTestManager(t), session.NewInMemoryStore())
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/default",
		strings.NewReader(`{"provider":"openai","model":"gpt-4o"}`)))
	if w.Code != http.StatusOK {
		t.Fatalf("set default status = %d", w.Code)
	}

	w = httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/chat",
		strings.NewReader(`{"message":{"role":"user","blocks":[{"type":"text","text":"hi"}]}}`)))
	if w.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502 (body %s)", w.Code, w.Body.String())
	}
	var payload map[string]string
	json.Unmarshal(w.Body.Bytes(), &payload)
	if payload["error"] == "" {
		t.Error("502 must carry a JSON error message")
	}
}

func TestManagedHandler_Chat_ReusesExistingSession(t *testing.T) {
	t.Parallel()

	srv := startChatUpstream(t, "reply")
	mgr := newTestManager(t, profile.WithHTTPClient(redirectClient(srv)))
	store := session.NewInMemoryStore()
	h := NewManagedHandler(mgr, store)
	loginAndSetDefault(t, h)

	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/chat",
		strings.NewReader(`{"message":{"role":"user","blocks":[{"type":"text","text":"first"}]}}`)))
	if w.Code != http.StatusOK {
		t.Fatalf("first chat status = %d", w.Code)
	}
	sessionID := sseDoneSessionID(t, w.Body.String())
	if sessionID == "" {
		t.Fatal("first chat done event missing session_id")
	}

	w = httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/chat",
		strings.NewReader(`{"session_id":"`+sessionID+`","message":{"role":"user","blocks":[{"type":"text","text":"second"}]}}`)))
	if w.Code != http.StatusOK {
		t.Fatalf("second chat status = %d", w.Code)
	}
	if got := sseDoneSessionID(t, w.Body.String()); got != sessionID {
		t.Fatalf("second chat session_id = %q, want %q", got, sessionID)
	}

	sess, err := store.Load(context.Background(), sessionID)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(sess.Messages) != 4 {
		t.Fatalf("messages = %d, want 4 (user/assistant twice)", len(sess.Messages))
	}
	if sess.Messages[2].TextContent() != "second" {
		t.Errorf("third message = %q, want the second user turn", sess.Messages[2].TextContent())
	}
}

func TestManagedHandler_Chat_UpstreamErrorEmitsSSEErrorAndPersists(t *testing.T) {
	t.Parallel()

	// 400 responses are not retried by the provider, so the agent
	// surfaces an EventError immediately.
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":{"message":"model overloaded"}}`, http.StatusBadRequest)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	mgr := newTestManager(t, profile.WithHTTPClient(redirectClient(srv)))
	store := session.NewInMemoryStore()
	h := NewManagedHandler(mgr, store)
	loginAndSetDefault(t, h)

	sess := session.New()
	if err := store.Save(context.Background(), sess); err != nil {
		t.Fatalf("seed save: %v", err)
	}

	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/chat",
		strings.NewReader(`{"session_id":"`+sess.ID+`","message":{"role":"user","blocks":[{"type":"text","text":"doomed"}]}}`)))
	if w.Code != http.StatusOK {
		t.Fatalf("SSE status = %d", w.Code)
	}
	if !strings.Contains(w.Body.String(), "event: error") {
		t.Fatalf("expected error event, body = %q", w.Body.String())
	}

	// The user's message must survive the failed run.
	loaded, err := store.Load(context.Background(), sess.ID)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(loaded.Messages) != 1 || loaded.Messages[0].TextContent() != "doomed" {
		t.Errorf("persisted messages = %+v, want the user message", loaded.Messages)
	}
}

func TestManagedHandler_Chat_StreamingUnsupported(t *testing.T) {
	t.Parallel()

	srv := startChatUpstream(t, "unused")
	mgr := newTestManager(t, profile.WithHTTPClient(redirectClient(srv)))
	h := NewManagedHandler(mgr, session.NewInMemoryStore())
	loginAndSetDefault(t, h)

	rec := httptest.NewRecorder()
	h.ServeHTTP(noFlushWriter{rec}, httptest.NewRequest("POST", "/chat",
		strings.NewReader(`{"message":{"role":"user","blocks":[{"type":"text","text":"hi"}]}}`)))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "streaming not supported") {
		t.Errorf("body = %q", rec.Body.String())
	}
}

// newCorruptConfigManager builds a Manager whose config store points
// at an unparseable file, so every config load fails.
func newCorruptConfigManager(t *testing.T) *profile.Manager {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte("{not json"), 0o600); err != nil {
		t.Fatalf("write corrupt config: %v", err)
	}
	return profile.New(
		profile.WithAuthStore(auth.NewMemoryStore()),
		profile.WithConfigStore(&config.Store{Path: path}),
	)
}

func TestManagedHandler_CorruptConfig_SurfacesErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		method     string
		path       string
		body       string
		wantStatus int
	}{
		{name: "list providers", method: "GET", path: "/providers", wantStatus: http.StatusInternalServerError},
		{name: "get default", method: "GET", path: "/default", wantStatus: http.StatusInternalServerError},
		{name: "list all models", method: "GET", path: "/models", wantStatus: http.StatusInternalServerError},
		{name: "set default", method: "POST", path: "/default", body: `{"provider":"openai","model":"gpt-4o"}`, wantStatus: http.StatusBadRequest},
		{name: "ollama login", method: "POST", path: "/providers/ollama/login", body: `{}`, wantStatus: http.StatusBadRequest},
		{name: "chat", method: "POST", path: "/chat", body: `{"message":{"role":"user","blocks":[{"type":"text","text":"hi"}]}}`, wantStatus: http.StatusInternalServerError},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			h := NewManagedHandler(newCorruptConfigManager(t), session.NewInMemoryStore())
			var body *strings.Reader
			if tt.body != "" {
				body = strings.NewReader(tt.body)
			} else {
				body = strings.NewReader("")
			}
			w := httptest.NewRecorder()
			h.ServeHTTP(w, httptest.NewRequest(tt.method, tt.path, body))
			if w.Code != tt.wantStatus {
				t.Fatalf("status = %d, want %d (body %s)", w.Code, tt.wantStatus, w.Body.String())
			}
			var payload map[string]string
			json.Unmarshal(w.Body.Bytes(), &payload)
			if payload["error"] == "" {
				t.Error("error responses must carry a JSON error message")
			}
		})
	}
}

func TestManagedHandler_APIKeyLogin_InvalidJSON(t *testing.T) {
	t.Parallel()

	h := NewManagedHandler(newTestManager(t), session.NewInMemoryStore())
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/providers/openai/login", strings.NewReader(`{`)))
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
	var payload map[string]string
	json.Unmarshal(w.Body.Bytes(), &payload)
	if payload["error"] == "" {
		t.Error("400 must carry a JSON error message")
	}
}

func TestManagedHandler_CopilotStatus_ReportsFlowError(t *testing.T) {
	t.Parallel()

	// Device code is issued, but the user denies the authorisation:
	// the poll loop hits a fatal error and the status endpoint must
	// surface it in the payload.
	mux := http.NewServeMux()
	mux.HandleFunc("/login/device/code", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"device_code":      "dev",
			"user_code":        "DENY-1",
			"verification_uri": "https://github.com/login/device",
			"interval":         0,
			"expires_in":       60,
		})
	})
	mux.HandleFunc("/login/oauth/access_token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"error": "access_denied"})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	mgr := newTestManager(t,
		profile.WithHTTPClient(redirectClient(srv)),
		profile.WithPollInterval(5*time.Millisecond),
	)
	h := NewManagedHandler(mgr, session.NewInMemoryStore())

	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/providers/copilot/login", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("start status = %d body = %s", w.Code, w.Body.String())
	}

	deadline := time.Now().Add(2 * time.Second)
	var status, errMsg string
	for time.Now().Before(deadline) {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest("GET", "/providers/copilot/status", nil))
		var p map[string]any
		json.Unmarshal(w.Body.Bytes(), &p)
		if s, _ := p["status"].(string); s == "error" {
			status = s
			errMsg, _ = p["error"].(string)
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if status != "error" {
		t.Fatalf("status = %q, want error", status)
	}
	if !strings.Contains(errMsg, "access_denied") {
		t.Errorf("error = %q, should carry the upstream denial", errMsg)
	}
}
