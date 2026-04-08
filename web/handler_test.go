package web

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stack-bound/stackllm/agent"
	"github.com/stack-bound/stackllm/auth"
	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/provider"
	"github.com/stack-bound/stackllm/session"
)

type mockProvider struct {
	responses [][]provider.Event
	callIndex int
}

func (m *mockProvider) Complete(_ context.Context, _ provider.Request) (<-chan provider.Event, error) {
	ch := make(chan provider.Event, 64)
	if m.callIndex >= len(m.responses) {
		close(ch)
		return ch, fmt.Errorf("mock: no more responses")
	}
	events := m.responses[m.callIndex]
	m.callIndex++
	go func() {
		defer close(ch)
		for _, ev := range events {
			ch <- ev
		}
	}()
	return ch, nil
}

func (m *mockProvider) Models(_ context.Context) ([]string, error) { return nil, nil }

func TestHandler_Chat(t *testing.T) {
	t.Parallel()

	p := &mockProvider{
		responses: [][]provider.Event{
			{
				{Type: provider.EventTypeToken, Content: "Hello"},
				{Type: provider.EventTypeToken, Content: " there"},
				{Type: provider.EventTypeDone},
			},
		},
	}

	a := agent.New(p)
	store := session.NewInMemoryStore()
	h := NewHandler(a, store)

	body := strings.NewReader(`{"message":"hi"}`)
	req := httptest.NewRequest("POST", "/chat", body)
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	// Parse SSE events.
	scanner := bufio.NewScanner(strings.NewReader(w.Body.String()))
	var events []string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "event: ") {
			events = append(events, strings.TrimPrefix(line, "event: "))
		}
	}

	// Should have token events and a done event.
	hasToken := false
	hasDone := false
	for _, e := range events {
		if e == "token" {
			hasToken = true
		}
		if e == "done" {
			hasDone = true
		}
	}
	if !hasToken {
		t.Error("expected token events")
	}
	if !hasDone {
		t.Error("expected done event")
	}
}

func TestHandler_Chat_PersistsAssistantMessage(t *testing.T) {
	t.Parallel()

	p := &mockProvider{
		responses: [][]provider.Event{
			{
				{Type: provider.EventTypeToken, Content: "Hello"},
				{Type: provider.EventTypeDone},
			},
		},
	}

	a := agent.New(p)
	store := session.NewInMemoryStore()
	h := NewHandler(a, store)

	body := strings.NewReader(`{"message":"hi"}`)
	req := httptest.NewRequest("POST", "/chat", body)
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	scanner := bufio.NewScanner(strings.NewReader(w.Body.String()))
	var sessionID string
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var payload map[string]string
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &payload); err == nil && payload["session_id"] != "" {
			sessionID = payload["session_id"]
		}
	}

	if sessionID == "" {
		t.Fatal("expected session_id in SSE done event")
	}

	sess, err := store.Load(context.Background(), sessionID)
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}
	if len(sess.Messages) != 2 {
		t.Fatalf("messages len = %d, want 2", len(sess.Messages))
	}
	if sess.Messages[1].Role != conversation.RoleAssistant {
		t.Fatalf("assistant role = %q", sess.Messages[1].Role)
	}
	if sess.Messages[1].Content != "Hello" {
		t.Fatalf("assistant content = %q", sess.Messages[1].Content)
	}
}

func TestHandler_Chat_EmptyMessage(t *testing.T) {
	t.Parallel()

	p := &mockProvider{}
	a := agent.New(p)
	store := session.NewInMemoryStore()
	h := NewHandler(a, store)

	body := strings.NewReader(`{"message":"  "}`)
	req := httptest.NewRequest("POST", "/chat", body)
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestHandler_Chat_InvalidBody(t *testing.T) {
	t.Parallel()

	p := &mockProvider{}
	a := agent.New(p)
	store := session.NewInMemoryStore()
	h := NewHandler(a, store)

	body := strings.NewReader(`not json`)
	req := httptest.NewRequest("POST", "/chat", body)
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestHandler_GetSession(t *testing.T) {
	t.Parallel()

	store := session.NewInMemoryStore()
	sess := session.New()
	store.Save(context.Background(), sess)

	p := &mockProvider{}
	a := agent.New(p)
	h := NewHandler(a, store)

	req := httptest.NewRequest("GET", "/sessions/"+sess.ID, nil)
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var result map[string]any
	json.NewDecoder(w.Body).Decode(&result)
	if result["ID"] != sess.ID {
		t.Errorf("ID = %v, want %q", result["ID"], sess.ID)
	}
}

func TestHandler_GetSession_NotFound(t *testing.T) {
	t.Parallel()

	store := session.NewInMemoryStore()
	p := &mockProvider{}
	a := agent.New(p)
	h := NewHandler(a, store)

	req := httptest.NewRequest("GET", "/sessions/nonexistent", nil)
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", w.Code)
	}
}

func TestHandler_DeleteSession(t *testing.T) {
	t.Parallel()

	store := session.NewInMemoryStore()
	sess := session.New()
	store.Save(context.Background(), sess)

	p := &mockProvider{}
	a := agent.New(p)
	h := NewHandler(a, store)

	req := httptest.NewRequest("DELETE", "/sessions/"+sess.ID, nil)
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusNoContent {
		t.Errorf("status = %d, want 204", w.Code)
	}

	// Verify deleted.
	_, err := store.Load(context.Background(), sess.ID)
	if err == nil {
		t.Error("expected session to be deleted")
	}
}

func TestSSEWriter(t *testing.T) {
	t.Parallel()

	w := httptest.NewRecorder()
	sse, err := newSSEWriter(w)
	if err != nil {
		t.Fatalf("newSSEWriter error: %v", err)
	}

	sse.writeEvent("token", map[string]string{"delta": "hello"})

	body := w.Body.String()
	if !strings.Contains(body, "event: token") {
		t.Errorf("expected 'event: token' in body, got %q", body)
	}
	if !strings.Contains(body, `"delta":"hello"`) {
		t.Errorf("expected data in body, got %q", body)
	}
}

func TestAuthRoutes_Status(t *testing.T) {
	t.Parallel()

	routes := NewAuthRoutes(AuthRoutesConfig{})
	req := httptest.NewRequest("GET", "/auth/status", nil)
	w := httptest.NewRecorder()

	routes.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}
}

func TestAuthRoutes_CopilotStatus_NotConfigured(t *testing.T) {
	t.Parallel()

	routes := NewAuthRoutes(AuthRoutesConfig{})
	req := httptest.NewRequest("GET", "/auth/copilot/status", nil)
	w := httptest.NewRecorder()

	routes.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", w.Code)
	}
}

func TestAuthRoutes_CopilotStart_NotConfigured(t *testing.T) {
	t.Parallel()

	routes := NewAuthRoutes(AuthRoutesConfig{})
	req := httptest.NewRequest("GET", "/auth/copilot/start", nil)
	w := httptest.NewRecorder()

	routes.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", w.Code)
	}
}

func TestAuthRoutes_OpenAIStart(t *testing.T) {
	t.Parallel()

	routes := NewAuthRoutes(AuthRoutesConfig{
		OpenAI: auth.NewOpenAIWebFlowSource(auth.OpenAIWebFlowConfig{
			ClientID: "client-id",
			Store:    auth.NewMemoryStore(),
		}),
	})

	req := httptest.NewRequest("GET", "http://example.com/auth/openai/start", nil)
	w := httptest.NewRecorder()

	routes.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var payload map[string]string
	if err := json.NewDecoder(w.Body).Decode(&payload); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if payload["status"] != "redirect" {
		t.Fatalf("status payload = %q, want redirect", payload["status"])
	}
	if !strings.Contains(payload["redirect_url"], "auth0.openai.com/authorize") {
		t.Fatalf("redirect_url = %q", payload["redirect_url"])
	}
}

func TestAuthRoutes_OpenAIStart_NotConfigured(t *testing.T) {
	t.Parallel()

	routes := NewAuthRoutes(AuthRoutesConfig{})
	req := httptest.NewRequest("GET", "/auth/openai/start", nil)
	w := httptest.NewRecorder()

	routes.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", w.Code)
	}
}

func TestAuthRoutes_OpenAICallback_NotConfigured(t *testing.T) {
	t.Parallel()

	routes := NewAuthRoutes(AuthRoutesConfig{})
	req := httptest.NewRequest("GET", "/auth/openai/callback", nil)
	w := httptest.NewRecorder()

	routes.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", w.Code)
	}
}

func TestAuthRoutes_CopilotStatus_NotStarted(t *testing.T) {
	t.Parallel()

	store := auth.NewMemoryStore()
	src := auth.NewCopilotSource(auth.CopilotConfig{Store: store})
	routes := NewAuthRoutes(AuthRoutesConfig{Copilot: src})

	req := httptest.NewRequest("GET", "/auth/copilot/status", nil)
	w := httptest.NewRecorder()

	routes.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var result map[string]string
	json.NewDecoder(w.Body).Decode(&result)
	if result["status"] != "not_started" {
		t.Errorf("status = %q, want not_started", result["status"])
	}
}

func TestAuthRoutes_SetDeviceCode(t *testing.T) {
	t.Parallel()

	store := auth.NewMemoryStore()
	src := auth.NewCopilotSource(auth.CopilotConfig{Store: store})
	routes := NewAuthRoutes(AuthRoutesConfig{Copilot: src})

	routes.SetDeviceCode("ABCD-1234", "https://github.com/login/device")

	// Verify the code is stored.
	routes.mu.Lock()
	code := routes.pendingCode
	url := routes.pendingURL
	routes.mu.Unlock()

	if code != "ABCD-1234" {
		t.Errorf("pendingCode = %q, want ABCD-1234", code)
	}
	if url != "https://github.com/login/device" {
		t.Errorf("pendingURL = %q, want https://github.com/login/device", url)
	}
}
