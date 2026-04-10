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

func (m *mockProvider) Models(_ context.Context) ([]provider.ModelMeta, error) { return nil, nil }

// textBlockEvents returns the canonical block event triplet a mock
// provider emits for a single text block with the given content.
func textBlockEvents(text string) []provider.Event {
	blk := conversation.Block{Type: conversation.BlockText, Text: text}
	return []provider.Event{
		{Type: provider.EventTypeBlockStart, BlockType: conversation.BlockText},
		{Type: provider.EventTypeBlockDelta, BlockType: conversation.BlockText, Content: text},
		{Type: provider.EventTypeBlockEnd, BlockType: conversation.BlockText, Block: &blk},
	}
}

func TestHandler_Chat(t *testing.T) {
	t.Parallel()

	events := append(textBlockEvents("Hello there"), provider.Event{Type: provider.EventTypeDone})
	p := &mockProvider{responses: [][]provider.Event{events}}

	a := agent.New(p)
	store := session.NewInMemoryStore()
	h := NewHandler(a, store)

	body := strings.NewReader(`{"message":{"role":"user","blocks":[{"type":"text","text":"hi"}]}}`)
	req := httptest.NewRequest("POST", "/chat", body)
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	// Parse SSE events.
	scanner := bufio.NewScanner(strings.NewReader(w.Body.String()))
	var eventNames []string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "event: ") {
			eventNames = append(eventNames, strings.TrimPrefix(line, "event: "))
		}
	}

	// Should have block events and a done event.
	hasStart := false
	hasDelta := false
	hasEnd := false
	hasDone := false
	for _, e := range eventNames {
		switch e {
		case "block_start":
			hasStart = true
		case "block_delta":
			hasDelta = true
		case "block_end":
			hasEnd = true
		case "done":
			hasDone = true
		}
	}
	if !hasStart {
		t.Error("expected block_start events")
	}
	if !hasDelta {
		t.Error("expected block_delta events")
	}
	if !hasEnd {
		t.Error("expected block_end events")
	}
	if !hasDone {
		t.Error("expected done event")
	}
}

func TestHandler_Chat_PersistsAssistantMessage(t *testing.T) {
	t.Parallel()

	events := append(textBlockEvents("Hello"), provider.Event{Type: provider.EventTypeDone})
	p := &mockProvider{responses: [][]provider.Event{events}}

	a := agent.New(p)
	store := session.NewInMemoryStore()
	h := NewHandler(a, store)

	body := strings.NewReader(`{"message":{"role":"user","blocks":[{"type":"text","text":"hi"}]}}`)
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
	if got := sess.Messages[1].TextContent(); got != "Hello" {
		t.Fatalf("assistant text = %q", got)
	}
	if len(sess.Messages[1].Blocks) != 1 || sess.Messages[1].Blocks[0].Type != conversation.BlockText {
		t.Fatalf("assistant blocks = %+v", sess.Messages[1].Blocks)
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
	if result["id"] != sess.ID {
		t.Errorf("id = %v, want %q", result["id"], sess.ID)
	}
}

func TestHandler_Chat_LegacyStringMessageStillAccepted(t *testing.T) {
	t.Parallel()

	events := append(textBlockEvents("Hello"), provider.Event{Type: provider.EventTypeDone})
	p := &mockProvider{responses: [][]provider.Event{events}}

	a := agent.New(p)
	store := session.NewInMemoryStore()
	h := NewHandler(a, store)

	req := httptest.NewRequest("POST", "/chat", strings.NewReader(`{"message":"hi"}`))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
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
