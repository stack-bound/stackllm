package web

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/stack-bound/stackllm/agent"
	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/provider"
	"github.com/stack-bound/stackllm/session"
)

// noFlushWriter hides the Flusher implementation of the wrapped
// ResponseWriter so SSE setup fails deterministically.
type noFlushWriter struct {
	http.ResponseWriter
}

// errorDeleteStore wraps a SessionStore and fails every Delete. The
// stock InMemoryStore deletes idempotently, so exercising the
// handlers' delete-error branch needs a store that actually errors.
type errorDeleteStore struct {
	session.SessionStore
	err error
}

func (s errorDeleteStore) Delete(context.Context, string) error { return s.err }

func TestHasMessageContent(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		msg  conversation.Message
		want bool
	}{
		{name: "no blocks", msg: conversation.Message{}, want: false},
		{
			name: "text whitespace only",
			msg:  conversation.Message{Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "  \t\n"}}},
			want: false,
		},
		{
			name: "text with content",
			msg:  conversation.Message{Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hi"}}},
			want: true,
		},
		{
			name: "thinking with content",
			msg:  conversation.Message{Blocks: []conversation.Block{{Type: conversation.BlockThinking, Text: "hmm"}}},
			want: true,
		},
		{
			name: "thinking whitespace only",
			msg:  conversation.Message{Blocks: []conversation.Block{{Type: conversation.BlockThinking, Text: " "}}},
			want: false,
		},
		{
			name: "image with url",
			msg:  conversation.Message{Blocks: []conversation.Block{{Type: conversation.BlockImage, ImageURL: "https://img.example/x.png"}}},
			want: true,
		},
		{
			name: "image with inline data",
			msg:  conversation.Message{Blocks: []conversation.Block{{Type: conversation.BlockImage, ImageData: []byte{1, 2}}}},
			want: true,
		},
		{
			name: "image with neither url nor data",
			msg:  conversation.Message{Blocks: []conversation.Block{{Type: conversation.BlockImage}}},
			want: false,
		},
		{
			name: "tool use",
			msg:  conversation.Message{Blocks: []conversation.Block{{Type: conversation.BlockToolUse, ToolCallID: "c1", ToolName: "read"}}},
			want: true,
		},
		{
			name: "tool result",
			msg:  conversation.Message{Blocks: []conversation.Block{{Type: conversation.BlockToolResult, ToolCallID: "c1"}}},
			want: true,
		},
		{
			name: "redacted thinking",
			msg:  conversation.Message{Blocks: []conversation.Block{{Type: conversation.BlockRedactedThinking, RedactedData: []byte{9}}}},
			want: true,
		},
		{
			name: "empty text followed by tool use",
			msg: conversation.Message{Blocks: []conversation.Block{
				{Type: conversation.BlockText, Text: ""},
				{Type: conversation.BlockToolUse, ToolCallID: "c2"},
			}},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := hasMessageContent(tt.msg); got != tt.want {
				t.Errorf("hasMessageContent() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBlockToJSON(t *testing.T) {
	t.Parallel()

	imgData := []byte("png-bytes-here")
	redacted := []byte("opaque-reasoning")

	tests := []struct {
		name  string
		block conversation.Block
		want  map[string]any
	}{
		{
			name:  "text",
			block: conversation.Block{ID: "b1", Type: conversation.BlockText, Text: "hello"},
			want:  map[string]any{"id": "b1", "type": "text", "text": "hello"},
		},
		{
			name: "tool use",
			block: conversation.Block{
				ID: "b2", Type: conversation.BlockToolUse,
				ToolCallID: "call-1", ToolName: "read_file", ToolArgsJSON: `{"path":"x"}`,
			},
			want: map[string]any{
				"id": "b2", "type": "tool_use",
				"tool_call_id": "call-1", "tool_name": "read_file", "tool_args": `{"path":"x"}`,
			},
		},
		{
			name: "tool result error",
			block: conversation.Block{
				ID: "b3", Type: conversation.BlockToolResult,
				ToolCallID: "call-1", Text: "Error: no such file", ToolIsError: true,
			},
			want: map[string]any{
				"id": "b3", "type": "tool_result",
				"tool_call_id": "call-1", "text": "Error: no such file", "tool_is_error": true,
			},
		},
		{
			// Binary payloads must be reported as byte lengths, never
			// raw bytes, to keep SSE lines small.
			name: "inline image emits byte length",
			block: conversation.Block{
				ID: "b4", Type: conversation.BlockImage,
				MimeType: "image/png", ImageData: imgData,
			},
			want: map[string]any{
				"id": "b4", "type": "image",
				"mime_type": "image/png", "image_bytes": len(imgData),
			},
		},
		{
			name: "external image emits url",
			block: conversation.Block{
				ID: "b5", Type: conversation.BlockImage,
				MimeType: "image/jpeg", ImageURL: "https://img.example/a.jpg",
			},
			want: map[string]any{
				"id": "b5", "type": "image",
				"mime_type": "image/jpeg", "image_url": "https://img.example/a.jpg",
			},
		},
		{
			name: "redacted thinking emits byte length",
			block: conversation.Block{
				ID: "b6", Type: conversation.BlockRedactedThinking, RedactedData: redacted,
			},
			want: map[string]any{
				"id": "b6", "type": "redacted_thinking", "redacted_bytes": len(redacted),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := blockToJSON(tt.block)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("blockToJSON() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestChatRequest_UnmarshalJSON(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		body    string
		wantErr bool
		want    chatRequest
	}{
		{
			name: "legacy string message",
			body: `{"session_id":"s1","message":"hi there"}`,
			want: chatRequest{
				SessionID: "s1",
				Message: conversation.Message{
					Role:   conversation.RoleUser,
					Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hi there"}},
				},
			},
		},
		{
			name: "null message leaves zero message",
			body: `{"session_id":"s2","message":null}`,
			want: chatRequest{SessionID: "s2"},
		},
		{
			name: "missing message leaves zero message",
			body: `{"session_id":"s3"}`,
			want: chatRequest{SessionID: "s3"},
		},
		{
			name: "block message without role defaults to user",
			body: `{"message":{"blocks":[{"type":"text","text":"hi"}]}}`,
			want: chatRequest{
				Message: conversation.Message{
					Role:   conversation.RoleUser,
					Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hi"}},
				},
			},
		},
		{
			name: "block message keeps explicit role",
			body: `{"message":{"role":"assistant","blocks":[{"type":"text","text":"yo"}]}}`,
			want: chatRequest{
				Message: conversation.Message{
					Role:   conversation.RoleAssistant,
					Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "yo"}},
				},
			},
		},
		{
			name:    "message of wrong type errors",
			body:    `{"message":42}`,
			wantErr: true,
		},
		{
			name:    "invalid top-level json errors",
			body:    `{`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var got chatRequest
			err := json.Unmarshal([]byte(tt.body), &got)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected an error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("chatRequest = %#v, want %#v", got, tt.want)
			}
		})
	}
}

// sseDoneSessionID extracts the session_id carried by the done event
// in a recorded SSE stream, or "" when no done event was emitted.
func sseDoneSessionID(t *testing.T, body string) string {
	t.Helper()
	var sessionID string
	scanner := bufio.NewScanner(strings.NewReader(body))
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
	return sessionID
}

func TestHandler_Chat_ExistingSession_AppendsHistory(t *testing.T) {
	t.Parallel()

	events := append(textBlockEvents("second reply"), provider.Event{Type: provider.EventTypeDone})
	p := &mockProvider{responses: [][]provider.Event{events}}
	store := session.NewInMemoryStore()
	h := NewHandler(agent.New(p), store)

	sess := session.New()
	sess.AppendMessage(conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "first question"}},
	})
	if err := store.Save(context.Background(), sess); err != nil {
		t.Fatalf("seed save: %v", err)
	}

	body := strings.NewReader(`{"session_id":"` + sess.ID + `","message":{"role":"user","blocks":[{"type":"text","text":"second question"}]}}`)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/chat", body))
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d", w.Code)
	}

	if got := sseDoneSessionID(t, w.Body.String()); got != sess.ID {
		t.Fatalf("done session_id = %q, want existing id %q", got, sess.ID)
	}

	loaded, err := store.Load(context.Background(), sess.ID)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(loaded.Messages) != 3 {
		t.Fatalf("messages = %d, want 3 (prior user, new user, assistant)", len(loaded.Messages))
	}
	if got := loaded.Messages[0].TextContent(); got != "first question" {
		t.Errorf("history head = %q", got)
	}
	if got := loaded.Messages[1].TextContent(); got != "second question" {
		t.Errorf("new user message = %q", got)
	}
	if loaded.Messages[2].Role != conversation.RoleAssistant || loaded.Messages[2].TextContent() != "second reply" {
		t.Errorf("assistant message = %+v", loaded.Messages[2])
	}
}

func TestHandler_Chat_UnknownSessionID_CreatesFreshSession(t *testing.T) {
	t.Parallel()

	events := append(textBlockEvents("hi"), provider.Event{Type: provider.EventTypeDone})
	p := &mockProvider{responses: [][]provider.Event{events}}
	store := session.NewInMemoryStore()
	h := NewHandler(agent.New(p), store)

	body := strings.NewReader(`{"session_id":"does-not-exist","message":{"role":"user","blocks":[{"type":"text","text":"hello"}]}}`)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/chat", body))
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d", w.Code)
	}

	sessionID := sseDoneSessionID(t, w.Body.String())
	if sessionID == "" {
		t.Fatal("missing session_id in done event")
	}
	if sessionID == "does-not-exist" {
		t.Fatal("handler must mint a fresh session when the requested one is missing")
	}
	loaded, err := store.Load(context.Background(), sessionID)
	if err != nil {
		t.Fatalf("Load fresh session: %v", err)
	}
	if len(loaded.Messages) != 2 {
		t.Errorf("messages = %d, want 2", len(loaded.Messages))
	}
}

func TestHandler_Chat_ProviderError_EmitsSSEErrorAndPersists(t *testing.T) {
	t.Parallel()

	// A mockProvider with no scripted responses fails on the first
	// Complete call, which surfaces as an agent EventError.
	p := &mockProvider{}
	store := session.NewInMemoryStore()
	h := NewHandler(agent.New(p), store)

	sess := session.New()
	if err := store.Save(context.Background(), sess); err != nil {
		t.Fatalf("seed save: %v", err)
	}

	body := strings.NewReader(`{"session_id":"` + sess.ID + `","message":{"role":"user","blocks":[{"type":"text","text":"doomed"}]}}`)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/chat", body))
	if w.Code != http.StatusOK {
		t.Fatalf("SSE status = %d", w.Code)
	}

	// The stream must carry an error event with the provider's message.
	if !strings.Contains(w.Body.String(), "event: error") {
		t.Fatalf("expected error event, body = %q", w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "no more responses") {
		t.Errorf("error event should carry the provider error, body = %q", w.Body.String())
	}

	// The user message must still have been persisted on the error path.
	loaded, err := store.Load(context.Background(), sess.ID)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(loaded.Messages) != 1 || loaded.Messages[0].TextContent() != "doomed" {
		t.Errorf("persisted messages = %+v, want the user message", loaded.Messages)
	}
}

func TestHandler_Chat_StreamingUnsupported(t *testing.T) {
	t.Parallel()

	p := &mockProvider{}
	h := NewHandler(agent.New(p), session.NewInMemoryStore())

	rec := httptest.NewRecorder()
	body := strings.NewReader(`{"message":{"role":"user","blocks":[{"type":"text","text":"hi"}]}}`)
	h.ServeHTTP(noFlushWriter{rec}, httptest.NewRequest("POST", "/chat", body))

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "streaming not supported") {
		t.Errorf("body = %q", rec.Body.String())
	}
}

func TestHandler_DeleteSession_NotFound(t *testing.T) {
	t.Parallel()

	store := errorDeleteStore{
		SessionStore: session.NewInMemoryStore(),
		err:          errors.New("session not found"),
	}
	h := NewHandler(agent.New(&mockProvider{}), store)

	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("DELETE", "/sessions/nope", nil))

	if w.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", w.Code)
	}
	if !strings.Contains(w.Body.String(), "nope") {
		t.Errorf("error body should name the missing session, got %q", w.Body.String())
	}
}

func TestSSEWriter_NoFlusher(t *testing.T) {
	t.Parallel()

	if _, err := newSSEWriter(noFlushWriter{httptest.NewRecorder()}); err == nil {
		t.Fatal("expected error for a writer without Flusher support")
	}
}

func TestSSEWriter_MarshalError(t *testing.T) {
	t.Parallel()

	sse, err := newSSEWriter(httptest.NewRecorder())
	if err != nil {
		t.Fatalf("newSSEWriter: %v", err)
	}
	if err := sse.writeEvent("bad", make(chan int)); err == nil {
		t.Fatal("expected marshal error for unencodable payload")
	}
}
