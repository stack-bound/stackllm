package web

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/stack-bound/stackllm/agent"
	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/session"
)

// Handler exposes an agent over HTTP with SSE streaming.
type Handler struct {
	agent *agent.Agent
	store session.SessionStore
	mux   *http.ServeMux
}

// NewHandler creates a new HTTP handler for an agent.
func NewHandler(a *agent.Agent, store session.SessionStore) *Handler {
	h := &Handler{
		agent: a,
		store: store,
		mux:   http.NewServeMux(),
	}
	h.mux.HandleFunc("POST /chat", h.handleChat)
	h.mux.HandleFunc("GET /sessions/{id}", h.handleGetSession)
	h.mux.HandleFunc("DELETE /sessions/{id}", h.handleDeleteSession)
	return h
}

// ServeHTTP implements http.Handler.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.mux.ServeHTTP(w, r)
}

type chatRequest struct {
	SessionID string `json:"session_id"`
	Message   string `json:"message"`
}

func (h *Handler) handleChat(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
		return
	}

	if strings.TrimSpace(req.Message) == "" {
		http.Error(w, `{"error":"message is required"}`, http.StatusBadRequest)
		return
	}

	// Load or create session.
	ctx := r.Context()
	var sess *session.Session
	if req.SessionID != "" {
		var err error
		sess, err = h.store.Load(ctx, req.SessionID)
		if err != nil {
			sess = session.New()
		}
	} else {
		sess = session.New()
	}

	// Append user message.
	sess.AppendMessage(conversation.Message{
		Role:    conversation.RoleUser,
		Content: req.Message,
	})

	// Set up SSE writer.
	sse, err := newSSEWriter(w)
	if err != nil {
		http.Error(w, `{"error":"streaming not supported"}`, http.StatusInternalServerError)
		return
	}

	// Run agent.
	events, err := h.agent.Run(ctx, sess.Messages)
	if err != nil {
		sse.writeEvent("error", map[string]string{"message": err.Error()})
		return
	}

	for ev := range events {
		switch ev.Type {
		case agent.EventToken:
			sse.writeEvent("token", map[string]string{"delta": ev.Content})

		case agent.EventToolCall:
			sse.writeEvent("tool_call", map[string]string{
				"id":        ev.ToolCall.ID,
				"name":      ev.ToolCall.Name,
				"arguments": ev.ToolCall.Arguments,
			})

		case agent.EventToolResult:
			sse.writeEvent("tool_result", map[string]string{
				"result": ev.ToolResult,
			})

		case agent.EventComplete:
			sess.Messages = append([]conversation.Message(nil), ev.Messages...)
			h.store.Save(context.Background(), sess)
			sse.writeEvent("done", map[string]string{"session_id": sess.ID})

		case agent.EventError:
			if len(ev.Messages) > 0 {
				sess.Messages = append([]conversation.Message(nil), ev.Messages...)
				h.store.Save(context.Background(), sess)
			}
			sse.writeEvent("error", map[string]string{"message": ev.Err.Error()})
		}
	}
}

func (h *Handler) handleGetSession(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	sess, err := h.store.Load(r.Context(), id)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"session not found: %s"}`, id), http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(sess)
}

func (h *Handler) handleDeleteSession(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if err := h.store.Delete(r.Context(), id); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"session not found: %s"}`, id), http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
