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
	SessionID string               `json:"session_id"`
	Message   conversation.Message `json:"message"`
}

func (r *chatRequest) UnmarshalJSON(data []byte) error {
	var raw struct {
		SessionID string          `json:"session_id"`
		Message   json.RawMessage `json:"message"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	r.SessionID = raw.SessionID

	if len(raw.Message) == 0 || string(raw.Message) == "null" {
		return nil
	}

	var legacy string
	if err := json.Unmarshal(raw.Message, &legacy); err == nil {
		r.Message = conversation.Message{
			Role:   conversation.RoleUser,
			Blocks: []conversation.Block{{Type: conversation.BlockText, Text: legacy}},
		}
		return nil
	}

	if err := json.Unmarshal(raw.Message, &r.Message); err != nil {
		return err
	}
	if r.Message.Role == "" {
		r.Message.Role = conversation.RoleUser
	}
	return nil
}

func (h *Handler) handleChat(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
		return
	}

	if !hasMessageContent(req.Message) {
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

	req.Message.Role = conversation.RoleUser
	sess.AppendMessage(req.Message)

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
		case agent.EventBlockStart:
			sse.writeEvent("block_start", map[string]string{
				"block_type": string(ev.BlockType),
			})

		case agent.EventBlockDelta:
			sse.writeEvent("block_delta", map[string]string{
				"block_type": string(ev.BlockType),
				"delta":      ev.Content,
			})

		case agent.EventBlockEnd:
			payload := map[string]any{
				"block_type": string(ev.BlockType),
			}
			if ev.Block != nil {
				payload["block"] = blockToJSON(*ev.Block)
			}
			sse.writeEvent("block_end", payload)

		case agent.EventToolCall:
			_ = ev

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

func hasMessageContent(msg conversation.Message) bool {
	if len(msg.Blocks) == 0 {
		return false
	}
	for _, b := range msg.Blocks {
		switch b.Type {
		case conversation.BlockText, conversation.BlockThinking:
			if strings.TrimSpace(b.Text) != "" {
				return true
			}
		case conversation.BlockImage:
			if b.ImageURL != "" || len(b.ImageData) > 0 {
				return true
			}
		case conversation.BlockToolUse, conversation.BlockToolResult, conversation.BlockRedactedThinking:
			return true
		}
	}
	return false
}

// blockToJSON serialises a Block to a map for SSE / JSON output.
// Binary payloads (ImageData, RedactedData) are emitted as byte
// lengths rather than raw bytes to keep SSE lines small; callers that
// need the bytes can hit a dedicated artifact endpoint (future work).
func blockToJSON(b conversation.Block) map[string]any {
	out := map[string]any{
		"id":   b.ID,
		"type": string(b.Type),
	}
	if b.Text != "" {
		out["text"] = b.Text
	}
	if b.ToolCallID != "" {
		out["tool_call_id"] = b.ToolCallID
	}
	if b.ToolName != "" {
		out["tool_name"] = b.ToolName
	}
	if b.ToolArgsJSON != "" {
		out["tool_args"] = b.ToolArgsJSON
	}
	if b.ToolIsError {
		out["tool_is_error"] = true
	}
	if b.MimeType != "" {
		out["mime_type"] = b.MimeType
	}
	if b.ImageURL != "" {
		out["image_url"] = b.ImageURL
	}
	if len(b.ImageData) > 0 {
		out["image_bytes"] = len(b.ImageData)
	}
	if len(b.RedactedData) > 0 {
		out["redacted_bytes"] = len(b.RedactedData)
	}
	return out
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
