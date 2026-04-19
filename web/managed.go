package web

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"sync"
	"time"

	"github.com/stack-bound/stackllm/agent"
	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/profile"
	"github.com/stack-bound/stackllm/session"
)

// ManagedHandler is a web adapter that wires a profile.Manager to
// HTTP endpoints so that a browser-based UI can drive the full
// stackllm lifecycle — authenticate providers, browse models, choose
// a default, and chat — without any TUI.
//
// Endpoints (all under the handler's root mount point):
//
//	GET    /providers                       — auth status for every provider
//	POST   /providers/openai/login          — body {"key":"..."} (API key)
//	POST   /providers/openai/oauth/login    — starts OpenAI Codex device flow (ChatGPT sign-in)
//	GET    /providers/openai/oauth/status   — polls in-progress Codex device flow
//	POST   /providers/gemini/login          — body {"key":"..."}
//	POST   /providers/ollama/login          — body {"base_url":"..."} (optional)
//	POST   /providers/copilot/login         — starts GitHub device flow, returns code
//	GET    /providers/copilot/status        — polls in-progress Copilot device flow
//	POST   /providers/{name}/logout         — clears credentials for provider
//	GET    /models                          — merged model list (all authenticated providers)
//	GET    /models/{provider}               — models for a single authenticated provider
//	GET    /default                         — current default {provider,model,endpoint}
//	POST   /default                         — set default; body {"provider":"...","model":"...","endpoint":"..."}
//	POST   /chat                            — SSE chat using current default provider/model
//	GET    /sessions/{id}                   — retrieve a session as JSON
//	DELETE /sessions/{id}                   — delete a session
//
// Chat requests fail with HTTP 409 when no default model is set; the
// UI should drive the user through /providers + /models + /default
// first. Each chat request builds a fresh provider from the current
// default so the UI can switch models without restarting the server.
type ManagedHandler struct {
	mgr       *profile.Manager
	store     session.SessionStore
	agentOpts []agent.Option
	mux       *http.ServeMux

	// Active device flows. At most one per provider runs at a time
	// so the UI can call the status endpoint without carrying a flow
	// ID. Starting a new flow while one is pending surfaces the
	// existing code so the user's open tab stays valid.
	//
	// The browser/PKCE flow is intentionally NOT exposed here: it
	// requires the OAuth callback to land on the same machine as
	// the user's browser, which is only true for a local dev
	// deployment. Remote-hosted web UIs should use the device flow
	// (which works regardless of where the server lives) or the CLI
	// helper in examples/login.
	mu         sync.Mutex
	copilotFlw *profile.DeviceFlow
	openaiFlw  *profile.DeviceFlow
}

// ManagedOption configures a ManagedHandler.
type ManagedOption func(*ManagedHandler)

// WithAgentOptions applies agent options (tools, max steps, hooks,
// temperature, etc.) to every chat agent built by the handler.
func WithAgentOptions(opts ...agent.Option) ManagedOption {
	return func(h *ManagedHandler) { h.agentOpts = append(h.agentOpts, opts...) }
}

// WithOpenAIOAuthClientID is retained for backwards compatibility
// but is now a no-op: the ManagedHandler always enables the OpenAI
// OAuth endpoints and drives them through the Codex CLI's public
// OAuth client ID. Embedders that want to use their own registered
// OpenAI OAuth app should call profile.Manager.BeginOpenAIDeviceLogin
// directly with their client ID and skip the ManagedHandler's
// /oauth and /pkce routes.
//
// Deprecated: codex-flow sign-in requires no client ID.
func WithOpenAIOAuthClientID(_ string) ManagedOption {
	return func(*ManagedHandler) {}
}

// NewManagedHandler wires a profile.Manager and session store into a
// web handler.
func NewManagedHandler(mgr *profile.Manager, store session.SessionStore, opts ...ManagedOption) *ManagedHandler {
	h := &ManagedHandler{
		mgr:   mgr,
		store: store,
		mux:   http.NewServeMux(),
	}
	for _, opt := range opts {
		opt(h)
	}
	h.mux.HandleFunc("GET /providers", h.handleListProviders)
	h.mux.HandleFunc("POST /providers/openai/login", h.handleAPIKeyLogin(profile.ProviderOpenAI))
	h.mux.HandleFunc("POST /providers/openai/oauth/login", h.handleOpenAIOAuthStart)
	h.mux.HandleFunc("GET /providers/openai/oauth/status", h.handleOpenAIOAuthStatus)
	h.mux.HandleFunc("POST /providers/gemini/login", h.handleAPIKeyLogin(profile.ProviderGemini))
	h.mux.HandleFunc("POST /providers/ollama/login", h.handleOllamaLogin)
	h.mux.HandleFunc("POST /providers/copilot/login", h.handleCopilotStart)
	h.mux.HandleFunc("GET /providers/copilot/status", h.handleCopilotStatus)
	h.mux.HandleFunc("POST /providers/{name}/logout", h.handleLogout)
	h.mux.HandleFunc("GET /models", h.handleListAllModels)
	h.mux.HandleFunc("GET /models/{provider}", h.handleListProviderModels)
	h.mux.HandleFunc("GET /default", h.handleGetDefault)
	h.mux.HandleFunc("POST /default", h.handleSetDefault)
	h.mux.HandleFunc("POST /chat", h.handleChat)
	h.mux.HandleFunc("GET /sessions/{id}", h.handleGetSession)
	h.mux.HandleFunc("DELETE /sessions/{id}", h.handleDeleteSession)
	return h
}

// ServeHTTP implements http.Handler.
func (h *ManagedHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.mux.ServeHTTP(w, r)
}

// ---- Providers ----

func (h *ManagedHandler) handleListProviders(w http.ResponseWriter, r *http.Request) {
	statuses, err := h.mgr.Status(r.Context())
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, err)
		return
	}
	// openai_oauth_ready used to advertise whether an embedder-supplied
	// client ID was configured. The codex flow works without one, so
	// it's now always true — kept for UI backwards compatibility.
	writeJSON(w, http.StatusOK, map[string]any{
		"providers":          statuses,
		"openai_oauth_ready": true,
	})
}

func (h *ManagedHandler) handleAPIKeyLogin(providerName string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Key string `json:"key"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeJSONError(w, http.StatusBadRequest, fmt.Errorf("invalid JSON body"))
			return
		}
		if err := h.mgr.SaveAPIKey(r.Context(), providerName, body.Key); err != nil {
			writeJSONError(w, http.StatusBadRequest, err)
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{
			"status":   "authenticated",
			"provider": providerName,
		})
	}
}

func (h *ManagedHandler) handleOllamaLogin(w http.ResponseWriter, r *http.Request) {
	var body struct {
		BaseURL string `json:"base_url"`
	}
	// Body is optional for ollama — empty falls back to localhost.
	_ = json.NewDecoder(r.Body).Decode(&body)
	if err := h.mgr.SaveOllamaURL(r.Context(), body.BaseURL); err != nil {
		writeJSONError(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{
		"status":   "authenticated",
		"provider": profile.ProviderOllama,
	})
}

func (h *ManagedHandler) handleOpenAIOAuthStart(w http.ResponseWriter, r *http.Request) {
	h.mu.Lock()
	if h.openaiFlw != nil && h.openaiFlw.State() == profile.DeviceFlowPending {
		flow := h.openaiFlw
		h.mu.Unlock()
		writeJSON(w, http.StatusOK, deviceFlowPayload(flow))
		return
	}
	h.mu.Unlock()

	flow, err := h.mgr.BeginOpenAICodexDeviceLogin(r.Context())
	if err != nil {
		status := http.StatusInternalServerError
		if flow != nil && flow.State() == profile.DeviceFlowError {
			status = http.StatusBadGateway
		}
		writeJSONError(w, status, err)
		return
	}

	h.mu.Lock()
	h.openaiFlw = flow
	h.mu.Unlock()

	writeJSON(w, http.StatusOK, deviceFlowPayload(flow))
}

func (h *ManagedHandler) handleOpenAIOAuthStatus(w http.ResponseWriter, r *http.Request) {
	h.mu.Lock()
	flow := h.openaiFlw
	h.mu.Unlock()

	if flow == nil {
		writeJSON(w, http.StatusOK, map[string]string{"status": "not_started"})
		return
	}
	writeJSON(w, http.StatusOK, deviceFlowPayload(flow))
}

func (h *ManagedHandler) handleCopilotStart(w http.ResponseWriter, r *http.Request) {
	h.mu.Lock()
	if h.copilotFlw != nil && h.copilotFlw.State() == profile.DeviceFlowPending {
		// Existing flow in flight — surface its code rather than
		// minting a new one so the user's open tab stays valid.
		flow := h.copilotFlw
		h.mu.Unlock()
		writeJSON(w, http.StatusOK, deviceFlowPayload(flow))
		return
	}
	h.mu.Unlock()

	flow, err := h.mgr.BeginCopilotLogin(r.Context())
	if err != nil {
		status := http.StatusInternalServerError
		if flow != nil && flow.State() == profile.DeviceFlowError {
			status = http.StatusBadGateway
		}
		writeJSONError(w, status, err)
		return
	}

	h.mu.Lock()
	h.copilotFlw = flow
	h.mu.Unlock()

	writeJSON(w, http.StatusOK, deviceFlowPayload(flow))
}

func (h *ManagedHandler) handleCopilotStatus(w http.ResponseWriter, r *http.Request) {
	h.mu.Lock()
	flow := h.copilotFlw
	h.mu.Unlock()

	if flow == nil {
		writeJSON(w, http.StatusOK, map[string]string{"status": "not_started"})
		return
	}
	writeJSON(w, http.StatusOK, deviceFlowPayload(flow))
}

// deviceFlowPayload produces the wire shape for a DeviceFlow. The
// shape is stable so the UI can render a consistent progress box.
func deviceFlowPayload(flow *profile.DeviceFlow) map[string]any {
	payload := map[string]any{
		"status":     string(flow.State()),
		"user_code":  flow.UserCode(),
		"verify_url": flow.VerifyURL(),
	}
	if err := flow.Err(); err != nil {
		payload["error"] = err.Error()
	}
	return payload
}

func (h *ManagedHandler) handleLogout(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")

	// If a device flow is still polling for this provider, cancel
	// it and wait for the goroutine to exit before clearing
	// credentials — otherwise a late successful poll could rewrite
	// the store after mgr.Logout deletes it.
	h.cancelPendingFlow(r.Context(), name)

	if err := h.mgr.Logout(r.Context(), name); err != nil {
		writeJSONError(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{
		"status":   "logged_out",
		"provider": name,
	})
}

// cancelPendingFlow cancels the device flow for the given provider
// (if any) and waits for its background goroutine to exit. Bounded
// by a short timeout so a misbehaving flow doesn't block logout
// indefinitely; in the pathological "never terminates" case we
// accept the residual risk because the alternative is a hung HTTP
// request.
func (h *ManagedHandler) cancelPendingFlow(ctx context.Context, providerName string) {
	h.mu.Lock()
	var dev *profile.DeviceFlow
	switch providerName {
	case profile.ProviderCopilot:
		dev = h.copilotFlw
		h.copilotFlw = nil
	case profile.ProviderOpenAI:
		dev = h.openaiFlw
		h.openaiFlw = nil
	}
	h.mu.Unlock()

	if dev == nil {
		return
	}
	dev.Cancel()

	waitCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	_ = dev.Wait(waitCtx)
}

// ---- Models ----

func (h *ManagedHandler) handleListAllModels(w http.ResponseWriter, r *http.Request) {
	models, err := h.mgr.ListAllModels(r.Context())
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": toModelPayloads(models)})
}

func (h *ManagedHandler) handleListProviderModels(w http.ResponseWriter, r *http.Request) {
	providerName := r.PathValue("provider")
	// ListProviderModels preserves the Endpoint metadata, which is
	// required to route models that are only reachable via /responses
	// (e.g. Copilot gpt-5.4-mini) on subsequent chat requests.
	models, err := h.mgr.ListProviderModels(r.Context(), providerName)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err)
		return
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Model < models[j].Model })
	writeJSON(w, http.StatusOK, map[string]any{"models": toModelPayloads(models)})
}

func toModelPayloads(models []profile.ModelInfo) []map[string]any {
	out := make([]map[string]any, len(models))
	for i, m := range models {
		entry := map[string]any{
			"provider": m.Provider,
			"model":    m.Model,
			"id":       m.String(),
		}
		if m.Endpoint != "" {
			entry["endpoint"] = m.Endpoint
		}
		if m.ContextWindow > 0 {
			entry["context_window"] = m.ContextWindow
		}
		out[i] = entry
	}
	return out
}

// ---- Default ----

func (h *ManagedHandler) handleGetDefault(w http.ResponseWriter, r *http.Request) {
	info, ok, err := h.mgr.Default(r.Context())
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, err)
		return
	}
	if !ok {
		writeJSON(w, http.StatusOK, map[string]any{"set": false})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"set":      true,
		"provider": info.Provider,
		"model":    info.Model,
		"endpoint": info.Endpoint,
	})
}

func (h *ManagedHandler) handleSetDefault(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Provider string `json:"provider"`
		Model    string `json:"model"`
		Endpoint string `json:"endpoint"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeJSONError(w, http.StatusBadRequest, fmt.Errorf("invalid JSON body"))
		return
	}
	info := profile.ModelInfo{Provider: body.Provider, Model: body.Model, Endpoint: body.Endpoint}
	if err := h.mgr.SetDefaultModel(info); err != nil {
		writeJSONError(w, http.StatusBadRequest, err)
		return
	}
	// Keep the recent-models list updated so future UIs can surface
	// the user's picks without re-querying upstream.
	if err := h.mgr.TrackRecentModel(r.Context(), info); err != nil {
		writeJSONError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status":   "ok",
		"provider": info.Provider,
		"model":    info.Model,
		"endpoint": info.Endpoint,
	})
}

// ---- Chat ----

func (h *ManagedHandler) handleChat(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSONError(w, http.StatusBadRequest, fmt.Errorf("invalid request body"))
		return
	}
	if !hasMessageContent(req.Message) {
		writeJSONError(w, http.StatusBadRequest, fmt.Errorf("message is required"))
		return
	}

	info, ok, err := h.mgr.Default(r.Context())
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, err)
		return
	}
	if !ok {
		writeJSONError(w, http.StatusConflict, fmt.Errorf("no default model set — call POST /default first"))
		return
	}

	p, err := h.mgr.LoadProviderForModel(r.Context(), info)
	if err != nil {
		writeJSONError(w, http.StatusBadGateway, err)
		return
	}

	opts := append([]agent.Option{agent.WithModel(info.Model)}, h.agentOpts...)
	a := agent.New(p, opts...)

	ctx := r.Context()
	var sess *session.Session
	if req.SessionID != "" {
		sess, err = h.store.Load(ctx, req.SessionID)
		if err != nil {
			sess = session.New()
		}
	} else {
		sess = session.New()
	}
	sess.Model = info.String()

	req.Message.Role = conversation.RoleUser
	sess.AppendMessage(req.Message)

	sse, err := newSSEWriter(w)
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, fmt.Errorf("streaming not supported"))
		return
	}

	events, err := a.Run(ctx, sess.Messages)
	if err != nil {
		sse.writeEvent("error", map[string]string{"message": err.Error()})
		return
	}

	for ev := range events {
		switch ev.Type {
		case agent.EventBlockStart:
			sse.writeEvent("block_start", map[string]string{"block_type": string(ev.BlockType)})
		case agent.EventBlockDelta:
			sse.writeEvent("block_delta", map[string]string{
				"block_type": string(ev.BlockType),
				"delta":      ev.Content,
			})
		case agent.EventBlockEnd:
			payload := map[string]any{"block_type": string(ev.BlockType)}
			if ev.Block != nil {
				payload["block"] = blockToJSON(*ev.Block)
			}
			sse.writeEvent("block_end", payload)
		case agent.EventComplete:
			sess.Messages = append([]conversation.Message(nil), ev.Messages...)
			// Use background ctx so we still persist if the client
			// disconnects mid-stream — chat history shouldn't be
			// silently dropped.
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

// ---- Sessions ----

func (h *ManagedHandler) handleGetSession(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	sess, err := h.store.Load(r.Context(), id)
	if err != nil {
		writeJSONError(w, http.StatusNotFound, fmt.Errorf("session not found: %s", id))
		return
	}
	writeJSON(w, http.StatusOK, sess)
}

func (h *ManagedHandler) handleDeleteSession(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if err := h.store.Delete(r.Context(), id); err != nil {
		writeJSONError(w, http.StatusNotFound, fmt.Errorf("session not found: %s", id))
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// ---- helpers ----

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeJSONError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error()})
}
