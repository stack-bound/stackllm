package web

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"

	"github.com/stack-bound/stackllm/auth"
)

// AuthRoutes returns an http.Handler with auth-related endpoints.
// These are optional and can be mounted alongside the main Handler.
type AuthRoutes struct {
	copilot *auth.CopilotTokenSource
	openai  *auth.OpenAIWebFlowSource
	mux     *http.ServeMux

	// Track pending device flow state.
	mu             sync.Mutex
	pendingCode    string
	pendingURL     string
	copilotAuthed  bool
	copilotErr     error
	copilotPending bool
}

// AuthRoutesConfig configures the auth routes.
type AuthRoutesConfig struct {
	// Optional: Copilot token source for device flow auth.
	Copilot *auth.CopilotTokenSource

	// Optional: OpenAI web flow source for PKCE auth.
	OpenAI *auth.OpenAIWebFlowSource
}

// NewAuthRoutes creates auth routes for Copilot device flow and OpenAI web flow.
func NewAuthRoutes(cfg AuthRoutesConfig) *AuthRoutes {
	a := &AuthRoutes{
		copilot: cfg.Copilot,
		openai:  cfg.OpenAI,
		mux:     http.NewServeMux(),
	}
	a.mux.HandleFunc("GET /auth/status", a.handleStatus)
	a.mux.HandleFunc("GET /auth/copilot/start", a.handleCopilotStart)
	a.mux.HandleFunc("GET /auth/copilot/status", a.handleCopilotStatus)
	a.mux.HandleFunc("GET /auth/openai/start", a.handleOpenAIStart)
	a.mux.HandleFunc("GET /auth/openai/callback", a.handleOpenAICallback)
	return a
}

// ServeHTTP implements http.Handler.
func (a *AuthRoutes) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	a.mux.ServeHTTP(w, r)
}

func (a *AuthRoutes) handleStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "ok",
	})
}

// handleCopilotStart initiates the GitHub device flow and returns the user code
// and verification URL. The flow continues polling in the background.
func (a *AuthRoutes) handleCopilotStart(w http.ResponseWriter, r *http.Request) {
	if a.copilot == nil {
		http.Error(w, `{"error":"copilot auth not configured"}`, http.StatusNotFound)
		return
	}

	a.mu.Lock()
	if a.copilotPending {
		// Already in progress — return current code.
		code := a.pendingCode
		url := a.pendingURL
		a.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"user_code":  code,
			"verify_url": url,
		})
		return
	}
	a.copilotPending = true
	a.copilotAuthed = false
	a.copilotErr = nil
	a.mu.Unlock()

	// Start login in background. The OnDeviceCode callback captures the code.
	go func() {
		err := a.copilot.Login(context.Background())
		a.mu.Lock()
		defer a.mu.Unlock()
		a.copilotPending = false
		if err != nil {
			a.copilotErr = err
		} else {
			a.copilotAuthed = true
		}
	}()

	// Wait briefly for the device code callback to fire.
	// The code is set synchronously in OnDeviceCode before Login returns.
	// We give it a moment to propagate.
	a.mu.Lock()
	code := a.pendingCode
	url := a.pendingURL
	a.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"user_code":  code,
		"verify_url": url,
	})
}

// handleCopilotStatus returns the current state of the Copilot auth flow.
func (a *AuthRoutes) handleCopilotStatus(w http.ResponseWriter, r *http.Request) {
	if a.copilot == nil {
		http.Error(w, `{"error":"copilot auth not configured"}`, http.StatusNotFound)
		return
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")

	if a.copilotErr != nil {
		json.NewEncoder(w).Encode(map[string]string{
			"status": "error",
			"error":  a.copilotErr.Error(),
		})
		return
	}
	if a.copilotAuthed {
		json.NewEncoder(w).Encode(map[string]string{
			"status": "authenticated",
		})
		return
	}
	if a.copilotPending {
		json.NewEncoder(w).Encode(map[string]string{
			"status": "pending",
		})
		return
	}

	json.NewEncoder(w).Encode(map[string]string{
		"status": "not_started",
	})
}

// handleOpenAIStart initiates the OpenAI PKCE web flow and returns the
// authorization URL for the client to redirect the user to.
func (a *AuthRoutes) handleOpenAIStart(w http.ResponseWriter, r *http.Request) {
	if a.openai == nil {
		http.Error(w, `{"error":"openai auth not configured"}`, http.StatusNotFound)
		return
	}

	authURL, err := a.openai.Begin(r.Context(), a.openAIRedirectURI(r))
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"%s"}`, err.Error()), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":       "redirect",
		"redirect_url": authURL,
	})
}

// handleOpenAICallback handles the OAuth redirect callback from OpenAI.
// This is registered as the redirect URI in the OpenAI OAuth application.
func (a *AuthRoutes) handleOpenAICallback(w http.ResponseWriter, r *http.Request) {
	if a.openai == nil {
		http.Error(w, `{"error":"openai auth not configured"}`, http.StatusNotFound)
		return
	}

	// The actual callback is handled by the local server inside OpenAIWebFlowSource.Login().
	// This endpoint exists for cases where the web adapter is the callback target.
	errMsg := r.URL.Query().Get("error")
	if errMsg != "" {
		desc := r.URL.Query().Get("error_description")
		http.Error(w, fmt.Sprintf(`{"error":"%s","description":"%s"}`, errMsg, desc), http.StatusBadRequest)
		return
	}

	code := r.URL.Query().Get("code")
	if code == "" {
		http.Error(w, `{"error":"no authorization code in callback"}`, http.StatusBadRequest)
		return
	}
	if err := a.openai.Complete(r.Context(), r.URL.Query().Get("state"), code); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"%s"}`, err.Error()), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "text/html")
	fmt.Fprint(w, "<html><body><h1>Authentication successful</h1><p>You can close this window.</p></body></html>")
}

// SetDeviceCode is called by the CopilotTokenSource's OnDeviceCode callback to
// capture the device code for the web API to return.
func (a *AuthRoutes) SetDeviceCode(userCode, verifyURL string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.pendingCode = userCode
	a.pendingURL = verifyURL
}

func (a *AuthRoutes) openAIRedirectURI(r *http.Request) string {
	scheme := "http"
	if r.TLS != nil {
		scheme = "https"
	}
	if forwarded := r.Header.Get("X-Forwarded-Proto"); forwarded != "" {
		scheme = forwarded
	}
	return fmt.Sprintf("%s://%s/auth/openai/callback", scheme, r.Host)
}
