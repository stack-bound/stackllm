package auth

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"sync"
	"time"
)

const (
	openaiAuthURL       = "https://auth0.openai.com/authorize"
	openaiWebTokenURL   = "https://auth0.openai.com/oauth/token"
	openaiWebStoreKey   = "openai_web_token"
	defaultCallbackPort = 1455
)

// OpenAIWebFlowConfig configures the PKCE authorization code flow.
type OpenAIWebFlowConfig struct {
	ClientID string
	Port     int // local callback port, default 1455
	Store    TokenStore

	// Called with the full authorization URL.
	OnOpenURL func(authURL string)
	OnSuccess func()

	// HTTP client override for testing.
	HTTPClient *http.Client

	// Optional endpoint overrides for testing.
	AuthURL  string
	TokenURL string
}

func (c *OpenAIWebFlowConfig) port() int {
	if c.Port != 0 {
		return c.Port
	}
	return defaultCallbackPort
}

func (c *OpenAIWebFlowConfig) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return http.DefaultClient
}

func (c *OpenAIWebFlowConfig) authURL() string {
	if c.AuthURL != "" {
		return c.AuthURL
	}
	return openaiAuthURL
}

func (c *OpenAIWebFlowConfig) tokenURL() string {
	if c.TokenURL != "" {
		return c.TokenURL
	}
	return openaiWebTokenURL
}

// OpenAIWebFlowSource implements the PKCE authorization code flow with a
// local callback server.
type OpenAIWebFlowSource struct {
	cfg    OpenAIWebFlowConfig
	record *openAITokenRecord
	mu     sync.Mutex
	flow   *openAIAuthFlow
}

type openAIAuthFlow struct {
	State       string
	Verifier    string
	RedirectURI string
}

// NewOpenAIWebFlowSource creates a new OpenAIWebFlowSource.
func NewOpenAIWebFlowSource(cfg OpenAIWebFlowConfig) *OpenAIWebFlowSource {
	return &OpenAIWebFlowSource{cfg: cfg}
}

// Token returns a valid token, performing the web flow if needed.
func (s *OpenAIWebFlowSource) Token(ctx context.Context) (*Token, error) {
	if s.record != nil && s.record.token().Valid() {
		return s.record.token(), nil
	}

	record, err := loadOpenAITokenRecord(ctx, s.cfg.Store, openaiWebStoreKey)
	if err == nil {
		s.record = record
		if s.record.token().Valid() {
			return s.record.token(), nil
		}
		if s.record.RefreshToken != "" {
			refreshed, refreshErr := s.refresh(ctx, s.record.RefreshToken)
			if refreshErr == nil {
				s.record = refreshed
				return s.record.token(), nil
			}
		}
	}

	if err := s.Login(ctx); err != nil {
		return nil, err
	}

	record, err = loadOpenAITokenRecord(ctx, s.cfg.Store, openaiWebStoreKey)
	if err != nil {
		return nil, fmt.Errorf("auth: openai web load after login: %w", err)
	}
	s.record = record
	return s.record.token(), nil
}

// Login performs the PKCE web flow.
func (s *OpenAIWebFlowSource) Login(ctx context.Context) error {
	// Generate PKCE code verifier and challenge.
	verifier, err := generateCodeVerifier()
	if err != nil {
		return fmt.Errorf("auth: openai web generate verifier: %w", err)
	}
	challenge := generateCodeChallenge(verifier)

	// Generate state parameter.
	stateBytes := make([]byte, 16)
	if _, err := rand.Read(stateBytes); err != nil {
		return fmt.Errorf("auth: openai web generate state: %w", err)
	}
	state := base64.RawURLEncoding.EncodeToString(stateBytes)

	port := s.cfg.port()
	redirectURI := fmt.Sprintf("http://localhost:%d/callback", port)

	// Build authorization URL.
	params := url.Values{
		"client_id":             {s.cfg.ClientID},
		"response_type":        {"code"},
		"redirect_uri":         {redirectURI},
		"scope":                {"openid profile email offline_access"},
		"audience":             {"https://api.openai.com/v1"},
		"state":                {state},
		"code_challenge":       {challenge},
		"code_challenge_method": {"S256"},
	}
	authURL := s.cfg.authURL() + "?" + params.Encode()

	// Channel to receive the authorization code.
	codeCh := make(chan string, 1)
	errCh := make(chan error, 1)

	// Start local callback server.
	mux := http.NewServeMux()
	mux.HandleFunc("/callback", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("state") != state {
			errCh <- fmt.Errorf("auth: openai web: state mismatch")
			http.Error(w, "State mismatch", http.StatusBadRequest)
			return
		}
		if errMsg := r.URL.Query().Get("error"); errMsg != "" {
			errCh <- fmt.Errorf("auth: openai web: %s: %s", errMsg, r.URL.Query().Get("error_description"))
			http.Error(w, errMsg, http.StatusBadRequest)
			return
		}
		code := r.URL.Query().Get("code")
		if code == "" {
			errCh <- fmt.Errorf("auth: openai web: no code in callback")
			http.Error(w, "No code", http.StatusBadRequest)
			return
		}
		fmt.Fprint(w, "<html><body><h1>Authentication successful</h1><p>You can close this window.</p></body></html>")
		codeCh <- code
	})

	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return fmt.Errorf("auth: openai web listen: %w", err)
	}

	server := &http.Server{Handler: mux}
	go func() {
		if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
			errCh <- fmt.Errorf("auth: openai web server: %w", err)
		}
	}()
	defer server.Close()

	// Tell the user to open the URL.
	if s.cfg.OnOpenURL != nil {
		s.cfg.OnOpenURL(authURL)
	}

	// Wait for callback with 5 minute timeout.
	flowCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()

	var code string
	select {
	case code = <-codeCh:
	case err := <-errCh:
		return err
	case <-flowCtx.Done():
		return fmt.Errorf("auth: openai web flow timed out")
	}

	// Exchange code for token.
	return s.exchangeCode(ctx, code, verifier, redirectURI)
}

// Logout deletes the stored token.
func (s *OpenAIWebFlowSource) Logout(ctx context.Context) error {
	s.record = nil
	s.mu.Lock()
	s.flow = nil
	s.mu.Unlock()
	return s.cfg.Store.Delete(ctx, openaiWebStoreKey)
}

// Begin prepares a PKCE web flow and returns the authorization URL.
func (s *OpenAIWebFlowSource) Begin(_ context.Context, redirectURI string) (string, error) {
	verifier, err := generateCodeVerifier()
	if err != nil {
		return "", fmt.Errorf("auth: openai web generate verifier: %w", err)
	}
	challenge := generateCodeChallenge(verifier)

	stateBytes := make([]byte, 16)
	if _, err := rand.Read(stateBytes); err != nil {
		return "", fmt.Errorf("auth: openai web generate state: %w", err)
	}
	state := base64.RawURLEncoding.EncodeToString(stateBytes)

	params := url.Values{
		"client_id":              {s.cfg.ClientID},
		"response_type":          {"code"},
		"redirect_uri":           {redirectURI},
		"scope":                  {"openid profile email offline_access"},
		"audience":               {"https://api.openai.com/v1"},
		"state":                  {state},
		"code_challenge":         {challenge},
		"code_challenge_method":  {"S256"},
	}

	s.mu.Lock()
	s.flow = &openAIAuthFlow{
		State:       state,
		Verifier:    verifier,
		RedirectURI: redirectURI,
	}
	s.mu.Unlock()

	return s.cfg.authURL() + "?" + params.Encode(), nil
}

// Complete finishes a previously started PKCE web flow.
func (s *OpenAIWebFlowSource) Complete(ctx context.Context, state, code string) error {
	s.mu.Lock()
	flow := s.flow
	s.mu.Unlock()

	if flow == nil {
		return fmt.Errorf("auth: openai web: no flow in progress")
	}
	if state != flow.State {
		return fmt.Errorf("auth: openai web: state mismatch")
	}

	return s.exchangeCode(ctx, code, flow.Verifier, flow.RedirectURI)
}

func (s *OpenAIWebFlowSource) exchangeCode(ctx context.Context, code, verifier, redirectURI string) error {
	form := url.Values{
		"client_id":     {s.cfg.ClientID},
		"grant_type":    {"authorization_code"},
		"code":          {code},
		"redirect_uri":  {redirectURI},
		"code_verifier": {verifier},
	}

	record, err := exchangeOpenAIToken(ctx, s.cfg.httpClient(), s.cfg.tokenURL(), form)
	if err != nil {
		return fmt.Errorf("auth: openai web exchange: %w", err)
	}
	if err := saveOpenAITokenRecord(ctx, s.cfg.Store, openaiWebStoreKey, *record); err != nil {
		return err
	}
	s.record = record
	s.mu.Lock()
	s.flow = nil
	s.mu.Unlock()

	if s.cfg.OnSuccess != nil {
		s.cfg.OnSuccess()
	}
	return nil
}

func (s *OpenAIWebFlowSource) refresh(ctx context.Context, refreshToken string) (*openAITokenRecord, error) {
	form := url.Values{
		"client_id":     {s.cfg.ClientID},
		"grant_type":    {"refresh_token"},
		"refresh_token": {refreshToken},
	}

	record, err := exchangeOpenAIToken(ctx, s.cfg.httpClient(), s.cfg.tokenURL(), form)
	if err != nil {
		return nil, err
	}
	if record.RefreshToken == "" {
		record.RefreshToken = refreshToken
	}
	if err := saveOpenAITokenRecord(ctx, s.cfg.Store, openaiWebStoreKey, *record); err != nil {
		return nil, err
	}
	return record, nil
}

func generateCodeVerifier() (string, error) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(b), nil
}

func generateCodeChallenge(verifier string) string {
	h := sha256.Sum256([]byte(verifier))
	return base64.RawURLEncoding.EncodeToString(h[:])
}
