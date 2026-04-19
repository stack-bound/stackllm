package auth

// OpenAI Codex OAuth ("Sign in with ChatGPT") flow.
//
// This flow lets a user sign in with their ChatGPT account and use that
// subscription as an LLM provider, without having to register their own
// OAuth application with OpenAI. It uses the OAuth client ID that
// OpenAI publishes for their own Codex CLI — the same ID opencode and
// the official codex-cli use.
//
// Tokens minted by this flow are NOT valid against the standard
// api.openai.com endpoints. They are only accepted by
// https://chatgpt.com/backend-api/codex/responses and require a
// ChatGPT-Account-Id header that must be extracted from the JWT
// returned by the token exchange.
//
// Two sources are exposed:
//   - CodexDeviceSource  — headless device-code flow.
//   - CodexWebFlowSource — PKCE authorization-code flow with a local
//                          http://localhost:1455/auth/callback listener.

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

// CodexDefaultClientID is the OAuth client ID that OpenAI publishes
// for their Codex CLI. It is intentionally hardcoded so embedders do
// not have to register their own OAuth app to use the codex flow —
// opencode, the official codex-cli, and stackllm all share it.
const CodexDefaultClientID = "app_EMoamEEZ73f0CkXaXp7hrann"

const (
	codexIssuer          = "https://auth.openai.com"
	codexAuthorizeURL    = codexIssuer + "/oauth/authorize"
	codexTokenURL        = codexIssuer + "/oauth/token"
	codexDeviceUserCode  = codexIssuer + "/api/accounts/deviceauth/usercode"
	codexDevicePollURL   = codexIssuer + "/api/accounts/deviceauth/token"
	codexDeviceVerifyURL = codexIssuer + "/codex/device"
	codexDeviceRedirect  = codexIssuer + "/deviceauth/callback"
	codexWebCallbackPath = "/auth/callback"
	codexDefaultPort     = 1455
	codexScopes          = "openid profile email offline_access"
	codexOriginator      = "stackllm"

	// CodexStoreKey is the auth-store key under which the codex token
	// record is persisted. Exposed so the profile package can read the
	// extracted account ID at provider-build time.
	CodexStoreKey = "openai_codex_token"
)

// CodexTokenRecord is the persisted credential for a codex sign-in.
// It carries the access / refresh tokens plus the ChatGPT account ID
// extracted from the id_token JWT — the account ID is stable across
// refreshes so we capture it once at login and reuse it on every
// request header.
type CodexTokenRecord struct {
	AccessToken      string    `json:"access_token"`
	RefreshToken     string    `json:"refresh_token,omitempty"`
	IDToken          string    `json:"id_token,omitempty"`
	ChatGPTAccountID string    `json:"chatgpt_account_id,omitempty"`
	ExpiresAt        time.Time `json:"expires_at,omitempty"`
}

func (r CodexTokenRecord) token() *Token {
	return &Token{AccessToken: r.AccessToken, ExpiresAt: r.ExpiresAt}
}

// LoadCodexRecord loads the persisted codex token record from the
// auth store. Returns an error if no record is present.
func LoadCodexRecord(ctx context.Context, store TokenStore) (*CodexTokenRecord, error) {
	raw, err := store.Load(ctx, CodexStoreKey)
	if err != nil {
		return nil, err
	}
	var rec CodexTokenRecord
	if err := json.Unmarshal([]byte(raw), &rec); err != nil {
		return nil, fmt.Errorf("auth: codex: decode record: %w", err)
	}
	if rec.AccessToken == "" {
		return nil, fmt.Errorf("auth: codex: empty access token in stored record")
	}
	return &rec, nil
}

func saveCodexRecord(ctx context.Context, store TokenStore, rec CodexTokenRecord) error {
	data, err := json.Marshal(rec)
	if err != nil {
		return fmt.Errorf("auth: codex: marshal record: %w", err)
	}
	if err := store.Save(ctx, CodexStoreKey, string(data)); err != nil {
		return fmt.Errorf("auth: codex: save record: %w", err)
	}
	return nil
}

// codexTokenResponse is the shape returned by the /oauth/token exchange.
type codexTokenResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	IDToken      string `json:"id_token"`
	ExpiresIn    int    `json:"expires_in"`
	Error        string `json:"error"`
	ErrorDesc    string `json:"error_description"`
}

func (r codexTokenResponse) toRecord() CodexTokenRecord {
	rec := CodexTokenRecord{
		AccessToken:  r.AccessToken,
		RefreshToken: r.RefreshToken,
		IDToken:      r.IDToken,
	}
	if r.ExpiresIn > 0 {
		rec.ExpiresAt = time.Now().Add(time.Duration(r.ExpiresIn) * time.Second)
	}
	if r.IDToken != "" {
		if acc := extractChatGPTAccountID(r.IDToken); acc != "" {
			rec.ChatGPTAccountID = acc
		}
	}
	return rec
}

// extractChatGPTAccountID pulls the ChatGPT account ID out of the
// id_token JWT payload. opencode looks at three locations in this
// order; we match that behaviour so tokens minted by either tool
// interpret the same way.
func extractChatGPTAccountID(idToken string) string {
	parts := strings.Split(idToken, ".")
	if len(parts) < 2 {
		return ""
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		// Some JWTs use padded base64 — try StdEncoding as a fallback.
		payload, err = base64.StdEncoding.DecodeString(parts[1])
		if err != nil {
			return ""
		}
	}
	var claims struct {
		ChatGPTAccountID string `json:"chatgpt_account_id"`
		AuthClaims       struct {
			ChatGPTAccountID string `json:"chatgpt_account_id"`
		} `json:"https://api.openai.com/auth"`
		Organizations []struct {
			ID string `json:"id"`
		} `json:"organizations"`
	}
	if err := json.Unmarshal(payload, &claims); err != nil {
		return ""
	}
	if claims.ChatGPTAccountID != "" {
		return claims.ChatGPTAccountID
	}
	if claims.AuthClaims.ChatGPTAccountID != "" {
		return claims.AuthClaims.ChatGPTAccountID
	}
	if len(claims.Organizations) > 0 && claims.Organizations[0].ID != "" {
		return claims.Organizations[0].ID
	}
	return ""
}

// exchangeCodexToken performs the form-encoded POST to the codex
// /oauth/token endpoint and decodes the response into a record. Used
// by both the device and PKCE flows, as well as refresh.
func exchangeCodexToken(ctx context.Context, client *http.Client, tokenURL string, form url.Values) (*CodexTokenRecord, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, tokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, fmt.Errorf("auth: codex: build token request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("auth: codex: token request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("auth: codex: token exchange: status %d: %s", resp.StatusCode, body)
	}

	var tr codexTokenResponse
	if err := json.NewDecoder(resp.Body).Decode(&tr); err != nil {
		return nil, fmt.Errorf("auth: codex: decode token response: %w", err)
	}
	if tr.Error != "" {
		return nil, fmt.Errorf("auth: codex: token exchange: %s: %s", tr.Error, tr.ErrorDesc)
	}
	if tr.AccessToken == "" {
		return nil, fmt.Errorf("auth: codex: token exchange: empty access token")
	}
	rec := tr.toRecord()
	return &rec, nil
}

// ---------- Device flow ----------

// CodexDeviceConfig configures CodexDeviceSource.
type CodexDeviceConfig struct {
	// ClientID overrides the Codex public client ID. Leave empty to
	// use CodexDefaultClientID — most callers should.
	ClientID string

	Store TokenStore

	// OnCode fires once with the one-time code and verification URL
	// that the user should open in a browser.
	OnCode    func(userCode, verifyURL string)
	OnPolling func()
	OnSuccess func()

	// PollInterval overrides the server-suggested poll cadence. The
	// server interval takes precedence when non-zero.
	PollInterval time.Duration

	// HTTPClient override for testing.
	HTTPClient *http.Client

	// Endpoint overrides for testing. Empty values fall through to the
	// production codex endpoints.
	DeviceCodeURL string
	DevicePollURL string
	TokenURL      string
	VerifyURL     string
	RedirectURI   string
}

func (c *CodexDeviceConfig) clientID() string {
	if c.ClientID != "" {
		return c.ClientID
	}
	return CodexDefaultClientID
}

func (c *CodexDeviceConfig) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return http.DefaultClient
}

func (c *CodexDeviceConfig) deviceCodeURL() string {
	if c.DeviceCodeURL != "" {
		return c.DeviceCodeURL
	}
	return codexDeviceUserCode
}

func (c *CodexDeviceConfig) devicePollURL() string {
	if c.DevicePollURL != "" {
		return c.DevicePollURL
	}
	return codexDevicePollURL
}

func (c *CodexDeviceConfig) tokenURL() string {
	if c.TokenURL != "" {
		return c.TokenURL
	}
	return codexTokenURL
}

func (c *CodexDeviceConfig) verifyURL() string {
	if c.VerifyURL != "" {
		return c.VerifyURL
	}
	return codexDeviceVerifyURL
}

func (c *CodexDeviceConfig) redirectURI() string {
	if c.RedirectURI != "" {
		return c.RedirectURI
	}
	return codexDeviceRedirect
}

func (c *CodexDeviceConfig) pollInterval() time.Duration {
	if c.PollInterval > 0 {
		return c.PollInterval
	}
	return 2 * time.Second
}

// CodexDeviceSource implements the Codex device-code OAuth flow.
type CodexDeviceSource struct {
	cfg    CodexDeviceConfig
	record *CodexTokenRecord
}

// NewCodexDeviceSource constructs a new source. Store is required.
func NewCodexDeviceSource(cfg CodexDeviceConfig) *CodexDeviceSource {
	return &CodexDeviceSource{cfg: cfg}
}

// Token returns a valid codex access token, refreshing if the cached
// record has expired. Unlike NewOpenAIDeviceSource, this does NOT
// trigger the interactive login flow on a cache miss — the profile /
// web layer is responsible for orchestrating user-facing login.
func (s *CodexDeviceSource) Token(ctx context.Context) (*Token, error) {
	if s.record != nil && s.record.token().Valid() {
		return s.record.token(), nil
	}
	rec, err := LoadCodexRecord(ctx, s.cfg.Store)
	if err != nil {
		return nil, err
	}
	s.record = rec
	if s.record.token().Valid() {
		return s.record.token(), nil
	}
	if s.record.RefreshToken == "" {
		return nil, fmt.Errorf("auth: codex: token expired and no refresh token available — re-run login")
	}
	refreshed, err := refreshCodex(ctx, s.cfg.httpClient(), s.cfg.tokenURL(), s.cfg.clientID(), s.record.RefreshToken, s.record.ChatGPTAccountID)
	if err != nil {
		return nil, err
	}
	if err := saveCodexRecord(ctx, s.cfg.Store, *refreshed); err != nil {
		return nil, err
	}
	s.record = refreshed
	return s.record.token(), nil
}

// Record returns the currently cached token record (or nil). Useful
// when the caller needs the ChatGPT account ID and not just the
// access token string.
func (s *CodexDeviceSource) Record() *CodexTokenRecord { return s.record }

// Login runs the full device-code flow end to end: request a user
// code, invoke OnCode, poll until success. The resulting record is
// persisted to the configured store.
func (s *CodexDeviceSource) Login(ctx context.Context) error {
	client := s.cfg.httpClient()

	// Step 1: request the device/user code.
	userCodeBody, _ := json.Marshal(map[string]string{"client_id": s.cfg.clientID()})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.deviceCodeURL(), bytes.NewReader(userCodeBody))
	if err != nil {
		return fmt.Errorf("auth: codex: build usercode request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("auth: codex: usercode request: %w", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("auth: codex: usercode: status %d: %s", resp.StatusCode, body)
	}

	var usercode struct {
		DeviceAuthID string `json:"device_auth_id"`
		UserCode     string `json:"user_code"`
		Interval     int    `json:"interval"`
		ExpiresIn    int    `json:"expires_in"`
	}
	if err := json.Unmarshal(body, &usercode); err != nil {
		return fmt.Errorf("auth: codex: decode usercode: %w", err)
	}
	if usercode.UserCode == "" || usercode.DeviceAuthID == "" {
		return fmt.Errorf("auth: codex: usercode response missing fields")
	}

	if s.cfg.OnCode != nil {
		s.cfg.OnCode(usercode.UserCode, s.cfg.verifyURL())
	}

	// Step 2: poll until the user completes authorisation.
	interval := s.cfg.pollInterval()
	if usercode.Interval > 0 {
		interval = time.Duration(usercode.Interval) * time.Second
	}
	deadline := time.Now().Add(15 * time.Minute)
	if usercode.ExpiresIn > 0 {
		deadline = time.Now().Add(time.Duration(usercode.ExpiresIn) * time.Second)
	}

	for time.Now().Before(deadline) {
		if s.cfg.OnPolling != nil {
			s.cfg.OnPolling()
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(interval):
		}

		pollBody, _ := json.Marshal(map[string]string{
			"device_auth_id": usercode.DeviceAuthID,
			"user_code":      usercode.UserCode,
		})
		pollReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.devicePollURL(), bytes.NewReader(pollBody))
		if err != nil {
			return fmt.Errorf("auth: codex: build poll request: %w", err)
		}
		pollReq.Header.Set("Content-Type", "application/json")
		pollReq.Header.Set("Accept", "application/json")

		pollResp, err := client.Do(pollReq)
		if err != nil {
			continue
		}
		pollBodyRaw, _ := io.ReadAll(pollResp.Body)
		pollResp.Body.Close()

		// 403 / 404 mean "still waiting" in the codex device flow.
		if pollResp.StatusCode == http.StatusForbidden || pollResp.StatusCode == http.StatusNotFound {
			continue
		}
		if pollResp.StatusCode != http.StatusOK {
			return fmt.Errorf("auth: codex: poll: status %d: %s", pollResp.StatusCode, pollBodyRaw)
		}

		var pr struct {
			AuthorizationCode string `json:"authorization_code"`
			CodeVerifier      string `json:"code_verifier"`
		}
		if err := json.Unmarshal(pollBodyRaw, &pr); err != nil {
			return fmt.Errorf("auth: codex: decode poll response: %w", err)
		}
		if pr.AuthorizationCode == "" {
			continue
		}

		// Step 3: exchange authorization_code for tokens.
		form := url.Values{
			"grant_type":    {"authorization_code"},
			"code":          {pr.AuthorizationCode},
			"redirect_uri":  {s.cfg.redirectURI()},
			"client_id":     {s.cfg.clientID()},
			"code_verifier": {pr.CodeVerifier},
		}
		rec, err := exchangeCodexToken(ctx, client, s.cfg.tokenURL(), form)
		if err != nil {
			return err
		}
		if err := saveCodexRecord(ctx, s.cfg.Store, *rec); err != nil {
			return err
		}
		s.record = rec
		if s.cfg.OnSuccess != nil {
			s.cfg.OnSuccess()
		}
		return nil
	}
	return fmt.Errorf("auth: codex: device flow timed out")
}

// Logout clears the persisted codex credential.
func (s *CodexDeviceSource) Logout(ctx context.Context) error {
	s.record = nil
	return s.cfg.Store.Delete(ctx, CodexStoreKey)
}

// refreshCodex trades a refresh token for a new access token,
// preserving the stored ChatGPT account ID when the new id_token
// does not include one (refresh responses may omit id_token).
func refreshCodex(ctx context.Context, client *http.Client, tokenURL, clientID, refreshToken, knownAccountID string) (*CodexTokenRecord, error) {
	form := url.Values{
		"grant_type":    {"refresh_token"},
		"client_id":     {clientID},
		"refresh_token": {refreshToken},
		"scope":         {codexScopes},
	}
	rec, err := exchangeCodexToken(ctx, client, tokenURL, form)
	if err != nil {
		return nil, err
	}
	if rec.RefreshToken == "" {
		rec.RefreshToken = refreshToken
	}
	if rec.ChatGPTAccountID == "" {
		rec.ChatGPTAccountID = knownAccountID
	}
	return rec, nil
}

// ---------- PKCE / web flow ----------

// CodexWebFlowConfig configures CodexWebFlowSource.
type CodexWebFlowConfig struct {
	// ClientID overrides the Codex public client ID. Leave empty to
	// use CodexDefaultClientID.
	ClientID string

	Store TokenStore

	// Port selects the localhost port for the callback listener.
	// Default 1455 — the port that OpenAI has whitelisted for the
	// Codex CLI OAuth app. Changing this will break the redirect
	// unless you have your own OAuth app registered with a matching
	// redirect URI.
	Port int

	// OnOpenURL receives the full authorization URL so callers can
	// open it in a browser (or print it for the user to open).
	OnOpenURL func(authURL string)
	OnSuccess func()

	HTTPClient *http.Client

	AuthURL   string // override for testing
	TokenURL  string // override for testing
}

func (c *CodexWebFlowConfig) clientID() string {
	if c.ClientID != "" {
		return c.ClientID
	}
	return CodexDefaultClientID
}

func (c *CodexWebFlowConfig) port() int {
	if c.Port != 0 {
		return c.Port
	}
	return codexDefaultPort
}

func (c *CodexWebFlowConfig) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return http.DefaultClient
}

func (c *CodexWebFlowConfig) authURL() string {
	if c.AuthURL != "" {
		return c.AuthURL
	}
	return codexAuthorizeURL
}

func (c *CodexWebFlowConfig) tokenURL() string {
	if c.TokenURL != "" {
		return c.TokenURL
	}
	return codexTokenURL
}

// CodexWebFlowSource implements the Codex PKCE authorization-code
// flow with a local HTTP callback listener. Supports both a blocking
// end-to-end Login (suitable for the login CLI) and a Begin / Await
// pair (suitable for a web adapter that wants to drive the flow
// asynchronously).
type CodexWebFlowSource struct {
	cfg    CodexWebFlowConfig
	record *CodexTokenRecord

	mu   sync.Mutex
	flow *codexPKCEState
}

type codexPKCEState struct {
	State       string
	Verifier    string
	RedirectURI string
}

// NewCodexWebFlowSource constructs a new source.
func NewCodexWebFlowSource(cfg CodexWebFlowConfig) *CodexWebFlowSource {
	return &CodexWebFlowSource{cfg: cfg}
}

// Token returns a valid codex access token, refreshing if needed.
// Does not trigger interactive login on a cache miss — that is the
// caller's responsibility.
func (s *CodexWebFlowSource) Token(ctx context.Context) (*Token, error) {
	if s.record != nil && s.record.token().Valid() {
		return s.record.token(), nil
	}
	rec, err := LoadCodexRecord(ctx, s.cfg.Store)
	if err != nil {
		return nil, err
	}
	s.record = rec
	if s.record.token().Valid() {
		return s.record.token(), nil
	}
	if s.record.RefreshToken == "" {
		return nil, fmt.Errorf("auth: codex: token expired and no refresh token available — re-run login")
	}
	refreshed, err := refreshCodex(ctx, s.cfg.httpClient(), s.cfg.tokenURL(), s.cfg.clientID(), s.record.RefreshToken, s.record.ChatGPTAccountID)
	if err != nil {
		return nil, err
	}
	if err := saveCodexRecord(ctx, s.cfg.Store, *refreshed); err != nil {
		return nil, err
	}
	s.record = refreshed
	return s.record.token(), nil
}

// Record returns the currently cached token record (or nil).
func (s *CodexWebFlowSource) Record() *CodexTokenRecord { return s.record }

// Login runs the full PKCE flow end to end with a local callback
// listener. Blocks until either the callback fires, the context is
// cancelled, or a 5-minute timeout elapses.
func (s *CodexWebFlowSource) Login(ctx context.Context) error {
	redirectURI := fmt.Sprintf("http://localhost:%d%s", s.cfg.port(), codexWebCallbackPath)
	authURL, state, verifier, err := s.buildAuthorizeURL(redirectURI)
	if err != nil {
		return err
	}

	codeCh := make(chan string, 1)
	errCh := make(chan error, 1)

	mux := http.NewServeMux()
	mux.HandleFunc(codexWebCallbackPath, func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		if q.Get("state") != state {
			http.Error(w, "state mismatch", http.StatusBadRequest)
			select {
			case errCh <- fmt.Errorf("auth: codex: state mismatch"):
			default:
			}
			return
		}
		if e := q.Get("error"); e != "" {
			http.Error(w, e, http.StatusBadRequest)
			select {
			case errCh <- fmt.Errorf("auth: codex: authorize: %s: %s", e, q.Get("error_description")):
			default:
			}
			return
		}
		code := q.Get("code")
		if code == "" {
			http.Error(w, "missing code", http.StatusBadRequest)
			select {
			case errCh <- fmt.Errorf("auth: codex: missing code in callback"):
			default:
			}
			return
		}
		fmt.Fprint(w, "<html><body><h1>Signed in to OpenAI</h1><p>You can close this window.</p></body></html>")
		select {
		case codeCh <- code:
		default:
		}
	})

	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", s.cfg.port()))
	if err != nil {
		return fmt.Errorf("auth: codex: listen on %d: %w", s.cfg.port(), err)
	}
	server := &http.Server{Handler: mux}
	go func() {
		_ = server.Serve(listener)
	}()
	defer server.Close()

	if s.cfg.OnOpenURL != nil {
		s.cfg.OnOpenURL(authURL)
	}

	waitCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()

	var code string
	select {
	case code = <-codeCh:
	case err := <-errCh:
		return err
	case <-waitCtx.Done():
		return fmt.Errorf("auth: codex: web flow timed out: %w", waitCtx.Err())
	}

	return s.exchangeAndPersist(ctx, code, verifier, redirectURI)
}

// Begin prepares a PKCE flow without starting a listener and returns
// the authorization URL the caller should open. The caller is
// expected to complete the flow by calling Complete with the state
// and code values extracted from the callback request it observed.
func (s *CodexWebFlowSource) Begin(_ context.Context, redirectURI string) (string, error) {
	authURL, state, verifier, err := s.buildAuthorizeURL(redirectURI)
	if err != nil {
		return "", err
	}
	s.mu.Lock()
	s.flow = &codexPKCEState{State: state, Verifier: verifier, RedirectURI: redirectURI}
	s.mu.Unlock()
	return authURL, nil
}

// Complete finishes a flow that was started via Begin.
func (s *CodexWebFlowSource) Complete(ctx context.Context, state, code string) error {
	s.mu.Lock()
	flow := s.flow
	s.mu.Unlock()
	if flow == nil {
		return fmt.Errorf("auth: codex: no flow in progress")
	}
	if state != flow.State {
		return fmt.Errorf("auth: codex: state mismatch")
	}
	if err := s.exchangeAndPersist(ctx, code, flow.Verifier, flow.RedirectURI); err != nil {
		return err
	}
	s.mu.Lock()
	s.flow = nil
	s.mu.Unlock()
	return nil
}

// Logout clears the persisted codex credential.
func (s *CodexWebFlowSource) Logout(ctx context.Context) error {
	s.record = nil
	s.mu.Lock()
	s.flow = nil
	s.mu.Unlock()
	return s.cfg.Store.Delete(ctx, CodexStoreKey)
}

func (s *CodexWebFlowSource) buildAuthorizeURL(redirectURI string) (authURL, state, verifier string, err error) {
	verifier, err = generateCodeVerifier()
	if err != nil {
		return "", "", "", fmt.Errorf("auth: codex: generate verifier: %w", err)
	}
	challenge := generateCodeChallenge(verifier)

	stateBytes := make([]byte, 16)
	if _, err := rand.Read(stateBytes); err != nil {
		return "", "", "", fmt.Errorf("auth: codex: generate state: %w", err)
	}
	state = base64.RawURLEncoding.EncodeToString(stateBytes)

	params := url.Values{
		"response_type":              {"code"},
		"client_id":                  {s.cfg.clientID()},
		"redirect_uri":               {redirectURI},
		"scope":                      {codexScopes},
		"code_challenge":             {challenge},
		"code_challenge_method":      {"S256"},
		"id_token_add_organizations": {"true"},
		"codex_cli_simplified_flow":  {"true"},
		"state":                      {state},
		"originator":                 {codexOriginator},
	}
	return s.cfg.authURL() + "?" + params.Encode(), state, verifier, nil
}

func (s *CodexWebFlowSource) exchangeAndPersist(ctx context.Context, code, verifier, redirectURI string) error {
	form := url.Values{
		"grant_type":    {"authorization_code"},
		"code":          {code},
		"redirect_uri":  {redirectURI},
		"client_id":     {s.cfg.clientID()},
		"code_verifier": {verifier},
	}
	rec, err := exchangeCodexToken(ctx, s.cfg.httpClient(), s.cfg.tokenURL(), form)
	if err != nil {
		return err
	}
	if err := saveCodexRecord(ctx, s.cfg.Store, *rec); err != nil {
		return err
	}
	s.record = rec
	if s.cfg.OnSuccess != nil {
		s.cfg.OnSuccess()
	}
	return nil
}

// Codex provider wiring ---------------------------------------------

// CodexProviderBaseURL is the base URL that codex-flow tokens must
// target. Paired with the standard /responses endpoint this resolves
// to https://chatgpt.com/backend-api/codex/responses, which is the
// only endpoint OpenAI authorises for tokens minted by the Codex
// OAuth app.
const CodexProviderBaseURL = "https://chatgpt.com/backend-api/codex"

// CodexAccountHeader is the header name carrying the ChatGPT account
// ID on requests made with codex-flow tokens.
const CodexAccountHeader = "ChatGPT-Account-Id"

// CodexOriginatorHeader identifies the client to OpenAI. Set on every
// request so opencode and stackllm sign-ins can be distinguished on
// the server side.
const CodexOriginatorHeader = "originator"

// PKCE helpers generateCodeVerifier / generateCodeChallenge live in
// openai_web.go — they are shared between the two OAuth flows.
