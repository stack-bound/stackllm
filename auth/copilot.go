package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const (
	copilotClientID    = "Iv1.b507a08c87ecfe98"
	copilotStoreKey    = "copilot_github_token"
	defaultGitHubHost  = "github.com"
	copilotTokenURL    = "https://api.github.com/copilot_internal/v2/token"
	copilotEditorVer   = "vscode/1.85.0"
	copilotPluginVer   = "copilot-chat/0.12.0"
	copilotIntegration = "vscode-chat"
	copilotUserAgent   = "GithubCopilot/1.0"
)

// CopilotConfig configures the two-phase Copilot auth flow.
type CopilotConfig struct {
	// Called during Phase 1 to display the one-time code to the user.
	// Must not block — return immediately after displaying.
	OnDeviceCode func(userCode, verifyURL string)

	// Called each time the poll loop checks for authorisation.
	OnPolling func()

	// Called when Phase 1 succeeds.
	OnSuccess func()

	// Token store for persisting the long-lived Phase 1 GitHub token.
	Store TokenStore

	// GitHub host. Defaults to "github.com". Override for GHE.
	Host string

	// HTTP client override for testing.
	HTTPClient *http.Client

	// Optional endpoint overrides for testing.
	DeviceCodeURL  string
	AccessTokenURL string
	CopilotTokenURL string
}

func (c *CopilotConfig) host() string {
	if c.Host != "" {
		return c.Host
	}
	return defaultGitHubHost
}

func (c *CopilotConfig) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return http.DefaultClient
}

func (c *CopilotConfig) deviceCodeURL() string {
	if c.DeviceCodeURL != "" {
		return c.DeviceCodeURL
	}
	return fmt.Sprintf("https://%s/login/device/code", c.host())
}

func (c *CopilotConfig) accessTokenURL() string {
	if c.AccessTokenURL != "" {
		return c.AccessTokenURL
	}
	return fmt.Sprintf("https://%s/login/oauth/access_token", c.host())
}

func (c *CopilotConfig) copilotTokenURL() string {
	if c.CopilotTokenURL != "" {
		return c.CopilotTokenURL
	}
	return copilotTokenURL
}

// CopilotTokenSource implements the two-phase GitHub → Copilot auth flow.
//
// Phase 1: GitHub OAuth device flow (RFC 8628)
// Phase 2: Copilot token exchange (~30 min TTL, auto-refreshed)
type CopilotTokenSource struct {
	cfg          CopilotConfig
	copilotToken *Token // cached Phase 2 token
}

// NewCopilotSource creates a new CopilotTokenSource.
func NewCopilotSource(cfg CopilotConfig) *CopilotTokenSource {
	return &CopilotTokenSource{cfg: cfg}
}

// Token returns a valid Copilot API token, performing auth flows as needed.
func (s *CopilotTokenSource) Token(ctx context.Context) (*Token, error) {
	// If we have a valid Phase 2 token, return it.
	if s.copilotToken.Valid() {
		return s.copilotToken, nil
	}

	// Get the Phase 1 GitHub token from store.
	ghToken, err := s.cfg.Store.Load(ctx, copilotStoreKey)
	if err != nil {
		// No stored token — need to login first.
		if err := s.Login(ctx); err != nil {
			return nil, fmt.Errorf("auth: copilot login: %w", err)
		}
		ghToken, err = s.cfg.Store.Load(ctx, copilotStoreKey)
		if err != nil {
			return nil, fmt.Errorf("auth: copilot load after login: %w", err)
		}
	}

	// Phase 2: exchange GitHub token for Copilot token.
	tok, err := s.exchangeForCopilotToken(ctx, ghToken)
	if err != nil {
		// If 401/403, the GitHub token is stale — delete and return error.
		_ = s.cfg.Store.Delete(ctx, copilotStoreKey)
		return nil, fmt.Errorf("auth: copilot token exchange failed (token cleared, re-authenticate): %w", err)
	}

	s.copilotToken = tok
	return tok, nil
}

// Login forces a Phase 1 re-authentication via the GitHub device flow.
func (s *CopilotTokenSource) Login(ctx context.Context) error {
	client := s.cfg.httpClient()

	// Step 1: Request device code.
	form := url.Values{
		"client_id": {copilotClientID},
		"scope":     {"read:user copilot"},
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.deviceCodeURL(), strings.NewReader(form.Encode()))
	if err != nil {
		return fmt.Errorf("auth: copilot device code request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("auth: copilot device code: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("auth: copilot device code: status %d: %s", resp.StatusCode, body)
	}

	var deviceResp struct {
		DeviceCode      string `json:"device_code"`
		UserCode        string `json:"user_code"`
		VerificationURI string `json:"verification_uri"`
		Interval        int    `json:"interval"`
		ExpiresIn       int    `json:"expires_in"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&deviceResp); err != nil {
		return fmt.Errorf("auth: copilot decode device response: %w", err)
	}

	// Display the code to the user.
	if s.cfg.OnDeviceCode != nil {
		s.cfg.OnDeviceCode(deviceResp.UserCode, deviceResp.VerificationURI)
	}

	// Step 2: Poll for authorization.
	interval := time.Duration(deviceResp.Interval) * time.Second
	if interval == 0 {
		interval = 5 * time.Second
	}
	deadline := time.Now().Add(time.Duration(deviceResp.ExpiresIn) * time.Second)
	if deviceResp.ExpiresIn == 0 {
		deadline = time.Now().Add(15 * time.Minute)
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

		pollForm := url.Values{
			"client_id":   {copilotClientID},
			"device_code": {deviceResp.DeviceCode},
			"grant_type":  {"urn:ietf:params:oauth:grant-type:device_code"},
		}

		pollReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.accessTokenURL(), strings.NewReader(pollForm.Encode()))
		if err != nil {
			return fmt.Errorf("auth: copilot poll request: %w", err)
		}
		pollReq.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		pollReq.Header.Set("Accept", "application/json")

		pollResp, err := client.Do(pollReq)
		if err != nil {
			continue
		}

		var tokenResp struct {
			AccessToken string `json:"access_token"`
			Error       string `json:"error"`
		}
		if err := json.NewDecoder(pollResp.Body).Decode(&tokenResp); err != nil {
			pollResp.Body.Close()
			continue
		}
		pollResp.Body.Close()

		switch tokenResp.Error {
		case "authorization_pending", "slow_down":
			if tokenResp.Error == "slow_down" {
				interval += 5 * time.Second
			}
			continue
		case "":
			// Success — we have the GitHub token.
			if tokenResp.AccessToken == "" {
				return fmt.Errorf("auth: copilot poll: empty access token")
			}
			if err := s.cfg.Store.Save(ctx, copilotStoreKey, tokenResp.AccessToken); err != nil {
				return fmt.Errorf("auth: copilot save token: %w", err)
			}
			if s.cfg.OnSuccess != nil {
				s.cfg.OnSuccess()
			}
			return nil
		default:
			return fmt.Errorf("auth: copilot poll: %s", tokenResp.Error)
		}
	}

	return fmt.Errorf("auth: copilot device flow timed out")
}

// Logout deletes the stored GitHub token.
func (s *CopilotTokenSource) Logout(ctx context.Context) error {
	s.copilotToken = nil
	return s.cfg.Store.Delete(ctx, copilotStoreKey)
}

// exchangeForCopilotToken performs Phase 2: exchanges a GitHub token for a
// short-lived Copilot API token.
func (s *CopilotTokenSource) exchangeForCopilotToken(ctx context.Context, ghToken string) (*Token, error) {
	client := s.cfg.httpClient()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.cfg.copilotTokenURL(), nil)
	if err != nil {
		return nil, fmt.Errorf("auth: copilot exchange request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+ghToken)
	req.Header.Set("Editor-Version", copilotEditorVer)
	req.Header.Set("Editor-Plugin-Version", copilotPluginVer)
	req.Header.Set("Copilot-Integration-Id", copilotIntegration)
	req.Header.Set("User-Agent", copilotUserAgent)

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("auth: copilot exchange: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("auth: copilot exchange: status %d: %s", resp.StatusCode, body)
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("auth: copilot exchange: status %d: %s", resp.StatusCode, body)
	}

	var tokenResp struct {
		Token     string `json:"token"`
		ExpiresAt int64  `json:"expires_at"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tokenResp); err != nil {
		return nil, fmt.Errorf("auth: copilot decode exchange response: %w", err)
	}

	return &Token{
		AccessToken: tokenResp.Token,
		ExpiresAt:   time.Unix(tokenResp.ExpiresAt, 0),
	}, nil
}
