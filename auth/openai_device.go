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
	openaiDeviceCodeURL = "https://auth0.openai.com/oauth/device/code"
	openaiTokenURL      = "https://auth0.openai.com/oauth/token"
	openaiStoreKey      = "openai_token"
)

// OpenAIDeviceConfig configures the OpenAI device code flow.
type OpenAIDeviceConfig struct {
	ClientID string
	Store    TokenStore

	// Called with the user code and verification URL to display to the user.
	OnCode    func(userCode, verifyURL string)
	OnPolling func()
	OnSuccess func()

	// PollInterval overrides the default device-flow poll interval (default 5s).
	// The server-provided interval takes precedence when non-zero.
	PollInterval time.Duration

	// HTTP client override for testing.
	HTTPClient *http.Client

	// Optional endpoint overrides for testing.
	DeviceCodeURL string
	TokenURL      string
}

func (c *OpenAIDeviceConfig) pollInterval() time.Duration {
	if c.PollInterval > 0 {
		return c.PollInterval
	}
	return 5 * time.Second
}

func (c *OpenAIDeviceConfig) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return http.DefaultClient
}

func (c *OpenAIDeviceConfig) deviceCodeURL() string {
	if c.DeviceCodeURL != "" {
		return c.DeviceCodeURL
	}
	return openaiDeviceCodeURL
}

func (c *OpenAIDeviceConfig) tokenURL() string {
	if c.TokenURL != "" {
		return c.TokenURL
	}
	return openaiTokenURL
}

// OpenAIDeviceSource implements the OpenAI device code OAuth flow.
type OpenAIDeviceSource struct {
	cfg    OpenAIDeviceConfig
	record *openAITokenRecord
}

// NewOpenAIDeviceSource creates a new OpenAIDeviceSource.
func NewOpenAIDeviceSource(cfg OpenAIDeviceConfig) *OpenAIDeviceSource {
	return &OpenAIDeviceSource{cfg: cfg}
}

// Token returns a valid OpenAI token, performing the device flow if needed.
func (s *OpenAIDeviceSource) Token(ctx context.Context) (*Token, error) {
	if s.record != nil && s.record.token().Valid() {
		return s.record.token(), nil
	}

	record, err := loadOpenAITokenRecord(ctx, s.cfg.Store, openaiStoreKey)
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

	record, err = loadOpenAITokenRecord(ctx, s.cfg.Store, openaiStoreKey)
	if err != nil {
		return nil, fmt.Errorf("auth: openai load after login: %w", err)
	}
	s.record = record
	return s.record.token(), nil
}

// Login performs the device code flow.
func (s *OpenAIDeviceSource) Login(ctx context.Context) error {
	client := s.cfg.httpClient()

	// Step 1: Request device code.
	form := url.Values{
		"client_id": {s.cfg.ClientID},
		"scope":     {"openid profile email offline_access"},
		"audience":  {"https://api.openai.com/v1"},
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.deviceCodeURL(), strings.NewReader(form.Encode()))
	if err != nil {
		return fmt.Errorf("auth: openai device code request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("auth: openai device code: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("auth: openai device code: status %d: %s", resp.StatusCode, body)
	}

	var deviceResp struct {
		DeviceCode              string `json:"device_code"`
		UserCode                string `json:"user_code"`
		VerificationURI         string `json:"verification_uri"`
		VerificationURIComplete string `json:"verification_uri_complete"`
		Interval                int    `json:"interval"`
		ExpiresIn               int    `json:"expires_in"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&deviceResp); err != nil {
		return fmt.Errorf("auth: openai decode device response: %w", err)
	}

	if s.cfg.OnCode != nil {
		verifyURL := deviceResp.VerificationURIComplete
		if verifyURL == "" {
			verifyURL = deviceResp.VerificationURI
		}
		s.cfg.OnCode(deviceResp.UserCode, verifyURL)
	}

	// Step 2: Poll for token.
	interval := time.Duration(deviceResp.Interval) * time.Second
	if interval == 0 {
		interval = s.cfg.pollInterval()
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
			"client_id":   {s.cfg.ClientID},
			"device_code": {deviceResp.DeviceCode},
			"grant_type":  {"urn:ietf:params:oauth:grant-type:device_code"},
		}

		pollReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.tokenURL(), strings.NewReader(pollForm.Encode()))
		if err != nil {
			return fmt.Errorf("auth: openai poll request: %w", err)
		}
		pollReq.Header.Set("Content-Type", "application/x-www-form-urlencoded")

		pollResp, err := client.Do(pollReq)
		if err != nil {
			continue
		}

		var tokenResp struct {
			AccessToken  string `json:"access_token"`
			RefreshToken string `json:"refresh_token"`
			ExpiresIn    int    `json:"expires_in"`
			Error        string `json:"error"`
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
			if tokenResp.AccessToken == "" {
				return fmt.Errorf("auth: openai poll: empty access token")
			}
			record := openAITokenRecord{
				AccessToken:  tokenResp.AccessToken,
				RefreshToken: tokenResp.RefreshToken,
			}
			if tokenResp.ExpiresIn > 0 {
				record.ExpiresAt = time.Now().Add(time.Duration(tokenResp.ExpiresIn) * time.Second)
			}
			if err := saveOpenAITokenRecord(ctx, s.cfg.Store, openaiStoreKey, record); err != nil {
				return err
			}
			s.record = &record
			if s.cfg.OnSuccess != nil {
				s.cfg.OnSuccess()
			}
			return nil
		default:
			return fmt.Errorf("auth: openai poll: %s", tokenResp.Error)
		}
	}

	return fmt.Errorf("auth: openai device flow timed out")
}

// Logout deletes the stored token.
func (s *OpenAIDeviceSource) Logout(ctx context.Context) error {
	s.record = nil
	return s.cfg.Store.Delete(ctx, openaiStoreKey)
}

func (s *OpenAIDeviceSource) refresh(ctx context.Context, refreshToken string) (*openAITokenRecord, error) {
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
	if err := saveOpenAITokenRecord(ctx, s.cfg.Store, openaiStoreKey, *record); err != nil {
		return nil, err
	}
	return record, nil
}
