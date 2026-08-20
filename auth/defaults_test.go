package auth

import (
	"net/http"
	"testing"
	"time"
)

// TestConfigDefaults pins the default (zero-config) behaviour of every
// config getter to the production constants defined in the package, so
// an accidental change to a default endpoint or interval fails a test.
func TestConfigDefaults(t *testing.T) {
	t.Parallel()

	t.Run("copilot", func(t *testing.T) {
		t.Parallel()
		c := &CopilotConfig{}
		if got := c.pollInterval(); got != 5*time.Second {
			t.Errorf("pollInterval() = %v, want 5s default", got)
		}
		if got := c.httpClient(); got != http.DefaultClient {
			t.Error("httpClient() should default to http.DefaultClient")
		}
	})

	t.Run("codex device", func(t *testing.T) {
		t.Parallel()
		c := &CodexDeviceConfig{}
		if got := c.clientID(); got != CodexDefaultClientID {
			t.Errorf("clientID() = %q, want CodexDefaultClientID", got)
		}
		if got := c.httpClient(); got != http.DefaultClient {
			t.Error("httpClient() should default to http.DefaultClient")
		}
		if got := c.deviceCodeURL(); got != codexDeviceUserCode {
			t.Errorf("deviceCodeURL() = %q, want %q", got, codexDeviceUserCode)
		}
		if got := c.devicePollURL(); got != codexDevicePollURL {
			t.Errorf("devicePollURL() = %q, want %q", got, codexDevicePollURL)
		}
		if got := c.tokenURL(); got != codexTokenURL {
			t.Errorf("tokenURL() = %q, want %q", got, codexTokenURL)
		}
		if got := c.verifyURL(); got != codexDeviceVerifyURL {
			t.Errorf("verifyURL() = %q, want %q", got, codexDeviceVerifyURL)
		}
		if got := c.redirectURI(); got != codexDeviceRedirect {
			t.Errorf("redirectURI() = %q, want %q", got, codexDeviceRedirect)
		}
		if got := c.pollInterval(); got != 2*time.Second {
			t.Errorf("pollInterval() = %v, want 2s default", got)
		}
	})

	t.Run("codex web", func(t *testing.T) {
		t.Parallel()
		c := &CodexWebFlowConfig{}
		if got := c.clientID(); got != CodexDefaultClientID {
			t.Errorf("clientID() = %q, want CodexDefaultClientID", got)
		}
		if got := c.httpClient(); got != http.DefaultClient {
			t.Error("httpClient() should default to http.DefaultClient")
		}
		if got := c.authURL(); got != codexAuthorizeURL {
			t.Errorf("authURL() = %q, want %q", got, codexAuthorizeURL)
		}
		if got := c.tokenURL(); got != codexTokenURL {
			t.Errorf("tokenURL() = %q, want %q", got, codexTokenURL)
		}
	})

	t.Run("openai device", func(t *testing.T) {
		t.Parallel()
		c := &OpenAIDeviceConfig{}
		if got := c.pollInterval(); got != 5*time.Second {
			t.Errorf("pollInterval() = %v, want 5s default", got)
		}
		if got := c.httpClient(); got != http.DefaultClient {
			t.Error("httpClient() should default to http.DefaultClient")
		}
		if got := c.deviceCodeURL(); got != openaiDeviceCodeURL {
			t.Errorf("deviceCodeURL() = %q, want %q", got, openaiDeviceCodeURL)
		}
		if got := c.tokenURL(); got != openaiTokenURL {
			t.Errorf("tokenURL() = %q, want %q", got, openaiTokenURL)
		}
	})

	t.Run("openai web", func(t *testing.T) {
		t.Parallel()
		c := &OpenAIWebFlowConfig{}
		if got := c.httpClient(); got != http.DefaultClient {
			t.Error("httpClient() should default to http.DefaultClient")
		}
		if got := c.authURL(); got != openaiAuthURL {
			t.Errorf("authURL() = %q, want %q", got, openaiAuthURL)
		}
		if got := c.tokenURL(); got != openaiWebTokenURL {
			t.Errorf("tokenURL() = %q, want %q", got, openaiWebTokenURL)
		}
	})
}
