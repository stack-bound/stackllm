package auth

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestCopilotConfig_HostAndURLs(t *testing.T) {
	t.Parallel()

	t.Run("default host", func(t *testing.T) {
		t.Parallel()
		cfg := &CopilotConfig{}
		if got := cfg.host(); got != defaultGitHubHost {
			t.Errorf("host() = %q, want %q", got, defaultGitHubHost)
		}
		wantDevice := fmt.Sprintf("https://%s/login/device/code", defaultGitHubHost)
		if got := cfg.deviceCodeURL(); got != wantDevice {
			t.Errorf("deviceCodeURL() = %q, want %q", got, wantDevice)
		}
		wantAccess := fmt.Sprintf("https://%s/login/oauth/access_token", defaultGitHubHost)
		if got := cfg.accessTokenURL(); got != wantAccess {
			t.Errorf("accessTokenURL() = %q, want %q", got, wantAccess)
		}
		if got := cfg.copilotTokenURL(); got != copilotTokenURL {
			t.Errorf("copilotTokenURL() = %q, want %q", got, copilotTokenURL)
		}
	})

	t.Run("GHE host override", func(t *testing.T) {
		t.Parallel()
		cfg := &CopilotConfig{Host: "github.example.com"}
		if got := cfg.host(); got != "github.example.com" {
			t.Errorf("host() = %q", got)
		}
		if got := cfg.deviceCodeURL(); got != "https://github.example.com/login/device/code" {
			t.Errorf("deviceCodeURL() = %q", got)
		}
		if got := cfg.accessTokenURL(); got != "https://github.example.com/login/oauth/access_token" {
			t.Errorf("accessTokenURL() = %q", got)
		}
	})
}

func TestCopilotTokenSource_ExchangeHonoursExpiry(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	ctx := context.Background()
	if err := store.Save(ctx, copilotStoreKey, "ghu_valid"); err != nil {
		t.Fatalf("seed: %v", err)
	}

	wantExpiry := time.Now().Add(30 * time.Minute).Unix()
	client := &http.Client{Transport: copilotRoundTripFunc(func(req *http.Request) (*http.Response, error) {
		return jsonResponse(map[string]any{
			"token":      "copilot-short-lived",
			"expires_at": wantExpiry,
		}), nil
	})}

	src := NewCopilotSource(CopilotConfig{
		Store:           store,
		HTTPClient:      client,
		CopilotTokenURL: "https://mcp.test/copilot_internal/v2/token",
	})

	tok, err := src.Token(ctx)
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if tok.AccessToken != "copilot-short-lived" {
		t.Errorf("AccessToken = %q", tok.AccessToken)
	}
	if !tok.ExpiresAt.Equal(time.Unix(wantExpiry, 0)) {
		t.Errorf("ExpiresAt = %v, want %v (expires_at honoured)", tok.ExpiresAt, time.Unix(wantExpiry, 0))
	}
	if !tok.Valid() {
		t.Error("exchanged token should be valid")
	}
}

func TestCopilotTokenSource_ExchangeServerError(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	ctx := context.Background()
	if err := store.Save(ctx, copilotStoreKey, "ghu_valid"); err != nil {
		t.Fatalf("seed: %v", err)
	}

	client := &http.Client{Transport: copilotRoundTripFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusInternalServerError,
			Body:       io.NopCloser(strings.NewReader("upstream broke")),
			Header:     make(http.Header),
		}, nil
	})}

	src := NewCopilotSource(CopilotConfig{
		Store:           store,
		HTTPClient:      client,
		CopilotTokenURL: "https://mcp.test/copilot_internal/v2/token",
	})

	_, err := src.Token(ctx)
	if err == nil {
		t.Fatal("expected exchange error")
	}
	if !strings.Contains(err.Error(), "status 500") {
		t.Errorf("error = %q, want status 500", err)
	}
}

func TestCopilotTokenSource_LoginErrors(t *testing.T) {
	t.Parallel()

	newSrc := func(transport copilotRoundTripFunc) *CopilotTokenSource {
		return NewCopilotSource(CopilotConfig{
			Store:           NewMemoryStore(),
			PollInterval:    time.Millisecond,
			HTTPClient:      &http.Client{Transport: transport},
			DeviceCodeURL:   "https://mcp.test/login/device/code",
			AccessTokenURL:  "https://mcp.test/login/oauth/access_token",
			CopilotTokenURL: "https://mcp.test/copilot_internal/v2/token",
		})
	}

	deviceOK := func() (*http.Response, error) {
		return jsonResponse(map[string]any{
			"device_code":      "dev-code",
			"user_code":        "ABCD-0000",
			"verification_uri": "https://github.com/login/device",
			"interval":         0,
			"expires_in":       60,
		}), nil
	}

	t.Run("device code non-200", func(t *testing.T) {
		t.Parallel()
		src := newSrc(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusServiceUnavailable,
				Body:       io.NopCloser(strings.NewReader("down")),
				Header:     make(http.Header),
			}, nil
		})
		err := src.Login(context.Background())
		if err == nil || !strings.Contains(err.Error(), "status 503") {
			t.Errorf("Login error = %v, want status 503", err)
		}
	})

	t.Run("poll fatal error", func(t *testing.T) {
		t.Parallel()
		src := newSrc(func(req *http.Request) (*http.Response, error) {
			switch req.URL.String() {
			case "https://mcp.test/login/device/code":
				return deviceOK()
			case "https://mcp.test/login/oauth/access_token":
				return jsonResponse(map[string]any{"error": "access_denied"}), nil
			default:
				t.Errorf("unexpected URL: %s", req.URL)
				return nil, fmt.Errorf("unexpected URL")
			}
		})
		err := src.Login(context.Background())
		if err == nil || !strings.Contains(err.Error(), "access_denied") {
			t.Errorf("Login error = %v, want access_denied", err)
		}
	})

	t.Run("empty access token", func(t *testing.T) {
		t.Parallel()
		src := newSrc(func(req *http.Request) (*http.Response, error) {
			switch req.URL.String() {
			case "https://mcp.test/login/device/code":
				return deviceOK()
			case "https://mcp.test/login/oauth/access_token":
				return jsonResponse(map[string]any{}), nil
			default:
				t.Errorf("unexpected URL: %s", req.URL)
				return nil, fmt.Errorf("unexpected URL")
			}
		})
		err := src.Login(context.Background())
		if err == nil || !strings.Contains(err.Error(), "empty access token") {
			t.Errorf("Login error = %v, want empty access token", err)
		}
	})

	t.Run("slow_down grows the interval", func(t *testing.T) {
		t.Parallel()
		var polls atomic.Int32
		src := newSrc(func(req *http.Request) (*http.Response, error) {
			switch req.URL.String() {
			case "https://mcp.test/login/device/code":
				return deviceOK()
			case "https://mcp.test/login/oauth/access_token":
				polls.Add(1)
				return jsonResponse(map[string]any{"error": "slow_down"}), nil
			default:
				t.Errorf("unexpected URL: %s", req.URL)
				return nil, fmt.Errorf("unexpected URL")
			}
		})
		// After the first slow_down the interval jumps to ~5s; the
		// context deadline fires during that longer wait, proving the
		// backoff was applied.
		ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
		defer cancel()
		start := time.Now()
		err := src.Login(ctx)
		if err == nil || !strings.Contains(err.Error(), context.DeadlineExceeded.Error()) {
			t.Errorf("Login error = %v, want context deadline exceeded", err)
		}
		if polls.Load() != 1 {
			t.Errorf("polls = %d, want exactly 1 (second wait should exceed deadline)", polls.Load())
		}
		if elapsed := time.Since(start); elapsed > 3*time.Second {
			t.Errorf("Login took %v — deadline should have cut the slow_down wait short", elapsed)
		}
	})

	t.Run("context cancelled during wait", func(t *testing.T) {
		t.Parallel()
		src := NewCopilotSource(CopilotConfig{
			Store:        NewMemoryStore(),
			PollInterval: time.Hour,
			HTTPClient: &http.Client{Transport: copilotRoundTripFunc(func(req *http.Request) (*http.Response, error) {
				return jsonResponse(map[string]any{
					"device_code":      "dev-code",
					"user_code":        "ABCD-1111",
					"verification_uri": "https://github.com/login/device",
					"interval":         0,
					"expires_in":       600,
				}), nil
			})},
			DeviceCodeURL: "https://mcp.test/login/device/code",
		})
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
		defer cancel()
		err := src.Login(ctx)
		if err == nil || !strings.Contains(err.Error(), context.DeadlineExceeded.Error()) {
			t.Errorf("Login error = %v, want context deadline exceeded", err)
		}
	})

	t.Run("device flow timeout", func(t *testing.T) {
		t.Parallel()
		src := newSrc(func(req *http.Request) (*http.Response, error) {
			switch req.URL.String() {
			case "https://mcp.test/login/device/code":
				return jsonResponse(map[string]any{
					"device_code":      "dev-code",
					"user_code":        "ABCD-2222",
					"verification_uri": "https://github.com/login/device",
					"interval":         0,
					"expires_in":       1, // 1s deadline
				}), nil
			case "https://mcp.test/login/oauth/access_token":
				return jsonResponse(map[string]any{"error": "authorization_pending"}), nil
			default:
				t.Errorf("unexpected URL: %s", req.URL)
				return nil, fmt.Errorf("unexpected URL")
			}
		})
		err := src.Login(context.Background())
		if err == nil || !strings.Contains(err.Error(), "timed out") {
			t.Errorf("Login error = %v, want timed out", err)
		}
	})
}
