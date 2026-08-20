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

func TestOpenAIDeviceSource_LoginErrors(t *testing.T) {
	t.Parallel()

	newSrc := func(store TokenStore, transport roundTripFunc) *OpenAIDeviceSource {
		return NewOpenAIDeviceSource(OpenAIDeviceConfig{
			ClientID:      "test-client",
			Store:         store,
			PollInterval:  time.Millisecond,
			HTTPClient:    &http.Client{Transport: transport},
			DeviceCodeURL: "https://test.openai.local/oauth/device/code",
			TokenURL:      "https://test.openai.local/oauth/token",
		})
	}

	deviceOK := func() (*http.Response, error) {
		return jsonResponse(map[string]any{
			"device_code":      "dev-abc",
			"user_code":        "CODE-0000",
			"verification_uri": "https://openai.com/verify",
			"interval":         0,
			"expires_in":       60,
		}), nil
	}

	t.Run("device code non-200", func(t *testing.T) {
		t.Parallel()
		src := newSrc(NewMemoryStore(), func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusBadRequest,
				Body:       io.NopCloser(strings.NewReader("bad client")),
				Header:     make(http.Header),
			}, nil
		})
		err := src.Login(context.Background())
		if err == nil || !strings.Contains(err.Error(), "status 400") {
			t.Errorf("Login error = %v, want status 400", err)
		}
	})

	t.Run("poll fatal error", func(t *testing.T) {
		t.Parallel()
		src := newSrc(NewMemoryStore(), func(req *http.Request) (*http.Response, error) {
			switch req.URL.String() {
			case "https://test.openai.local/oauth/device/code":
				return deviceOK()
			case "https://test.openai.local/oauth/token":
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
		src := newSrc(NewMemoryStore(), func(req *http.Request) (*http.Response, error) {
			switch req.URL.String() {
			case "https://test.openai.local/oauth/device/code":
				return deviceOK()
			case "https://test.openai.local/oauth/token":
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
		src := newSrc(NewMemoryStore(), func(req *http.Request) (*http.Response, error) {
			switch req.URL.String() {
			case "https://test.openai.local/oauth/device/code":
				return deviceOK()
			case "https://test.openai.local/oauth/token":
				polls.Add(1)
				return jsonResponse(map[string]any{"error": "slow_down"}), nil
			default:
				t.Errorf("unexpected URL: %s", req.URL)
				return nil, fmt.Errorf("unexpected URL")
			}
		})
		ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
		defer cancel()
		err := src.Login(ctx)
		if err == nil || !strings.Contains(err.Error(), context.DeadlineExceeded.Error()) {
			t.Errorf("Login error = %v, want context deadline exceeded", err)
		}
		if polls.Load() != 1 {
			t.Errorf("polls = %d, want exactly 1 (second wait should exceed deadline)", polls.Load())
		}
	})

	t.Run("context cancelled during wait", func(t *testing.T) {
		t.Parallel()
		src := NewOpenAIDeviceSource(OpenAIDeviceConfig{
			ClientID:     "test-client",
			Store:        NewMemoryStore(),
			PollInterval: time.Hour,
			HTTPClient: &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				return jsonResponse(map[string]any{
					"device_code":      "dev-abc",
					"user_code":        "CODE-1111",
					"verification_uri": "https://openai.com/verify",
					"interval":         0,
					"expires_in":       600,
				}), nil
			})},
			DeviceCodeURL: "https://test.openai.local/oauth/device/code",
			TokenURL:      "https://test.openai.local/oauth/token",
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
		src := newSrc(NewMemoryStore(), func(req *http.Request) (*http.Response, error) {
			switch req.URL.String() {
			case "https://test.openai.local/oauth/device/code":
				return jsonResponse(map[string]any{
					"device_code":      "dev-abc",
					"user_code":        "CODE-2222",
					"verification_uri": "https://openai.com/verify",
					"interval":         0,
					"expires_in":       1, // 1s deadline
				}), nil
			case "https://test.openai.local/oauth/token":
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

func TestOpenAIDeviceSource_OnCodePrefersCompleteURI(t *testing.T) {
	t.Parallel()

	var polls atomic.Int32
	var gotVerifyURL string
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		switch req.URL.String() {
		case "https://test.openai.local/oauth/device/code":
			return jsonResponse(map[string]any{
				"device_code":               "dev-abc",
				"user_code":                 "CODE-3333",
				"verification_uri":          "https://openai.com/verify",
				"verification_uri_complete": "https://openai.com/verify?user_code=CODE-3333",
				"interval":                  0,
				"expires_in":                60,
			}), nil
		case "https://test.openai.local/oauth/token":
			if polls.Add(1) == 1 {
				return jsonResponse(map[string]any{"error": "authorization_pending"}), nil
			}
			return jsonResponse(map[string]any{
				"access_token": "tok",
				"expires_in":   3600,
			}), nil
		default:
			t.Errorf("unexpected URL: %s", req.URL)
			return nil, fmt.Errorf("unexpected URL")
		}
	})}

	src := NewOpenAIDeviceSource(OpenAIDeviceConfig{
		ClientID:      "test-client",
		Store:         NewMemoryStore(),
		OnCode:        func(_, verifyURL string) { gotVerifyURL = verifyURL },
		PollInterval:  time.Millisecond,
		HTTPClient:    client,
		DeviceCodeURL: "https://test.openai.local/oauth/device/code",
		TokenURL:      "https://test.openai.local/oauth/token",
	})

	if err := src.Login(context.Background()); err != nil {
		t.Fatalf("Login: %v", err)
	}
	if gotVerifyURL != "https://openai.com/verify?user_code=CODE-3333" {
		t.Errorf("verifyURL = %q, want the verification_uri_complete value", gotVerifyURL)
	}
}

func TestOpenAIDeviceSource_TokenFallsBackToLoginWhenRefreshFails(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	ctx := context.Background()
	expired := openAITokenRecord{
		AccessToken:  "old-token",
		RefreshToken: "dead-refresh",
		ExpiresAt:    time.Now().Add(-time.Hour),
	}
	if err := saveOpenAITokenRecord(ctx, store, openaiStoreKey, expired); err != nil {
		t.Fatalf("seed: %v", err)
	}

	var refreshTried, loginPolled atomic.Int32
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		switch req.URL.String() {
		case "https://test.openai.local/oauth/device/code":
			return jsonResponse(map[string]any{
				"device_code":      "dev-abc",
				"user_code":        "CODE-4444",
				"verification_uri": "https://openai.com/verify",
				"interval":         0,
				"expires_in":       60,
			}), nil
		case "https://test.openai.local/oauth/token":
			if err := req.ParseForm(); err != nil {
				t.Errorf("ParseForm: %v", err)
			}
			switch req.Form.Get("grant_type") {
			case "refresh_token":
				refreshTried.Add(1)
				return &http.Response{
					StatusCode: http.StatusBadRequest,
					Body:       io.NopCloser(strings.NewReader(`{"error":"invalid_grant"}`)),
					Header:     make(http.Header),
				}, nil
			default: // device_code poll
				loginPolled.Add(1)
				return jsonResponse(map[string]any{
					"access_token":  "relogin-token",
					"refresh_token": "relogin-refresh",
					"expires_in":    3600,
				}), nil
			}
		default:
			t.Errorf("unexpected URL: %s", req.URL)
			return nil, fmt.Errorf("unexpected URL")
		}
	})}

	src := NewOpenAIDeviceSource(OpenAIDeviceConfig{
		ClientID:      "test-client",
		Store:         store,
		PollInterval:  time.Millisecond,
		HTTPClient:    client,
		DeviceCodeURL: "https://test.openai.local/oauth/device/code",
		TokenURL:      "https://test.openai.local/oauth/token",
	})

	tok, err := src.Token(ctx)
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if refreshTried.Load() != 1 {
		t.Errorf("refresh attempts = %d, want 1", refreshTried.Load())
	}
	if loginPolled.Load() == 0 {
		t.Error("expected fallback to device-flow login after failed refresh")
	}
	if tok.AccessToken != "relogin-token" {
		t.Errorf("AccessToken = %q, want relogin-token", tok.AccessToken)
	}

	// The re-login result must round-trip through the store.
	record, err := loadOpenAITokenRecord(ctx, store, openaiStoreKey)
	if err != nil {
		t.Fatalf("loadOpenAITokenRecord: %v", err)
	}
	if record.AccessToken != "relogin-token" || record.RefreshToken != "relogin-refresh" {
		t.Errorf("stored record = %+v, want re-login tokens", record)
	}
}
