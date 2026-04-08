package auth

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"sync/atomic"
	"testing"
	"time"
)

func TestCopilotTokenSource_FullFlow(t *testing.T) {
	store := NewMemoryStore()
	var polls atomic.Int32
	var deviceCodeCalled bool
	var successCalled bool

	client := &http.Client{Transport: copilotRoundTripFunc(func(req *http.Request) (*http.Response, error) {
		switch req.URL.String() {
		case "https://mcp.test/login/device/code":
			return jsonResponse(map[string]any{
				"device_code":      "dev-code",
				"user_code":        "ABCD-1234",
				"verification_uri": "https://github.com/login/device",
				"interval":         0,
				"expires_in":       60,
			}), nil
		case "https://mcp.test/login/oauth/access_token":
			if polls.Add(1) == 1 {
				return jsonResponse(map[string]any{"error": "authorization_pending"}), nil
			}
			return jsonResponse(map[string]any{"access_token": "ghu_token"}), nil
		case "https://mcp.test/copilot_internal/v2/token":
			if req.Header.Get("Authorization") != "Bearer ghu_token" {
				t.Fatalf("Authorization = %q", req.Header.Get("Authorization"))
			}
			return jsonResponse(map[string]any{
				"token":      "copilot-token",
				"expires_at": time.Now().Add(30 * time.Minute).Unix(),
			}), nil
		default:
			t.Fatalf("unexpected URL: %s", req.URL.String())
			return nil, nil
		}
	})}

	src := NewCopilotSource(CopilotConfig{
		OnDeviceCode: func(userCode, verifyURL string) {
			deviceCodeCalled = true
			if userCode != "ABCD-1234" || verifyURL == "" {
				t.Fatalf("device callback = %q %q", userCode, verifyURL)
			}
		},
		OnSuccess: func() { successCalled = true },
		Store:     store,
		HTTPClient: client,
		DeviceCodeURL:  "https://mcp.test/login/device/code",
		AccessTokenURL: "https://mcp.test/login/oauth/access_token",
		CopilotTokenURL: "https://mcp.test/copilot_internal/v2/token",
	})

	tok, err := src.Token(context.Background())
	if err != nil {
		t.Fatalf("Token error: %v", err)
	}
	if tok.AccessToken != "copilot-token" {
		t.Fatalf("AccessToken = %q", tok.AccessToken)
	}
	if !deviceCodeCalled {
		t.Fatal("expected device code callback")
	}
	if !successCalled {
		t.Fatal("expected success callback")
	}
}

func TestCopilotTokenSource_ClearsStaleGitHubToken(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()
	if err := store.Save(ctx, copilotStoreKey, "ghu_stale"); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	client := &http.Client{Transport: copilotRoundTripFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusUnauthorized,
			Body:       io.NopCloser(bytes.NewReader([]byte(`{"error":"unauthorized"}`))),
			Header:     make(http.Header),
		}, nil
	})}

	src := NewCopilotSource(CopilotConfig{
		Store:           store,
		HTTPClient:      client,
		CopilotTokenURL: "https://mcp.test/copilot_internal/v2/token",
	})

	if _, err := src.Token(ctx); err == nil {
		t.Fatal("expected token error")
	}
	if _, err := store.Load(ctx, copilotStoreKey); err == nil {
		t.Fatal("expected stale GitHub token to be deleted")
	}
}

func TestCopilotTokenSource_Logout(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	ctx := context.Background()

	store.Save(ctx, copilotStoreKey, "ghu_to_delete")

	src := NewCopilotSource(CopilotConfig{Store: store})
	src.copilotToken = &Token{AccessToken: "cached"}

	if err := src.Logout(ctx); err != nil {
		t.Fatalf("Logout error: %v", err)
	}
	if src.copilotToken != nil {
		t.Error("copilotToken should be nil after logout")
	}
	if _, err := store.Load(ctx, copilotStoreKey); err == nil {
		t.Error("expected error loading after logout")
	}
}

type copilotRoundTripFunc func(*http.Request) (*http.Response, error)

func (fn copilotRoundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}

func jsonResponse(v any) *http.Response {
	data, _ := json.Marshal(v)
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewReader(data)),
		Header:     make(http.Header),
	}
}
