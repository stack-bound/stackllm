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

func TestOpenAIDeviceSource_FullFlow(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	var polls atomic.Int32
	var codeCalled, successCalled bool

	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		switch req.URL.String() {
		case "https://test.openai.local/oauth/device/code":
			return jsonResponse(map[string]any{
				"device_code":      "dev-abc",
				"user_code":        "WXYZ-9999",
				"verification_uri": "https://openai.com/verify",
				"interval":         1,
				"expires_in":       60,
			}), nil
		case "https://test.openai.local/oauth/token":
			if polls.Add(1) == 1 {
				return jsonResponse(map[string]any{"error": "authorization_pending"}), nil
			}
			return jsonResponse(map[string]any{
				"access_token":  "oai-access-tok",
				"refresh_token": "oai-refresh-tok",
				"expires_in":    3600,
			}), nil
		default:
			t.Fatalf("unexpected URL: %s", req.URL.String())
			return nil, nil
		}
	})}

	src := NewOpenAIDeviceSource(OpenAIDeviceConfig{
		ClientID: "test-client",
		Store:    store,
		OnCode: func(userCode, verifyURL string) {
			codeCalled = true
			if userCode != "WXYZ-9999" {
				t.Errorf("userCode = %q, want WXYZ-9999", userCode)
			}
			if verifyURL != "https://openai.com/verify" {
				t.Errorf("verifyURL = %q", verifyURL)
			}
		},
		OnSuccess:     func() { successCalled = true },
		HTTPClient:    client,
		DeviceCodeURL: "https://test.openai.local/oauth/device/code",
		TokenURL:      "https://test.openai.local/oauth/token",
	})

	tok, err := src.Token(context.Background())
	if err != nil {
		t.Fatalf("Token error: %v", err)
	}
	if tok.AccessToken != "oai-access-tok" {
		t.Errorf("AccessToken = %q, want oai-access-tok", tok.AccessToken)
	}
	if tok.ExpiresAt.IsZero() {
		t.Error("expected non-zero ExpiresAt")
	}
	if tok.ExpiresAt.Before(time.Now()) {
		t.Error("expected ExpiresAt in the future")
	}
	if !codeCalled {
		t.Error("expected OnCode callback")
	}
	if !successCalled {
		t.Error("expected OnSuccess callback")
	}

	// Verify the stored record has a refresh token.
	record, err := loadOpenAITokenRecord(context.Background(), store, openaiStoreKey)
	if err != nil {
		t.Fatalf("loadOpenAITokenRecord error: %v", err)
	}
	if record.RefreshToken != "oai-refresh-tok" {
		t.Errorf("stored RefreshToken = %q, want oai-refresh-tok", record.RefreshToken)
	}
	if record.ExpiresAt.IsZero() {
		t.Error("stored ExpiresAt should be non-zero")
	}
}

func TestOpenAIDeviceSource_TokenCachesAndRefreshes(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	ctx := context.Background()

	// Pre-populate store with an expired token that has a refresh token.
	expired := openAITokenRecord{
		AccessToken:  "old-token",
		RefreshToken: "the-refresh",
		ExpiresAt:    time.Now().Add(-1 * time.Hour),
	}
	saveOpenAITokenRecord(ctx, store, openaiStoreKey, expired)

	var refreshCalled bool
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		// This should be a refresh token request.
		if err := req.ParseForm(); err != nil {
			t.Fatalf("ParseForm error: %v", err)
		}
		if req.Form.Get("grant_type") != "refresh_token" {
			t.Fatalf("grant_type = %q, want refresh_token", req.Form.Get("grant_type"))
		}
		if req.Form.Get("refresh_token") != "the-refresh" {
			t.Fatalf("refresh_token = %q, want the-refresh", req.Form.Get("refresh_token"))
		}
		refreshCalled = true
		return jsonResponse(map[string]any{
			"access_token":  "refreshed-token",
			"refresh_token": "new-refresh",
			"expires_in":    7200,
		}), nil
	})}

	src := NewOpenAIDeviceSource(OpenAIDeviceConfig{
		ClientID:   "test-client",
		Store:      store,
		HTTPClient: client,
		TokenURL:   "https://test.openai.local/oauth/token",
	})

	tok, err := src.Token(ctx)
	if err != nil {
		t.Fatalf("Token error: %v", err)
	}
	if !refreshCalled {
		t.Fatal("expected refresh to be called for expired token")
	}
	if tok.AccessToken != "refreshed-token" {
		t.Errorf("AccessToken = %q, want refreshed-token", tok.AccessToken)
	}

	// Second call should use the cached token, not call refresh again.
	refreshCalled = false
	tok2, err := src.Token(ctx)
	if err != nil {
		t.Fatalf("Token error: %v", err)
	}
	if refreshCalled {
		t.Error("expected cached token, not another refresh")
	}
	if tok2.AccessToken != "refreshed-token" {
		t.Errorf("AccessToken = %q, want refreshed-token", tok2.AccessToken)
	}
}

func TestOpenAIDeviceSource_Logout(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	ctx := context.Background()
	store.Save(ctx, openaiStoreKey, `{"access_token":"tok","refresh_token":"ref"}`)

	src := NewOpenAIDeviceSource(OpenAIDeviceConfig{Store: store})
	src.record = &openAITokenRecord{AccessToken: "tok"}

	if err := src.Logout(ctx); err != nil {
		t.Fatalf("Logout error: %v", err)
	}
	if src.record != nil {
		t.Error("record should be nil after logout")
	}
	if _, err := store.Load(ctx, openaiStoreKey); err == nil {
		t.Error("expected error loading after logout")
	}
}

// roundTripFunc is a functional http.RoundTripper for tests.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}

// jsonResp creates a 200 JSON response for testing.
func jsonResp(v any) *http.Response {
	data, _ := json.Marshal(v)
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewReader(data)),
		Header:     make(http.Header),
	}
}
