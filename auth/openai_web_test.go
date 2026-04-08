package auth

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestOpenAIWebFlowSource_BeginComplete(t *testing.T) {
	t.Parallel()

	// Mock token endpoint.
	tokenServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Fatalf("ParseForm error: %v", err)
		}
		if r.Form.Get("grant_type") != "authorization_code" {
			t.Errorf("grant_type = %q, want authorization_code", r.Form.Get("grant_type"))
		}
		if r.Form.Get("code") != "auth-code-123" {
			t.Errorf("code = %q, want auth-code-123", r.Form.Get("code"))
		}
		if r.Form.Get("code_verifier") == "" {
			t.Error("expected code_verifier")
		}
		json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "web-access-tok",
			"refresh_token": "web-refresh-tok",
			"expires_in":    3600,
		})
	}))
	defer tokenServer.Close()

	store := NewMemoryStore()
	var successCalled bool

	src := NewOpenAIWebFlowSource(OpenAIWebFlowConfig{
		ClientID:   "test-web-client",
		Store:      store,
		OnSuccess:  func() { successCalled = true },
		HTTPClient: tokenServer.Client(),
		AuthURL:    "https://test.openai.local/authorize",
		TokenURL:   tokenServer.URL,
	})

	ctx := context.Background()

	// Step 1: Begin the flow — get the auth URL.
	authURL, err := src.Begin(ctx, "http://localhost:9999/callback")
	if err != nil {
		t.Fatalf("Begin error: %v", err)
	}
	if authURL == "" {
		t.Fatal("expected non-empty authURL")
	}

	// Verify the auth URL uses our custom AuthURL.
	if len(authURL) < len("https://test.openai.local/authorize") {
		t.Fatalf("authURL too short: %q", authURL)
	}
	if authURL[:len("https://test.openai.local/authorize")] != "https://test.openai.local/authorize" {
		t.Errorf("authURL should start with test AuthURL, got %q", authURL)
	}

	// The flow state should be stored.
	src.mu.Lock()
	flow := src.flow
	src.mu.Unlock()
	if flow == nil {
		t.Fatal("expected flow state to be set after Begin")
	}

	// Step 2: Complete with the code (simulating the OAuth callback).
	err = src.Complete(ctx, flow.State, "auth-code-123")
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	if !successCalled {
		t.Error("expected OnSuccess callback")
	}

	// Verify the token is stored with expiry.
	record, err := loadOpenAITokenRecord(ctx, store, openaiWebStoreKey)
	if err != nil {
		t.Fatalf("loadOpenAITokenRecord error: %v", err)
	}
	if record.AccessToken != "web-access-tok" {
		t.Errorf("stored AccessToken = %q, want web-access-tok", record.AccessToken)
	}
	if record.RefreshToken != "web-refresh-tok" {
		t.Errorf("stored RefreshToken = %q, want web-refresh-tok", record.RefreshToken)
	}
	if record.ExpiresAt.IsZero() {
		t.Error("stored ExpiresAt should be non-zero")
	}
	if record.ExpiresAt.Before(time.Now()) {
		t.Error("stored ExpiresAt should be in the future")
	}

	// Flow state should be cleared after completion.
	src.mu.Lock()
	flowAfter := src.flow
	src.mu.Unlock()
	if flowAfter != nil {
		t.Error("expected flow state to be nil after Complete")
	}
}

func TestOpenAIWebFlowSource_CompleteStateMismatch(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	src := NewOpenAIWebFlowSource(OpenAIWebFlowConfig{
		ClientID: "test",
		Store:    store,
		AuthURL:  "https://test.openai.local/authorize",
		TokenURL: "https://test.openai.local/oauth/token",
	})

	ctx := context.Background()

	// Begin a flow.
	_, err := src.Begin(ctx, "http://localhost:9999/callback")
	if err != nil {
		t.Fatalf("Begin error: %v", err)
	}

	// Complete with wrong state.
	err = src.Complete(ctx, "wrong-state", "some-code")
	if err == nil {
		t.Fatal("expected state mismatch error")
	}
}

func TestOpenAIWebFlowSource_CompleteWithoutBegin(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	src := NewOpenAIWebFlowSource(OpenAIWebFlowConfig{
		ClientID: "test",
		Store:    store,
	})

	err := src.Complete(context.Background(), "any-state", "any-code")
	if err == nil {
		t.Fatal("expected 'no flow in progress' error")
	}
}

func TestOpenAIWebFlowSource_TokenRefreshes(t *testing.T) {
	t.Parallel()

	// Mock token endpoint that handles refresh.
	tokenServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Fatalf("ParseForm error: %v", err)
		}
		if r.Form.Get("grant_type") != "refresh_token" {
			t.Errorf("grant_type = %q, want refresh_token", r.Form.Get("grant_type"))
		}
		if r.Form.Get("refresh_token") != "web-refresh" {
			t.Errorf("refresh_token = %q, want web-refresh", r.Form.Get("refresh_token"))
		}
		json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "refreshed-web-tok",
			"refresh_token": "new-web-refresh",
			"expires_in":    7200,
		})
	}))
	defer tokenServer.Close()

	store := NewMemoryStore()
	ctx := context.Background()

	// Pre-populate with expired token.
	expired := openAITokenRecord{
		AccessToken:  "old-web-token",
		RefreshToken: "web-refresh",
		ExpiresAt:    time.Now().Add(-1 * time.Hour),
	}
	saveOpenAITokenRecord(ctx, store, openaiWebStoreKey, expired)

	src := NewOpenAIWebFlowSource(OpenAIWebFlowConfig{
		ClientID:   "test",
		Store:      store,
		HTTPClient: tokenServer.Client(),
		TokenURL:   tokenServer.URL,
	})

	tok, err := src.Token(ctx)
	if err != nil {
		t.Fatalf("Token error: %v", err)
	}
	if tok.AccessToken != "refreshed-web-tok" {
		t.Errorf("AccessToken = %q, want refreshed-web-tok", tok.AccessToken)
	}

	// Verify updated record in store.
	record, err := loadOpenAITokenRecord(ctx, store, openaiWebStoreKey)
	if err != nil {
		t.Fatalf("loadOpenAITokenRecord error: %v", err)
	}
	if record.RefreshToken != "new-web-refresh" {
		t.Errorf("stored RefreshToken = %q, want new-web-refresh", record.RefreshToken)
	}
}

func TestOpenAIWebFlowSource_Logout(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	ctx := context.Background()
	saveOpenAITokenRecord(ctx, store, openaiWebStoreKey, openAITokenRecord{
		AccessToken:  "tok",
		RefreshToken: "ref",
	})

	src := NewOpenAIWebFlowSource(OpenAIWebFlowConfig{
		ClientID: "test",
		Store:    store,
		AuthURL:  "https://test.openai.local/authorize",
	})
	src.record = &openAITokenRecord{AccessToken: "tok"}

	// Start a flow so we can verify it's cleared on logout.
	src.Begin(ctx, "http://localhost:9999/callback")

	if err := src.Logout(ctx); err != nil {
		t.Fatalf("Logout error: %v", err)
	}
	if src.record != nil {
		t.Error("record should be nil after logout")
	}
	src.mu.Lock()
	f := src.flow
	src.mu.Unlock()
	if f != nil {
		t.Error("flow should be nil after logout")
	}
	if _, err := store.Load(ctx, openaiWebStoreKey); err == nil {
		t.Error("expected error loading after logout")
	}
}
