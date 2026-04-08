package auth

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestToken_Valid(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		token *Token
		want  bool
	}{
		{"nil token", nil, false},
		{"empty access token", &Token{}, false},
		{"never expires", &Token{AccessToken: "abc"}, true},
		{"expires in future", &Token{AccessToken: "abc", ExpiresAt: time.Now().Add(5 * time.Minute)}, true},
		{"expires in 10 seconds (within buffer)", &Token{AccessToken: "abc", ExpiresAt: time.Now().Add(10 * time.Second)}, false},
		{"already expired", &Token{AccessToken: "abc", ExpiresAt: time.Now().Add(-1 * time.Minute)}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := tt.token.Valid(); got != tt.want {
				t.Errorf("Valid() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestStaticTokenSource(t *testing.T) {
	t.Parallel()

	src := NewStatic("sk-test-key")
	tok, err := src.Token(context.Background())
	if err != nil {
		t.Fatalf("Token() error = %v", err)
	}
	if tok.AccessToken != "sk-test-key" {
		t.Errorf("AccessToken = %q, want %q", tok.AccessToken, "sk-test-key")
	}
	if !tok.Valid() {
		t.Error("static token should be valid")
	}

	// Second call returns same token.
	tok2, _ := src.Token(context.Background())
	if tok2.AccessToken != tok.AccessToken {
		t.Error("static source should return same token")
	}
}

// mockTokenSource is a test helper that counts calls.
type mockTokenSource struct {
	mu    sync.Mutex
	calls int
	token *Token
	err   error
}

func (m *mockTokenSource) Token(_ context.Context) (*Token, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.calls++
	return m.token, m.err
}

func (m *mockTokenSource) callCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.calls
}

func TestCachingSource_CachesValidToken(t *testing.T) {
	t.Parallel()

	inner := &mockTokenSource{
		token: &Token{AccessToken: "cached", ExpiresAt: time.Now().Add(10 * time.Minute)},
	}
	src := NewCachingSource(inner)

	// First call fetches from inner.
	tok1, err := src.Token(context.Background())
	if err != nil {
		t.Fatalf("Token() error = %v", err)
	}
	if tok1.AccessToken != "cached" {
		t.Errorf("AccessToken = %q, want %q", tok1.AccessToken, "cached")
	}

	// Second call should use cache — inner not called again.
	tok2, err := src.Token(context.Background())
	if err != nil {
		t.Fatalf("Token() error = %v", err)
	}
	if tok2.AccessToken != "cached" {
		t.Errorf("AccessToken = %q, want %q", tok2.AccessToken, "cached")
	}

	if inner.callCount() != 1 {
		t.Errorf("inner called %d times, want 1", inner.callCount())
	}
}

func TestCachingSource_RefreshesExpiredToken(t *testing.T) {
	t.Parallel()

	inner := &mockTokenSource{
		token: &Token{AccessToken: "expired", ExpiresAt: time.Now().Add(-1 * time.Minute)},
	}
	src := NewCachingSource(inner)

	// First call — token is expired, so inner is called.
	_, err := src.Token(context.Background())
	if err != nil {
		t.Fatalf("Token() error = %v", err)
	}

	// Update inner to return a fresh token.
	inner.mu.Lock()
	inner.token = &Token{AccessToken: "fresh", ExpiresAt: time.Now().Add(10 * time.Minute)}
	inner.mu.Unlock()

	// Second call — cached token is expired, so inner is called again.
	tok, err := src.Token(context.Background())
	if err != nil {
		t.Fatalf("Token() error = %v", err)
	}
	if tok.AccessToken != "fresh" {
		t.Errorf("AccessToken = %q, want %q", tok.AccessToken, "fresh")
	}
	if inner.callCount() != 2 {
		t.Errorf("inner called %d times, want 2", inner.callCount())
	}
}

func TestCachingSource_PropagatesError(t *testing.T) {
	t.Parallel()

	inner := &mockTokenSource{
		err: fmt.Errorf("network error"),
	}
	src := NewCachingSource(inner)

	_, err := src.Token(context.Background())
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestCachingSource_ThreadSafety(t *testing.T) {
	t.Parallel()

	inner := &mockTokenSource{
		token: &Token{AccessToken: "safe", ExpiresAt: time.Now().Add(10 * time.Minute)},
	}
	src := NewCachingSource(inner)

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			tok, err := src.Token(context.Background())
			if err != nil {
				t.Errorf("Token() error = %v", err)
				return
			}
			if tok.AccessToken != "safe" {
				t.Errorf("AccessToken = %q, want %q", tok.AccessToken, "safe")
			}
		}()
	}
	wg.Wait()
}
