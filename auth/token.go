package auth

import (
	"context"
	"sync"
	"time"
)

// Token represents an authentication token with optional expiry.
type Token struct {
	AccessToken string
	ExpiresAt   time.Time // zero value means never expires
}

// Valid reports whether the token is usable. A token is invalid if it is nil,
// has an empty access token, or will expire within 30 seconds.
func (t *Token) Valid() bool {
	if t == nil || t.AccessToken == "" {
		return false
	}
	if t.ExpiresAt.IsZero() {
		return true
	}
	return time.Now().Before(t.ExpiresAt.Add(-30 * time.Second))
}

// TokenSource is anything that can provide a token.
type TokenSource interface {
	Token(ctx context.Context) (*Token, error)
}

// CachingSource wraps a TokenSource and caches the token until it expires.
// All callers should wrap non-static sources in this.
type CachingSource struct {
	mu      sync.Mutex
	inner   TokenSource
	current *Token
}

// NewCachingSource creates a CachingSource that wraps the given TokenSource.
func NewCachingSource(inner TokenSource) *CachingSource {
	return &CachingSource{inner: inner}
}

// Token returns a cached token if still valid, otherwise fetches a new one.
func (c *CachingSource) Token(ctx context.Context) (*Token, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.current.Valid() {
		return c.current, nil
	}

	tok, err := c.inner.Token(ctx)
	if err != nil {
		return nil, err
	}
	c.current = tok
	return tok, nil
}
