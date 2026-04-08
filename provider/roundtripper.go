package provider

import (
	"net/http"

	"github.com/stack-bound/stackllm/auth"
)

// authRoundTripper injects the Authorization header from a TokenSource
// before each request.
type authRoundTripper struct {
	inner  http.RoundTripper
	source auth.TokenSource
}

func newAuthRoundTripper(source auth.TokenSource, inner http.RoundTripper) http.RoundTripper {
	if inner == nil {
		inner = http.DefaultTransport
	}
	return &authRoundTripper{inner: inner, source: source}
}

func (rt *authRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	tok, err := rt.source.Token(req.Context())
	if err != nil {
		return nil, err
	}
	req = req.Clone(req.Context())
	req.Header.Set("Authorization", "Bearer "+tok.AccessToken)
	return rt.inner.RoundTrip(req)
}
