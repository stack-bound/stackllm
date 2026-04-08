package auth

import "context"

// staticTokenSource returns the same token every time. Never expires.
type staticTokenSource struct {
	token *Token
}

// NewStatic creates a TokenSource that always returns the given token string.
// Use for API keys (OpenAI, Gemini, Azure, Ollama).
func NewStatic(token string) TokenSource {
	return &staticTokenSource{
		token: &Token{AccessToken: token},
	}
}

func (s *staticTokenSource) Token(_ context.Context) (*Token, error) {
	return s.token, nil
}
