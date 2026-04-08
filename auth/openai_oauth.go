package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type openAITokenRecord struct {
	AccessToken  string    `json:"access_token"`
	RefreshToken string    `json:"refresh_token,omitempty"`
	ExpiresAt    time.Time `json:"expires_at,omitempty"`
}

func (r openAITokenRecord) token() *Token {
	return &Token{
		AccessToken: r.AccessToken,
		ExpiresAt:   r.ExpiresAt,
	}
}

func loadOpenAITokenRecord(ctx context.Context, store TokenStore, key string) (*openAITokenRecord, error) {
	raw, err := store.Load(ctx, key)
	if err != nil {
		return nil, err
	}

	var record openAITokenRecord
	if err := json.Unmarshal([]byte(raw), &record); err == nil && record.AccessToken != "" {
		return &record, nil
	}

	// Backward compatibility for older store entries that persisted only the access token.
	if raw != "" {
		return &openAITokenRecord{AccessToken: raw}, nil
	}

	return nil, fmt.Errorf("auth: empty token record")
}

func saveOpenAITokenRecord(ctx context.Context, store TokenStore, key string, record openAITokenRecord) error {
	data, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("auth: marshal token record: %w", err)
	}
	if err := store.Save(ctx, key, string(data)); err != nil {
		return fmt.Errorf("auth: save token record: %w", err)
	}
	return nil
}

func exchangeOpenAIToken(ctx context.Context, client *http.Client, form url.Values) (*openAITokenRecord, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, openaiTokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, fmt.Errorf("auth: openai token request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("auth: openai token exchange: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("auth: openai token exchange: status %d: %s", resp.StatusCode, body)
	}

	var tokenResp struct {
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
		ExpiresIn    int    `json:"expires_in"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tokenResp); err != nil {
		return nil, fmt.Errorf("auth: openai decode token response: %w", err)
	}
	if tokenResp.AccessToken == "" {
		return nil, fmt.Errorf("auth: openai token exchange: empty access token")
	}

	record := &openAITokenRecord{
		AccessToken:  tokenResp.AccessToken,
		RefreshToken: tokenResp.RefreshToken,
	}
	if tokenResp.ExpiresIn > 0 {
		record.ExpiresAt = time.Now().Add(time.Duration(tokenResp.ExpiresIn) * time.Second)
	}
	return record, nil
}
