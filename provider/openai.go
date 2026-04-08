package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/stack-bound/stackllm/auth"
	"github.com/stack-bound/stackllm/conversation"
)

// Config configures the OpenAI-compatible provider.
type Config struct {
	BaseURL      string
	TokenSource  auth.TokenSource
	Model        string
	APIVersion   string            // Azure requires e.g. "2024-02-01"
	ExtraHeaders map[string]string // static headers added to every request
	HTTPClient   *http.Client      // override for testing
	MaxRetries   int               // default 3; retries on 429 and 5xx
	BaseBackoff  time.Duration     // base for exponential backoff; default 1s
}

// OpenAIConfig returns config for the OpenAI API.
func OpenAIConfig(model string, ts auth.TokenSource) Config {
	return Config{
		BaseURL:     "https://api.openai.com/v1",
		TokenSource: ts,
		Model:       model,
		MaxRetries:  3,
	}
}

// AzureConfig returns config for Azure OpenAI.
func AzureConfig(endpoint, deployment, apiVersion string, ts auth.TokenSource) Config {
	return Config{
		BaseURL:     fmt.Sprintf("%s/openai/deployments/%s", endpoint, deployment),
		TokenSource: ts,
		APIVersion:  apiVersion,
		MaxRetries:  3,
	}
}

// OllamaConfig returns config for a local Ollama instance.
func OllamaConfig(baseURL, model string) Config {
	return Config{
		BaseURL:     baseURL + "/v1",
		TokenSource: auth.NewStatic("ollama"), // Ollama doesn't need auth
		Model:       model,
		MaxRetries:  1,
	}
}

// CopilotConfig returns config for GitHub Copilot.
func CopilotConfig(model string, ts auth.TokenSource) Config {
	return Config{
		BaseURL:     "https://api.githubcopilot.com",
		TokenSource: ts,
		Model:       model,
		ExtraHeaders: map[string]string{
			"Editor-Version":       "vscode/1.85.0",
			"Editor-Plugin-Version": "copilot-chat/0.12.0",
			"Copilot-Integration-Id": "vscode-chat",
		},
		MaxRetries: 3,
	}
}

// GeminiConfig returns config for Google Gemini via OpenAI-compat endpoint.
func GeminiConfig(model string, ts auth.TokenSource) Config {
	return Config{
		BaseURL:     "https://generativelanguage.googleapis.com/v1beta/openai",
		TokenSource: ts,
		Model:       model,
		MaxRetries:  3,
	}
}

// OpenAIProvider implements Provider using the OpenAI chat completions API.
type OpenAIProvider struct {
	cfg Config
}

// New creates a new OpenAI-compatible provider.
func New(cfg Config) *OpenAIProvider {
	if cfg.MaxRetries <= 0 {
		cfg.MaxRetries = 3
	}
	if cfg.BaseBackoff <= 0 {
		cfg.BaseBackoff = time.Second
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{
			Transport: newAuthRoundTripper(cfg.TokenSource, nil),
			Timeout:   5 * time.Minute,
		}
	} else if cfg.TokenSource != nil {
		// Wrap the existing client's transport with auth injection.
		cfg.HTTPClient = &http.Client{
			Transport:     newAuthRoundTripper(cfg.TokenSource, cfg.HTTPClient.Transport),
			Timeout:       cfg.HTTPClient.Timeout,
			CheckRedirect: cfg.HTTPClient.CheckRedirect,
			Jar:           cfg.HTTPClient.Jar,
		}
	}
	return &OpenAIProvider{cfg: cfg}
}

// Complete makes a streaming chat completion request and returns a channel of events.
func (p *OpenAIProvider) Complete(ctx context.Context, req Request) (<-chan Event, error) {
	body := p.buildRequestBody(req)

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("provider: marshal request: %w", err)
	}

	url := p.cfg.BaseURL + "/chat/completions"
	if p.cfg.APIVersion != "" {
		url += "?api-version=" + p.cfg.APIVersion
	}

	events := make(chan Event, 64)

	go func() {
		defer close(events)
		p.doWithRetry(ctx, url, jsonBody, events)
	}()

	return events, nil
}

// Models returns available model names.
func (p *OpenAIProvider) Models(ctx context.Context) ([]string, error) {
	url := p.cfg.BaseURL + "/models"
	if p.cfg.APIVersion != "" {
		url += "?api-version=" + p.cfg.APIVersion
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("provider: models request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.cfg.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("provider: models: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("provider: models: status %d: %s", resp.StatusCode, body)
	}

	var result struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("provider: models decode: %w", err)
	}

	models := make([]string, len(result.Data))
	for i, m := range result.Data {
		models[i] = m.ID
	}
	return models, nil
}

func (p *OpenAIProvider) buildRequestBody(req Request) map[string]any {
	// Convert messages to OpenAI format.
	msgs := make([]map[string]any, len(req.Messages))
	for i, m := range req.Messages {
		msg := map[string]any{
			"role":    string(m.Role),
			"content": m.Content,
		}
		if m.ToolCallID != "" {
			msg["tool_call_id"] = m.ToolCallID
		}
		if len(m.ToolCalls) > 0 {
			tcs := make([]map[string]any, len(m.ToolCalls))
			for j, tc := range m.ToolCalls {
				tcs[j] = map[string]any{
					"id":   tc.ID,
					"type": "function",
					"function": map[string]any{
						"name":      tc.Name,
						"arguments": tc.Arguments,
					},
				}
			}
			msg["tool_calls"] = tcs
		}
		msgs[i] = msg
	}

	body := map[string]any{
		"model":    req.Model,
		"messages": msgs,
		"stream":   req.Stream,
	}

	if req.Model == "" {
		body["model"] = p.cfg.Model
	}

	if req.MaxTokens > 0 {
		body["max_tokens"] = req.MaxTokens
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}

	// Convert tools to OpenAI format.
	if len(req.Tools) > 0 {
		oaiTools := make([]map[string]any, len(req.Tools))
		for i, t := range req.Tools {
			oaiTools[i] = map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        t.Name,
					"description": t.Description,
					"parameters":  t.Parameters,
				},
			}
		}
		body["tools"] = oaiTools
	}

	return body
}

func (p *OpenAIProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	for k, v := range p.cfg.ExtraHeaders {
		req.Header.Set(k, v)
	}
}

func (p *OpenAIProvider) doWithRetry(ctx context.Context, url string, body []byte, events chan<- Event) {
	var lastErr error
	for attempt := 0; attempt < p.cfg.MaxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(1<<uint(attempt-1)) * p.cfg.BaseBackoff
			select {
			case <-ctx.Done():
				events <- Event{Type: EventTypeError, Err: ctx.Err()}
				return
			case <-time.After(backoff):
			}
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			events <- Event{Type: EventTypeError, Err: fmt.Errorf("provider: create request: %w", err)}
			return
		}
		p.setHeaders(httpReq)

		resp, err := p.cfg.HTTPClient.Do(httpReq)
		if err != nil {
			lastErr = err
			continue
		}

		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
			resp.Body.Close()
			lastErr = fmt.Errorf("provider: status %d", resp.StatusCode)
			continue
		}

		if resp.StatusCode != http.StatusOK {
			respBody, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			events <- Event{Type: EventTypeError, Err: fmt.Errorf("provider: status %d: %s", resp.StatusCode, respBody)}
			return
		}

		p.readSSE(resp.Body, events)
		resp.Body.Close()
		return
	}

	events <- Event{Type: EventTypeError, Err: fmt.Errorf("provider: max retries exceeded: %w", lastErr)}
}

type toolCallAcc struct {
	ID        string
	Name      string
	Arguments strings.Builder
}

func (p *OpenAIProvider) readSSE(body io.Reader, events chan<- Event) {
	scanner := bufio.NewScanner(body)

	toolCalls := make(map[int]*toolCallAcc)
	var maxIndex int = -1

	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		if data == "[DONE]" {
			// Emit accumulated tool calls in index order.
			for i := 0; i <= maxIndex; i++ {
				tc, ok := toolCalls[i]
				if !ok {
					continue
				}
				events <- Event{
					Type: EventTypeToolCall,
					Call: &conversation.ToolCall{
						ID:        tc.ID,
						Name:      tc.Name,
						Arguments: tc.Arguments.String(),
					},
				}
			}
			events <- Event{Type: EventTypeDone}
			return
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content   string `json:"content"`
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}

		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		delta := chunk.Choices[0].Delta

		if delta.Content != "" {
			events <- Event{Type: EventTypeToken, Content: delta.Content}
		}

		for _, tc := range delta.ToolCalls {
			acc, ok := toolCalls[tc.Index]
			if !ok {
				acc = &toolCallAcc{}
				toolCalls[tc.Index] = acc
				if tc.Index > maxIndex {
					maxIndex = tc.Index
				}
			}
			if tc.ID != "" {
				acc.ID = tc.ID
			}
			if tc.Function.Name != "" {
				acc.Name = tc.Function.Name
			}
			if tc.Function.Arguments != "" {
				acc.Arguments.WriteString(tc.Function.Arguments)
			}
		}
	}
}
