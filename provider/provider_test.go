package provider

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/stack-bound/stackllm/auth"
	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/tools"
)

func TestOpenAIProvider_CompleteTextOnly(t *testing.T) {
	t.Parallel()

	sseData := `data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"content":" world"}}]}

data: [DONE]

`
	p := New(Config{
		BaseURL:     "http://provider.test/v1",
		TokenSource: auth.NewStatic("test-key"),
		Model:       "gpt-4",
		HTTPClient:  newTestClient(func(req *http.Request) (*http.Response, error) {
			if req.URL.Path != "/v1/chat/completions" {
				t.Errorf("unexpected path: %s", req.URL.Path)
			}
			if req.Header.Get("Authorization") != "Bearer test-key" {
				t.Errorf("Authorization = %q", req.Header.Get("Authorization"))
			}
			return textResponse(http.StatusOK, "text/event-stream", sseData), nil
		}),
		MaxRetries: 1,
	})

	events, err := p.Complete(context.Background(), Request{
		Messages: []conversation.Message{{Role: conversation.RoleUser, Content: "Hi"}},
		Stream:   true,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var tokens []string
	var done bool
	for ev := range events {
		switch ev.Type {
		case EventTypeToken:
			tokens = append(tokens, ev.Content)
		case EventTypeDone:
			done = true
		case EventTypeError:
			t.Fatalf("unexpected error: %v", ev.Err)
		}
	}

	if !done {
		t.Error("expected done event")
	}
	if got := strings.Join(tokens, ""); got != "Hello world" {
		t.Errorf("result = %q, want %q", got, "Hello world")
	}
}

func TestOpenAIProvider_CompleteWithToolCalls(t *testing.T) {
	t.Parallel()

	sseData := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read_file","arguments":""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"/tmp/test\"}"}}]}}]}

data: [DONE]

`
	p := New(Config{
		BaseURL:     "http://provider.test/v1",
		TokenSource: auth.NewStatic("key"),
		Model:       "gpt-4",
		HTTPClient:  newTestClient(func(req *http.Request) (*http.Response, error) { return textResponse(http.StatusOK, "text/event-stream", sseData), nil }),
		MaxRetries:  1,
	})

	events, err := p.Complete(context.Background(), Request{
		Messages: []conversation.Message{{Role: conversation.RoleUser, Content: "read /tmp/test"}},
		Tools:    []tools.Definition{{Name: "read_file", Description: "read a file"}},
		Stream:   true,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var toolCalls []ToolCall
	for ev := range events {
		switch ev.Type {
		case EventTypeToolCall:
			toolCalls = append(toolCalls, *ev.Call)
		case EventTypeError:
			t.Fatalf("unexpected error: %v", ev.Err)
		}
	}

	if len(toolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(toolCalls))
	}
	if toolCalls[0].Name != "read_file" || toolCalls[0].ID != "call_1" || toolCalls[0].Arguments != `{"path":"/tmp/test"}` {
		t.Fatalf("tool call = %#v", toolCalls[0])
	}
}

func TestOpenAIProvider_RetryOn429(t *testing.T) {
	t.Parallel()

	var attempts int
	p := New(Config{
		BaseURL:     "http://provider.test/v1",
		TokenSource: auth.NewStatic("key"),
		Model:       "gpt-4",
		HTTPClient: newTestClient(func(req *http.Request) (*http.Response, error) {
			attempts++
			if attempts < 3 {
				return textResponse(http.StatusTooManyRequests, "text/plain", ""), nil
			}
			return textResponse(http.StatusOK, "text/event-stream", "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n"), nil
		}),
		MaxRetries:  3,
		BaseBackoff: time.Millisecond,
	})

	events, err := p.Complete(context.Background(), Request{
		Messages: []conversation.Message{{Role: conversation.RoleUser, Content: "test"}},
		Stream:   true,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var gotContent bool
	for ev := range events {
		if ev.Type == EventTypeToken && ev.Content == "ok" {
			gotContent = true
		}
		if ev.Type == EventTypeError {
			t.Fatalf("unexpected error: %v", ev.Err)
		}
	}

	if !gotContent {
		t.Error("expected content after retry")
	}
	if attempts != 3 {
		t.Errorf("attempts = %d, want 3", attempts)
	}
}

func TestOpenAIProvider_ErrorResponse(t *testing.T) {
	t.Parallel()

	p := New(Config{
		BaseURL:     "http://provider.test/v1",
		TokenSource: auth.NewStatic("key"),
		Model:       "gpt-4",
		HTTPClient:  newTestClient(func(req *http.Request) (*http.Response, error) { return textResponse(http.StatusBadRequest, "application/json", `{"error":{"message":"bad request"}}`), nil }),
		MaxRetries:  1,
	})

	events, err := p.Complete(context.Background(), Request{
		Messages: []conversation.Message{{Role: conversation.RoleUser, Content: "test"}},
		Stream:   true,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var gotError bool
	for ev := range events {
		if ev.Type == EventTypeError {
			gotError = true
		}
	}
	if !gotError {
		t.Error("expected error event for 400 response")
	}
}

func TestAuthRoundTripper(t *testing.T) {
	t.Parallel()

	var receivedAuth string
	rt := newAuthRoundTripper(auth.NewStatic("my-token"), roundTripFunc(func(req *http.Request) (*http.Response, error) {
		receivedAuth = req.Header.Get("Authorization")
		return textResponse(http.StatusOK, "text/plain", ""), nil
	}))
	client := &http.Client{Transport: rt}

	resp, err := client.Get("http://provider.test")
	if err != nil {
		t.Fatalf("Get error: %v", err)
	}
	resp.Body.Close()

	if receivedAuth != "Bearer my-token" {
		t.Errorf("Authorization = %q, want %q", receivedAuth, "Bearer my-token")
	}
}

func TestProviderConfigs(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		cfg     Config
		baseURL string
	}{
		{"OpenAI", OpenAIConfig("gpt-4", auth.NewStatic("k")), "https://api.openai.com/v1"},
		{"Ollama", OllamaConfig("http://localhost:11434", "llama3"), "http://localhost:11434/v1"},
		{"Copilot", CopilotConfig("gpt-4", auth.NewStatic("k")), "https://api.githubcopilot.com"},
		{"Gemini", GeminiConfig("gemini-pro", auth.NewStatic("k")), "https://generativelanguage.googleapis.com/v1beta/openai"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if tt.cfg.BaseURL != tt.baseURL {
				t.Errorf("BaseURL = %q, want %q", tt.cfg.BaseURL, tt.baseURL)
			}
		})
	}
}

func TestOpenAIProvider_Models(t *testing.T) {
	t.Parallel()

	p := New(Config{
		BaseURL:     "http://provider.test/v1",
		TokenSource: auth.NewStatic("key"),
		HTTPClient: newTestClient(func(req *http.Request) (*http.Response, error) {
			if req.URL.Path != "/v1/models" {
				t.Errorf("unexpected path: %s", req.URL.Path)
			}
			return textResponse(http.StatusOK, "application/json", `{"data":[{"id":"gpt-4"},{"id":"gpt-3.5-turbo"}]}`), nil
		}),
	})

	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models error: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("got %d models, want 2", len(models))
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}

func newTestClient(fn roundTripFunc) *http.Client {
	return &http.Client{Transport: fn}
}

func textResponse(status int, contentType, body string) *http.Response {
	resp := &http.Response{
		StatusCode: status,
		Header:     make(http.Header),
		Body:       io.NopCloser(bytes.NewBufferString(body)),
	}
	if contentType != "" {
		resp.Header.Set("Content-Type", contentType)
	}
	return resp
}

func TestAzureConfig(t *testing.T) {
	t.Parallel()

	cfg := AzureConfig("https://example.openai.azure.com", "deployment", "2024-02-01", auth.NewStatic("k"))
	if got := cfg.BaseURL; got != "https://example.openai.azure.com/openai/deployments/deployment" {
		t.Fatalf("BaseURL = %q", got)
	}
	if got := cfg.APIVersion; got != "2024-02-01" {
		t.Fatalf("APIVersion = %q", got)
	}
}

func ExampleOpenAIConfig() {
	cfg := OpenAIConfig("gpt-4o", auth.NewStatic("key"))
	fmt.Println(cfg.BaseURL)
	// Output: https://api.openai.com/v1
}
