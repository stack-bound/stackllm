package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/stack-bound/stackllm/auth"
	"github.com/stack-bound/stackllm/conversation"
)

// TestBuildRequestBody_MaxCompletionTokens asserts that the chat
// completions body carries max_completion_tokens (the parameter newer
// OpenAI/Azure models require) and never the legacy max_tokens, and
// that a zero MaxTokens sends neither.
func TestBuildRequestBody_MaxCompletionTokens(t *testing.T) {
	t.Parallel()

	p := New(Config{Model: "gpt-4o", TokenSource: auth.NewStatic("k")})

	tests := []struct {
		name      string
		maxTokens int
		wantMCT   bool
	}{
		{name: "positive sends max_completion_tokens", maxTokens: 128, wantMCT: true},
		{name: "zero sends neither parameter", maxTokens: 0, wantMCT: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body := p.buildRequestBody(Request{
				Messages:  []conversation.Message{userText("hi")},
				MaxTokens: tt.maxTokens,
			})

			if _, present := body["max_tokens"]; present {
				t.Errorf("body contains max_tokens = %v; the primary parameter must be max_completion_tokens", body["max_tokens"])
			}
			mct, present := body["max_completion_tokens"]
			if present != tt.wantMCT {
				t.Fatalf("max_completion_tokens present = %v, want %v", present, tt.wantMCT)
			}
			if tt.wantMCT && mct != tt.maxTokens {
				t.Errorf("max_completion_tokens = %v, want %d", mct, tt.maxTokens)
			}
		})
	}
}

// tokenParamRecorder records, per request, which token-limit parameter
// the wire body carried.
type tokenParamRecorder struct {
	mu     sync.Mutex
	bodies []map[string]any
}

func (rec *tokenParamRecorder) record(t *testing.T, r *http.Request) map[string]any {
	t.Helper()
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		t.Errorf("read request body: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Errorf("unmarshal request body: %v", err)
	}
	rec.mu.Lock()
	defer rec.mu.Unlock()
	rec.bodies = append(rec.bodies, body)
	return body
}

func (rec *tokenParamRecorder) snapshot() []map[string]any {
	rec.mu.Lock()
	defer rec.mu.Unlock()
	return append([]map[string]any(nil), rec.bodies...)
}

const unsupportedMCTBody = `{"error":{"message":"Unsupported parameter: 'max_completion_tokens' is not supported with this model. Use 'max_tokens' instead.","type":"invalid_request_error","param":"max_completion_tokens","code":"unsupported_parameter"}}`

// TestOpenAIProvider_MaxCompletionTokensFallback drives the full HTTP
// path against a fake backend that rejects max_completion_tokens the
// way older models/api-versions do, and asserts the provider retries
// once with max_tokens and streams the successful response.
func TestOpenAIProvider_MaxCompletionTokensFallback(t *testing.T) {
	t.Parallel()

	rec := &tokenParamRecorder{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := rec.record(t, r)
		if _, hasMCT := body["max_completion_tokens"]; hasMCT {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprint(w, unsupportedMCTBody)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()

	p := New(Config{
		BaseURL:     srv.URL,
		TokenSource: auth.NewStatic("k"),
		Model:       "gpt-4",
		MaxRetries:  1,
	})

	events, err := p.Complete(context.Background(), Request{
		Messages:  []conversation.Message{userText("hi")},
		MaxTokens: 64,
		Stream:    true,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var text strings.Builder
	var done bool
	for ev := range events {
		switch ev.Type {
		case EventTypeBlockDelta:
			if ev.BlockType == conversation.BlockText {
				text.WriteString(ev.Content)
			}
		case EventTypeDone:
			done = true
		case EventTypeError:
			t.Fatalf("unexpected error: %v", ev.Err)
		}
	}
	if !done {
		t.Error("expected done event after fallback retry")
	}
	if got := text.String(); got != "ok" {
		t.Errorf("streamed text = %q, want %q", got, "ok")
	}

	bodies := rec.snapshot()
	if len(bodies) != 2 {
		t.Fatalf("server saw %d requests, want 2 (primary + fallback)", len(bodies))
	}
	if got, want := bodies[0]["max_completion_tokens"], float64(64); got != want {
		t.Errorf("first request max_completion_tokens = %v, want %v", got, want)
	}
	if _, present := bodies[0]["max_tokens"]; present {
		t.Errorf("first request must not carry max_tokens, got %v", bodies[0]["max_tokens"])
	}
	if got, want := bodies[1]["max_tokens"], float64(64); got != want {
		t.Errorf("fallback request max_tokens = %v, want %v", got, want)
	}
	if _, present := bodies[1]["max_completion_tokens"]; present {
		t.Errorf("fallback request must not carry max_completion_tokens, got %v", bodies[1]["max_completion_tokens"])
	}
}

// TestOpenAIProvider_MaxCompletionTokensFallback_OnlyOnce asserts the
// parameter swap is one-shot: a backend that keeps answering 400 gets
// exactly two requests and the caller sees the error.
func TestOpenAIProvider_MaxCompletionTokensFallback_OnlyOnce(t *testing.T) {
	t.Parallel()

	rec := &tokenParamRecorder{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec.record(t, r)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, unsupportedMCTBody)
	}))
	defer srv.Close()

	p := New(Config{
		BaseURL:     srv.URL,
		TokenSource: auth.NewStatic("k"),
		Model:       "gpt-4",
		MaxRetries:  3,
	})

	events, err := p.Complete(context.Background(), Request{
		Messages:  []conversation.Message{userText("hi")},
		MaxTokens: 64,
		Stream:    true,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var gotErr error
	for ev := range events {
		if ev.Type == EventTypeError {
			gotErr = ev.Err
		}
	}
	if gotErr == nil {
		t.Fatal("expected an error event when the fallback is also rejected")
	}
	if !strings.Contains(gotErr.Error(), "status 400") {
		t.Errorf("error = %v, want status 400 surfaced", gotErr)
	}
	if got := len(rec.snapshot()); got != 2 {
		t.Errorf("server saw %d requests, want exactly 2 (no repeated fallback)", got)
	}
}

// TestOpenAIProvider_Unrelated400_NoFallback asserts a 400 that has
// nothing to do with max_completion_tokens is surfaced immediately
// without a retry.
func TestOpenAIProvider_Unrelated400_NoFallback(t *testing.T) {
	t.Parallel()

	rec := &tokenParamRecorder{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec.record(t, r)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":{"message":"Invalid value for 'temperature'","type":"invalid_request_error","param":"temperature","code":"invalid_value"}}`)
	}))
	defer srv.Close()

	p := New(Config{
		BaseURL:     srv.URL,
		TokenSource: auth.NewStatic("k"),
		Model:       "gpt-4",
		MaxRetries:  3,
	})

	events, err := p.Complete(context.Background(), Request{
		Messages:  []conversation.Message{userText("hi")},
		MaxTokens: 64,
		Stream:    true,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var gotErr error
	for ev := range events {
		if ev.Type == EventTypeError {
			gotErr = ev.Err
		}
	}
	if gotErr == nil {
		t.Fatal("expected an error event for the unrelated 400")
	}
	if got := len(rec.snapshot()); got != 1 {
		t.Errorf("server saw %d requests, want 1 (no fallback for unrelated 400)", got)
	}
}

func TestIsUnsupportedMaxCompletionTokens(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		body string
		want bool
	}{
		{
			name: "openai unsupported_parameter",
			body: unsupportedMCTBody,
			want: true,
		},
		{
			name: "azure unrecognized request argument",
			body: `{"error":{"message":"Unrecognized request argument supplied: max_completion_tokens","type":"invalid_request_error"}}`,
			want: true,
		},
		{
			name: "unknown parameter phrasing",
			body: `{"error":{"message":"Unknown parameter: max_completion_tokens"}}`,
			want: true,
		},
		{
			name: "mentions parameter but different failure",
			body: `{"error":{"message":"max_completion_tokens must be at least 1","code":"invalid_value"}}`,
			want: false,
		},
		{
			name: "unrelated error",
			body: `{"error":{"message":"Unsupported parameter: 'logit_bias'","code":"unsupported_parameter"}}`,
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := isUnsupportedMaxCompletionTokens([]byte(tt.body)); got != tt.want {
				t.Errorf("isUnsupportedMaxCompletionTokens(%q) = %v, want %v", tt.body, got, tt.want)
			}
		})
	}
}
