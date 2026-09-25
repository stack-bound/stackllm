package provider

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/stack-bound/stackllm/auth"
	"github.com/stack-bound/stackllm/conversation"
)

// extraBodyServer answers /chat/completions and /responses with a
// minimal successful stream and records every wire body it receives.
func extraBodyServer(t *testing.T, rec *tokenParamRecorder) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec.record(t, r)
		w.Header().Set("Content-Type", "text/event-stream")
		switch r.URL.Path {
		case "/chat/completions":
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n")
		case "/responses":
			fmt.Fprint(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{}}\n\n")
		default:
			t.Errorf("unexpected path %s", r.URL.Path)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func drain(t *testing.T, events <-chan Event) {
	t.Helper()
	for ev := range events {
		if ev.Type == EventTypeError {
			t.Fatalf("stream error: %v", ev.Err)
		}
	}
}

// TestOpenAIProvider_ExtraBody asserts the merge rules end to end on
// the wire, for both endpoints: config fields are sent, request fields
// override config per key, a nil request value removes a configured
// key, extras override provider-derived fields, and unrelated body
// fields are left intact.
func TestOpenAIProvider_ExtraBody(t *testing.T) {
	t.Parallel()

	routing := map[string]any{"order": []any{"groq", "together"}, "allow_fallbacks": false}

	tests := []struct {
		name     string
		cfgExtra map[string]any
		reqExtra map[string]any
		want     map[string]any // keys that must be on the wire with these values
		absent   []string       // keys that must not be on the wire
	}{
		{
			name:     "config only",
			cfgExtra: map[string]any{"provider": routing},
			want:     map[string]any{"provider": routing},
		},
		{
			name:     "request only",
			reqExtra: map[string]any{"provider": routing, "transforms": []any{"middle-out"}},
			want:     map[string]any{"provider": routing, "transforms": []any{"middle-out"}},
		},
		{
			name:     "request overrides config per key and keeps the rest",
			cfgExtra: map[string]any{"provider": map[string]any{"sort": "price"}, "user": "cfg-user"},
			reqExtra: map[string]any{"provider": routing},
			want:     map[string]any{"provider": routing, "user": "cfg-user"},
		},
		{
			name:     "nil request value removes a configured key",
			cfgExtra: map[string]any{"provider": routing, "user": "cfg-user"},
			reqExtra: map[string]any{"provider": nil},
			want:     map[string]any{"user": "cfg-user"},
			absent:   []string{"provider"},
		},
		{
			name:     "extras override provider-derived fields",
			reqExtra: map[string]any{"temperature": 0.25},
			want:     map[string]any{"temperature": 0.25},
		},
	}

	for _, endpoint := range []string{"", EndpointResponses} {
		for _, tt := range tests {
			t.Run(fmt.Sprintf("%s/%s", endpointName(endpoint), tt.name), func(t *testing.T) {
				t.Parallel()

				rec := &tokenParamRecorder{}
				srv := extraBodyServer(t, rec)
				p := New(Config{
					BaseURL:     srv.URL,
					TokenSource: auth.NewStatic("k"),
					Model:       "openai/gpt-4o",
					MaxRetries:  1,
					Endpoint:    endpoint,
					ExtraBody:   tt.cfgExtra,
				})
				temp := 0.9
				events, err := p.Complete(context.Background(), Request{
					Messages:    []conversation.Message{userText("hi")},
					Temperature: &temp,
					Stream:      true,
					ExtraBody:   tt.reqExtra,
				})
				if err != nil {
					t.Fatalf("Complete error: %v", err)
				}
				drain(t, events)

				bodies := rec.snapshot()
				if len(bodies) != 1 {
					t.Fatalf("server saw %d requests, want 1", len(bodies))
				}
				body := bodies[0]
				for k, want := range tt.want {
					if got := body[k]; !reflect.DeepEqual(got, want) {
						t.Errorf("body[%q] = %#v, want %#v", k, got, want)
					}
				}
				for _, k := range tt.absent {
					if v, present := body[k]; present {
						t.Errorf("body[%q] = %#v, want absent", k, v)
					}
				}
				// Typed fields the extras did not touch still reach the wire.
				if body["model"] != "openai/gpt-4o" {
					t.Errorf("model = %v, want openai/gpt-4o", body["model"])
				}
				if body["stream"] != true {
					t.Errorf("stream = %v, want true", body["stream"])
				}
				if _, overridden := tt.want["temperature"]; !overridden && body["temperature"] != 0.9 {
					t.Errorf("temperature = %v, want 0.9 from the typed field", body["temperature"])
				}
			})
		}
	}
}

func endpointName(endpoint string) string {
	if endpoint == "" {
		return "chat"
	}
	return "responses"
}

// TestOpenAIProvider_ExtraBody_ReservedKeys asserts that extras cannot
// replace the fields the provider derives from the conversation and
// stream mode: Complete returns an error naming the key and no request
// reaches the server.
func TestOpenAIProvider_ExtraBody_ReservedKeys(t *testing.T) {
	t.Parallel()

	for _, key := range []string{"model", "messages", "input", "stream"} {
		for _, inConfig := range []bool{true, false} {
			t.Run(fmt.Sprintf("%s/config=%v", key, inConfig), func(t *testing.T) {
				t.Parallel()

				rec := &tokenParamRecorder{}
				srv := extraBodyServer(t, rec)
				cfg := Config{BaseURL: srv.URL, TokenSource: auth.NewStatic("k"), Model: "m", MaxRetries: 1}
				req := Request{Messages: []conversation.Message{userText("hi")}, Stream: true}
				extra := map[string]any{key: "x"}
				if inConfig {
					cfg.ExtraBody = extra
				} else {
					req.ExtraBody = extra
				}

				for _, endpoint := range []string{"", EndpointResponses} {
					cfg.Endpoint = endpoint
					_, err := New(cfg).Complete(context.Background(), req)
					if err == nil || !strings.Contains(err.Error(), fmt.Sprintf("%q is reserved", key)) {
						t.Errorf("%s: Complete error = %v, want reserved-key error for %q", endpointName(endpoint), err, key)
					}
				}
				if n := len(rec.snapshot()); n != 0 {
					t.Errorf("server saw %d requests, want 0", n)
				}
			})
		}
	}
}

// TestOpenAIProvider_ExtraBody_SurvivesMaxTokensFallback asserts the
// max_tokens retry body, built after the extras are merged, still
// carries them.
func TestOpenAIProvider_ExtraBody_SurvivesMaxTokensFallback(t *testing.T) {
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
	t.Cleanup(srv.Close)

	routing := map[string]any{"only": []any{"groq"}}
	p := New(Config{BaseURL: srv.URL, TokenSource: auth.NewStatic("k"), Model: "m", MaxRetries: 1})
	events, err := p.Complete(context.Background(), Request{
		Messages:  []conversation.Message{userText("hi")},
		MaxTokens: 64,
		Stream:    true,
		ExtraBody: map[string]any{"provider": routing},
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	drain(t, events)

	bodies := rec.snapshot()
	if len(bodies) != 2 {
		t.Fatalf("server saw %d requests, want 2 (primary + fallback)", len(bodies))
	}
	for i, body := range bodies {
		if !reflect.DeepEqual(body["provider"], routing) {
			t.Errorf("request %d provider = %#v, want %#v", i, body["provider"], routing)
		}
	}
	if bodies[1]["max_tokens"] != float64(64) {
		t.Errorf("fallback max_tokens = %v, want 64", bodies[1]["max_tokens"])
	}
}
