package provider

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stack-bound/stackllm/auth"
	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/tools"
)

func TestOpenAIProvider_ModelGetter(t *testing.T) {
	t.Parallel()
	p := New(Config{Model: "the-model", TokenSource: auth.NewStatic("k")})
	if got := p.Model(); got != "the-model" {
		t.Errorf("Model() = %q, want the-model", got)
	}
}

func TestComplete_MarshalErrorFromToolParameters(t *testing.T) {
	t.Parallel()

	// A channel inside tool parameters is not JSON-marshalable, so the
	// request body marshal must fail before any HTTP call.
	badTools := []tools.Definition{{
		Name:       "bad",
		Parameters: map[string]any{"ch": make(chan int)},
	}}

	chat := New(Config{BaseURL: "http://unused.test", TokenSource: auth.NewStatic("k"), Model: "m"})
	if _, err := chat.Complete(context.Background(), Request{Tools: badTools}); err == nil || !strings.Contains(err.Error(), "marshal request") {
		t.Errorf("chat error = %v, want marshal request error", err)
	}

	resp := New(Config{BaseURL: "http://unused.test", TokenSource: auth.NewStatic("k"), Model: "m", Endpoint: EndpointResponses})
	if _, err := resp.Complete(context.Background(), Request{Tools: badTools}); err == nil || !strings.Contains(err.Error(), "marshal responses request") {
		t.Errorf("responses error = %v, want marshal responses request error", err)
	}
}

func TestCompleteResponses_InputConversionError(t *testing.T) {
	t.Parallel()

	// A tool-role message without tool_result blocks cannot be converted
	// for the Responses API — Complete must surface the error eagerly.
	p := New(Config{BaseURL: "http://unused.test", TokenSource: auth.NewStatic("k"), Model: "m", Endpoint: EndpointResponses})
	msgs := []conversation.Message{{Role: conversation.RoleTool}}
	if _, err := p.Complete(context.Background(), Request{Messages: msgs}); err == nil || !strings.Contains(err.Error(), "no tool_result blocks") {
		t.Errorf("error = %v, want tool_result conversion error", err)
	}
}

func TestAPIVersion_AppendedToAllEndpoints(t *testing.T) {
	t.Parallel()

	sse := "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n"
	respSSE := "event: response.completed\ndata: {\"response\":{}}\n\n"

	tests := []struct {
		name     string
		endpoint string
		wantPath string
		body     string
		invoke   func(t *testing.T, p *OpenAIProvider)
	}{
		{
			name:     "chat completions",
			wantPath: "/chat/completions",
			body:     sse,
			invoke: func(t *testing.T, p *OpenAIProvider) {
				events, err := p.Complete(context.Background(), Request{Messages: []conversation.Message{userText("hi")}, Stream: true})
				if err != nil {
					t.Fatalf("Complete: %v", err)
				}
				for range events {
				}
			},
		},
		{
			name:     "responses",
			endpoint: EndpointResponses,
			wantPath: "/responses",
			body:     respSSE,
			invoke: func(t *testing.T, p *OpenAIProvider) {
				events, err := p.Complete(context.Background(), Request{Messages: []conversation.Message{userText("hi")}, Stream: true})
				if err != nil {
					t.Fatalf("Complete: %v", err)
				}
				for range events {
				}
			},
		},
		{
			name:     "models",
			wantPath: "/models",
			body:     `{"data":[{"id":"m1"}]}`,
			invoke: func(t *testing.T, p *OpenAIProvider) {
				if _, err := p.Models(context.Background()); err != nil {
					t.Fatalf("Models: %v", err)
				}
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			var gotVersion atomic.Value
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == tt.wantPath {
					gotVersion.Store(r.URL.Query().Get("api-version"))
				}
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = w.Write([]byte(tt.body))
			}))
			t.Cleanup(srv.Close)

			p := New(Config{
				BaseURL:     srv.URL,
				TokenSource: auth.NewStatic("k"),
				Model:       "m",
				APIVersion:  "2024-02-01",
				Endpoint:    tt.endpoint,
				MaxRetries:  1,
			})
			tt.invoke(t, p)

			if got, _ := gotVersion.Load().(string); got != "2024-02-01" {
				t.Errorf("api-version query = %q, want 2024-02-01", got)
			}
		})
	}
}

func TestModels_ErrorPaths(t *testing.T) {
	t.Parallel()

	t.Run("invalid base URL", func(t *testing.T) {
		t.Parallel()
		p := New(Config{BaseURL: "http://bad url\x7f", TokenSource: auth.NewStatic("k")})
		if _, err := p.Models(context.Background()); err == nil || !strings.Contains(err.Error(), "models request") {
			t.Errorf("error = %v, want models request error", err)
		}
	})

	t.Run("connection failure", func(t *testing.T) {
		t.Parallel()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
		srv.Close()
		p := New(Config{BaseURL: srv.URL, TokenSource: auth.NewStatic("k")})
		if _, err := p.Models(context.Background()); err == nil || !strings.Contains(err.Error(), "models:") {
			t.Errorf("error = %v, want transport error", err)
		}
	})

	t.Run("invalid JSON body", func(t *testing.T) {
		t.Parallel()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte("not json"))
		}))
		t.Cleanup(srv.Close)
		p := New(Config{BaseURL: srv.URL, TokenSource: auth.NewStatic("k")})
		if _, err := p.Models(context.Background()); err == nil || !strings.Contains(err.Error(), "decode") {
			t.Errorf("error = %v, want decode error", err)
		}
	})
}

// --- chat wire-format edge cases ---

func TestMessageToChatCompletions_EdgeCases(t *testing.T) {
	t.Parallel()

	t.Run("system message", func(t *testing.T) {
		t.Parallel()
		out := messageToChatCompletions(conversation.Message{
			Role:   conversation.RoleSystem,
			Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "be brief"}},
		})
		if len(out) != 1 || out[0]["role"] != "system" || out[0]["content"] != "be brief" {
			t.Errorf("system message = %+v", out)
		}
	})

	t.Run("tool role without tool_result blocks falls back to text", func(t *testing.T) {
		t.Parallel()
		out := messageToChatCompletions(conversation.Message{
			Role:   conversation.RoleTool,
			Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "raw output"}},
		})
		if len(out) != 1 || out[0]["role"] != "tool" || out[0]["content"] != "raw output" {
			t.Errorf("fallback tool message = %+v", out)
		}
	})

	t.Run("unknown role passes through", func(t *testing.T) {
		t.Parallel()
		out := messageToChatCompletions(conversation.Message{
			Role:   conversation.Role("critic"),
			Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hm"}},
		})
		if len(out) != 1 || out[0]["role"] != "critic" || out[0]["content"] != "hm" {
			t.Errorf("unknown-role message = %+v", out)
		}
	})

	t.Run("assistant tool_use with empty args defaults to {}", func(t *testing.T) {
		t.Parallel()
		out := messageToChatCompletions(conversation.Message{
			Role: conversation.RoleAssistant,
			Blocks: []conversation.Block{{
				Type:       conversation.BlockToolUse,
				ToolCallID: "c1",
				ToolName:   "ping",
			}},
		})
		if len(out) != 1 {
			t.Fatalf("got %d messages", len(out))
		}
		calls, ok := out[0]["tool_calls"].([]map[string]any)
		if !ok || len(calls) != 1 {
			t.Fatalf("tool_calls = %+v", out[0]["tool_calls"])
		}
		fn, ok := calls[0]["function"].(map[string]any)
		if !ok || fn["arguments"] != "{}" {
			t.Errorf("arguments = %v, want {}", fn["arguments"])
		}
	})
}

func TestUserContentForChat_InlineImageDefaultsMime(t *testing.T) {
	t.Parallel()

	content := userContentForChat(conversation.Message{
		Role: conversation.RoleUser,
		Blocks: []conversation.Block{
			{Type: conversation.BlockText, Text: "look"},
			{Type: conversation.BlockImage, ImageData: []byte{0x89, 0x50}},
		},
	})
	parts, ok := content.([]map[string]any)
	if !ok || len(parts) != 2 {
		t.Fatalf("content = %+v, want two parts", content)
	}
	img, ok := parts[1]["image_url"].(map[string]any)
	if !ok {
		t.Fatalf("image part = %+v", parts[1])
	}
	url, _ := img["url"].(string)
	if !strings.HasPrefix(url, "data:image/png;base64,") {
		t.Errorf("image url = %q, want data:image/png prefix when mime is empty", url)
	}
}

// --- retry / transport branches ---

func TestDoStreamingPOST_RetriesThenSucceeds(t *testing.T) {
	t.Parallel()

	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			http.Error(w, "overloaded", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"recovered\"}}]}\n\ndata: [DONE]\n\n"))
	}))
	t.Cleanup(srv.Close)

	p := New(Config{
		BaseURL:     srv.URL,
		TokenSource: auth.NewStatic("k"),
		Model:       "m",
		MaxRetries:  3,
		BaseBackoff: time.Millisecond,
	})
	events, err := p.Complete(context.Background(), Request{Messages: []conversation.Message{userText("hi")}, Stream: true})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var text strings.Builder
	for ev := range events {
		if ev.Type == EventTypeError {
			t.Fatalf("unexpected error after retry: %v", ev.Err)
		}
		if ev.Type == EventTypeBlockDelta && ev.BlockType == conversation.BlockText {
			text.WriteString(ev.Content)
		}
	}
	if text.String() != "recovered" {
		t.Errorf("text = %q, want recovered", text.String())
	}
	if calls.Load() != 2 {
		t.Errorf("backend calls = %d, want 2 (one failure + one retry)", calls.Load())
	}
}

func TestDoStreamingPOST_MaxRetriesExceeded(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	srv.Close() // every attempt fails at the transport level

	p := New(Config{
		BaseURL:     srv.URL,
		TokenSource: auth.NewStatic("k"),
		Model:       "m",
		MaxRetries:  2,
		BaseBackoff: time.Millisecond,
	})
	events, err := p.Complete(context.Background(), Request{Messages: []conversation.Message{userText("hi")}, Stream: true})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var gotErr error
	for ev := range events {
		if ev.Type == EventTypeError {
			gotErr = ev.Err
		}
	}
	if gotErr == nil || !strings.Contains(gotErr.Error(), "max retries exceeded") {
		t.Errorf("error = %v, want max retries exceeded", gotErr)
	}
}

func TestDoStreamingPOST_InvalidURL(t *testing.T) {
	t.Parallel()

	p := New(Config{
		BaseURL:     "http://bad url\x7f",
		TokenSource: auth.NewStatic("k"),
		Model:       "m",
		MaxRetries:  1,
	})
	events, err := p.Complete(context.Background(), Request{Messages: []conversation.Message{userText("hi")}, Stream: true})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var gotErr error
	for ev := range events {
		if ev.Type == EventTypeError {
			gotErr = ev.Err
		}
	}
	if gotErr == nil || !strings.Contains(gotErr.Error(), "create request") {
		t.Errorf("error = %v, want create request error", gotErr)
	}
}

// --- chat SSE parsing branches ---

func TestReadChatSSE_ConsecutiveReasoningDeltas(t *testing.T) {
	t.Parallel()

	stream := "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think \"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"harder\"}}]}\n\n" +
		"data: [DONE]\n\n"

	p := &OpenAIProvider{}
	events := make(chan Event, 32)
	go func() {
		defer close(events)
		p.readChatSSE(strings.NewReader(stream), events)
	}()

	var thinking []conversation.Block
	starts := 0
	for ev := range events {
		if ev.Type == EventTypeBlockStart && ev.BlockType == conversation.BlockThinking {
			starts++
		}
		if ev.Type == EventTypeBlockEnd && ev.Block != nil && ev.Block.Type == conversation.BlockThinking {
			thinking = append(thinking, *ev.Block)
		}
	}
	if starts != 1 {
		t.Errorf("thinking BlockStart count = %d, want 1 (consecutive deltas share a block)", starts)
	}
	if len(thinking) != 1 || thinking[0].Text != "think harder" {
		t.Errorf("thinking blocks = %+v, want single 'think harder'", thinking)
	}
}

func TestReadChatSSE_MalformedChunkSkipped(t *testing.T) {
	t.Parallel()

	stream := "data: {not json}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"still fine\"}}]}\n\n" +
		"data: [DONE]\n\n"

	p := &OpenAIProvider{}
	events := make(chan Event, 32)
	go func() {
		defer close(events)
		p.readChatSSE(strings.NewReader(stream), events)
	}()

	var text strings.Builder
	var done bool
	for ev := range events {
		if ev.Type == EventTypeBlockDelta && ev.BlockType == conversation.BlockText {
			text.WriteString(ev.Content)
		}
		if ev.Type == EventTypeDone {
			done = true
		}
	}
	if text.String() != "still fine" {
		t.Errorf("text = %q, want 'still fine' (malformed chunk skipped)", text.String())
	}
	if !done {
		t.Error("expected done event after malformed chunk was skipped")
	}
}

func TestReadChatSSE_TruncatedStreamClosesOpenBlock(t *testing.T) {
	t.Parallel()

	// The stream ends without [DONE]: the open text block must still be
	// closed so no partial content is lost.
	stream := "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n"

	p := &OpenAIProvider{}
	events := make(chan Event, 32)
	go func() {
		defer close(events)
		p.readChatSSE(strings.NewReader(stream), events)
	}()

	var blocks []conversation.Block
	var done bool
	for ev := range events {
		if ev.Type == EventTypeBlockEnd && ev.Block != nil {
			blocks = append(blocks, *ev.Block)
		}
		if ev.Type == EventTypeDone {
			done = true
		}
	}
	if len(blocks) != 1 || blocks[0].Text != "partial" {
		t.Errorf("blocks = %+v, want single 'partial' text block", blocks)
	}
	if done {
		t.Error("no [DONE] sentinel: EventTypeDone must not be emitted")
	}
}

// --- responses wire-format edge cases ---

func TestBuildResponsesBody_MaxTokensAndTemperature(t *testing.T) {
	t.Parallel()

	temp := 0.3
	p := New(Config{Model: "m", TokenSource: auth.NewStatic("k")})
	body, err := p.buildResponsesBody(Request{
		Messages:    []conversation.Message{userText("hi")},
		MaxTokens:   77,
		Temperature: &temp,
	})
	if err != nil {
		t.Fatalf("buildResponsesBody: %v", err)
	}
	if body["max_output_tokens"] != 77 {
		t.Errorf("max_output_tokens = %v, want 77", body["max_output_tokens"])
	}
	if body["temperature"] != 0.3 {
		t.Errorf("temperature = %v, want 0.3", body["temperature"])
	}
}

func TestMessagesToInput_EdgeCases(t *testing.T) {
	t.Parallel()

	t.Run("inline image defaults mime", func(t *testing.T) {
		t.Parallel()
		out, err := messagesToInput([]conversation.Message{{
			Role:   conversation.RoleUser,
			Blocks: []conversation.Block{{Type: conversation.BlockImage, ImageData: []byte{1, 2, 3}}},
		}})
		if err != nil {
			t.Fatalf("messagesToInput: %v", err)
		}
		parts, ok := out[0]["content"].([]map[string]any)
		if !ok || len(parts) != 1 {
			t.Fatalf("content = %+v", out[0]["content"])
		}
		url, _ := parts[0]["image_url"].(string)
		if !strings.HasPrefix(url, "data:image/png;base64,") {
			t.Errorf("image_url = %q, want data:image/png prefix", url)
		}
	})

	t.Run("user message with no usable blocks emits empty text part", func(t *testing.T) {
		t.Parallel()
		out, err := messagesToInput([]conversation.Message{{Role: conversation.RoleUser}})
		if err != nil {
			t.Fatalf("messagesToInput: %v", err)
		}
		parts, ok := out[0]["content"].([]map[string]any)
		if !ok || len(parts) != 1 {
			t.Fatalf("content = %+v", out[0]["content"])
		}
		if parts[0]["type"] != "input_text" || parts[0]["text"] != "" {
			t.Errorf("placeholder part = %+v, want empty input_text", parts[0])
		}
	})

	t.Run("assistant tool_use empty args defaults to {}", func(t *testing.T) {
		t.Parallel()
		out, err := messagesToInput([]conversation.Message{{
			Role:   conversation.RoleAssistant,
			Blocks: []conversation.Block{{Type: conversation.BlockToolUse, ToolCallID: "c1", ToolName: "ping"}},
		}})
		if err != nil {
			t.Fatalf("messagesToInput: %v", err)
		}
		if out[0]["arguments"] != "{}" {
			t.Errorf("arguments = %v, want {}", out[0]["arguments"])
		}
	})

	t.Run("unknown role errors", func(t *testing.T) {
		t.Parallel()
		_, err := messagesToInput([]conversation.Message{{Role: conversation.Role("critic")}})
		if err == nil || !strings.Contains(err.Error(), "unknown role") {
			t.Errorf("error = %v, want unknown role error", err)
		}
	})
}

// --- responses SSE parsing branches ---

// collectResponsesEvents runs readResponsesSSE over the stream and
// returns all events.
func collectResponsesEvents(t *testing.T, stream string) []Event {
	t.Helper()
	p := &OpenAIProvider{}
	ch := make(chan Event, 64)
	go func() {
		defer close(ch)
		p.readResponsesSSE(strings.NewReader(stream), ch)
	}()
	var out []Event
	for ev := range ch {
		out = append(out, ev)
	}
	return out
}

func TestReadResponsesSSE_IgnoresGarbageAndEmptyDeltas(t *testing.T) {
	t.Parallel()

	stream := ": keepalive comment\n" +
		"event: response.output_item.added\n" +
		"data: {not json}\n\n" +
		"event: response.output_text.delta\n" +
		"data: {not json}\n\n" +
		"event: response.output_text.delta\n" +
		`data: {"output_index":0,"delta":""}` + "\n\n" +
		"event: response.reasoning_summary_text.delta\n" +
		"data: {not json}\n\n" +
		"event: response.reasoning_summary_text.delta\n" +
		`data: {"output_index":1,"delta":""}` + "\n\n" +
		"event: response.function_call_arguments.delta\n" +
		"data: {not json}\n\n" +
		"event: response.function_call_arguments.delta\n" +
		`data: {"output_index":2,"delta":""}` + "\n\n" +
		"event: response.output_item.done\n" +
		"data: {not json}\n\n" +
		"event: response.output_text.delta\n" +
		`data: {"output_index":0,"delta":"real"}` + "\n\n" +
		"event: response.completed\n" +
		`data: {"response":{}}` + "\n\n"

	events := collectResponsesEvents(t, stream)

	var deltas []string
	var done bool
	for _, ev := range events {
		switch ev.Type {
		case EventTypeBlockDelta:
			deltas = append(deltas, ev.Content)
		case EventTypeDone:
			done = true
		case EventTypeError:
			t.Fatalf("unexpected error: %v", ev.Err)
		}
	}
	if len(deltas) != 1 || deltas[0] != "real" {
		t.Errorf("deltas = %q, want only [real] (garbage and empty deltas skipped)", deltas)
	}
	if !done {
		t.Error("expected done event")
	}
}

func TestReadResponsesSSE_FailedEvent(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		stream  string
		wantMsg string
	}{
		{
			name: "response-level error message",
			stream: "event: response.failed\n" +
				`data: {"response":{"error":{"message":"quota exhausted","code":"quota"}}}` + "\n\n",
			wantMsg: "quota exhausted",
		},
		{
			name: "top-level error message",
			stream: "event: response.error\n" +
				`data: {"error":{"message":"bad request","code":"invalid"}}` + "\n\n",
			wantMsg: "bad request",
		},
		{
			name: "no message falls back to generic",
			stream: "event: response.failed\n" +
				`data: {}` + "\n\n",
			wantMsg: "responses stream error",
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			events := collectResponsesEvents(t, tt.stream)

			var gotErr error
			var done bool
			for _, ev := range events {
				if ev.Type == EventTypeError {
					gotErr = ev.Err
				}
				if ev.Type == EventTypeDone {
					done = true
				}
			}
			if gotErr == nil || !strings.Contains(gotErr.Error(), tt.wantMsg) {
				t.Errorf("error = %v, want containing %q", gotErr, tt.wantMsg)
			}
			if done {
				t.Error("failed stream must not emit EventTypeDone")
			}
		})
	}
}
