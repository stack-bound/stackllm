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

func userText(s string) conversation.Message {
	return conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: s}},
	}
}

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
		HTTPClient: newTestClient(func(req *http.Request) (*http.Response, error) {
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
		Messages: []conversation.Message{userText("Hi")},
		Stream:   true,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var blocks []conversation.Block
	var tokens []string
	var done bool
	for ev := range events {
		switch ev.Type {
		case EventTypeBlockDelta:
			if ev.BlockType == conversation.BlockText {
				tokens = append(tokens, ev.Content)
			}
		case EventTypeBlockEnd:
			if ev.Block != nil {
				blocks = append(blocks, *ev.Block)
			}
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
		t.Errorf("streamed text = %q, want %q", got, "Hello world")
	}
	if len(blocks) != 1 || blocks[0].Type != conversation.BlockText || blocks[0].Text != "Hello world" {
		t.Errorf("closed blocks = %+v", blocks)
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
		Messages: []conversation.Message{userText("read /tmp/test")},
		Tools:    []tools.Definition{{Name: "read_file", Description: "read a file"}},
		Stream:   true,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var blocks []conversation.Block
	var toolCalls []ToolCall
	for ev := range events {
		switch ev.Type {
		case EventTypeBlockEnd:
			if ev.Block != nil {
				blocks = append(blocks, *ev.Block)
			}
		case EventTypeToolCall:
			toolCalls = append(toolCalls, *ev.Call)
		case EventTypeError:
			t.Fatalf("unexpected error: %v", ev.Err)
		}
	}

	if len(blocks) != 1 {
		t.Fatalf("got %d closed blocks, want 1", len(blocks))
	}
	if blocks[0].Type != conversation.BlockToolUse {
		t.Errorf("block type = %q, want tool_use", blocks[0].Type)
	}
	if blocks[0].ToolName != "read_file" || blocks[0].ToolCallID != "call_1" || blocks[0].ToolArgsJSON != `{"path":"/tmp/test"}` {
		t.Fatalf("tool_use block = %+v", blocks[0])
	}
	if len(toolCalls) != 1 {
		t.Fatalf("convenience ToolCall events = %d, want 1", len(toolCalls))
	}
	if toolCalls[0].Name != "read_file" || toolCalls[0].Arguments != `{"path":"/tmp/test"}` {
		t.Fatalf("ToolCall event = %+v", toolCalls[0])
	}
}

// TestOpenAIProvider_StreamMultipleToolCallsAfterText asserts that
// chat-completions streams with leading text followed by multiple
// tool_calls produce a BlockText block first, then BlockToolUse blocks
// in call-index order. This matches the plan's requirement that
// BlockToolUse blocks "appear after any preceding text, in call index
// order".
func TestOpenAIProvider_StreamMultipleToolCallsAfterText(t *testing.T) {
	t.Parallel()

	sseData := `data: {"choices":[{"delta":{"content":"let me try both"}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read","arguments":"{}"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c2","function":{"name":"grep","arguments":"{}"}}]}}]}

data: [DONE]

`
	p := New(Config{
		BaseURL:     "http://provider.test/v1",
		TokenSource: auth.NewStatic("k"),
		Model:       "gpt-4",
		HTTPClient:  newTestClient(func(req *http.Request) (*http.Response, error) { return textResponse(http.StatusOK, "text/event-stream", sseData), nil }),
		MaxRetries:  1,
	})

	events, err := p.Complete(context.Background(), Request{
		Messages: []conversation.Message{userText("go")},
		Tools:    []tools.Definition{{Name: "read"}, {Name: "grep"}},
		Stream:   true,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var blocks []conversation.Block
	for ev := range events {
		if ev.Type == EventTypeBlockEnd && ev.Block != nil {
			blocks = append(blocks, *ev.Block)
		}
	}

	if len(blocks) != 3 {
		t.Fatalf("blocks = %d, want 3 (text + 2 tool_use); got %+v", len(blocks), blocks)
	}
	if blocks[0].Type != conversation.BlockText || blocks[0].Text != "let me try both" {
		t.Errorf("blocks[0] = %+v", blocks[0])
	}
	if blocks[1].Type != conversation.BlockToolUse || blocks[1].ToolName != "read" || blocks[1].ToolCallID != "c1" {
		t.Errorf("blocks[1] = %+v", blocks[1])
	}
	if blocks[2].Type != conversation.BlockToolUse || blocks[2].ToolName != "grep" || blocks[2].ToolCallID != "c2" {
		t.Errorf("blocks[2] = %+v", blocks[2])
	}

	// Tool_use blocks must carry a non-empty Block.ID too — distinct
	// from ToolCallID, which is the provider's opaque call_id. The
	// Block.ID is the stackllm stable identifier used for persistence
	// and SSE keying.
	seen := make(map[string]bool, len(blocks))
	for i, b := range blocks {
		if b.ID == "" {
			t.Errorf("blocks[%d].ID is empty; provider must mint IDs at construction", i)
		}
		if seen[b.ID] {
			t.Errorf("blocks[%d].ID = %q collides with an earlier block", i, b.ID)
		}
		seen[b.ID] = true
	}
}

func TestOpenAIProvider_StreamInterleavedThinkingAndText(t *testing.T) {
	t.Parallel()

	// Simulates an OpenAI-compatible backend that exposes reasoning
	// under delta.reasoning_content interleaved with content deltas.
	// The parser must emit block events that preserve the ordering:
	//   thinking("plan") → text("answer") → thinking("recheck") → text("done")
	sseData := `data: {"choices":[{"delta":{"reasoning_content":"plan"}}]}

data: {"choices":[{"delta":{"content":"answer"}}]}

data: {"choices":[{"delta":{"reasoning_content":"recheck"}}]}

data: {"choices":[{"delta":{"content":"done"}}]}

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
		Messages: []conversation.Message{userText("hi")},
		Stream:   true,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var blocks []conversation.Block
	for ev := range events {
		if ev.Type == EventTypeBlockEnd && ev.Block != nil {
			blocks = append(blocks, *ev.Block)
		}
	}

	if len(blocks) != 4 {
		t.Fatalf("got %d blocks, want 4; blocks=%+v", len(blocks), blocks)
	}
	wantTypes := []conversation.BlockType{
		conversation.BlockThinking, conversation.BlockText,
		conversation.BlockThinking, conversation.BlockText,
	}
	wantTexts := []string{"plan", "answer", "recheck", "done"}
	for i, want := range wantTypes {
		if blocks[i].Type != want {
			t.Errorf("blocks[%d].Type = %q, want %q", i, blocks[i].Type, want)
		}
		if blocks[i].Text != wantTexts[i] {
			t.Errorf("blocks[%d].Text = %q, want %q", i, blocks[i].Text, wantTexts[i])
		}
	}

	// Every BlockEnd event must carry a non-empty, unique Block.ID —
	// downstream consumers (web SSE, agent hooks) key on this. The
	// provider mints IDs at construction time so the hook chain sees
	// them, not the agent's post-stream EnsureMessageIDs pass.
	seen := make(map[string]bool, len(blocks))
	for i, b := range blocks {
		if b.ID == "" {
			t.Errorf("blocks[%d].ID is empty; provider must mint IDs at construction", i)
		}
		if seen[b.ID] {
			t.Errorf("blocks[%d].ID = %q collides with an earlier block", i, b.ID)
		}
		seen[b.ID] = true
	}
}

// TestOpenAIProvider_StreamReasoningTextFieldName asserts that reasoning
// deltas delivered under `reasoning_text` (the field name used by
// GitHub Copilot's proxy for Gemini 3.x models) map to BlockThinking.
// Historically the parser only recognised `reasoning_content` /
// `reasoning`, so Copilot-proxied Gemini thinking was dropped silently.
func TestOpenAIProvider_StreamReasoningTextFieldName(t *testing.T) {
	t.Parallel()

	sseData := `data: {"choices":[{"delta":{"reasoning_text":"let me think"}}]}

data: {"choices":[{"delta":{"content":"the answer"}}]}

data: [DONE]

`
	p := New(Config{
		BaseURL:     "http://provider.test/v1",
		TokenSource: auth.NewStatic("key"),
		Model:       "gemini-3.1-pro-preview",
		HTTPClient:  newTestClient(func(req *http.Request) (*http.Response, error) { return textResponse(http.StatusOK, "text/event-stream", sseData), nil }),
		MaxRetries:  1,
	})

	events, err := p.Complete(context.Background(), Request{
		Messages: []conversation.Message{userText("hi")},
		Stream:   true,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var blocks []conversation.Block
	for ev := range events {
		if ev.Type == EventTypeBlockEnd && ev.Block != nil {
			blocks = append(blocks, *ev.Block)
		}
	}

	if len(blocks) != 2 {
		t.Fatalf("got %d blocks, want 2; blocks=%+v", len(blocks), blocks)
	}
	if blocks[0].Type != conversation.BlockThinking || blocks[0].Text != "let me think" {
		t.Errorf("blocks[0] = {%s %q}, want thinking/%q", blocks[0].Type, blocks[0].Text, "let me think")
	}
	if blocks[1].Type != conversation.BlockText || blocks[1].Text != "the answer" {
		t.Errorf("blocks[1] = {%s %q}, want text/%q", blocks[1].Type, blocks[1].Text, "the answer")
	}
}

func TestOpenAIProvider_BuildRequestBody_DropsThinkingAndFlattens(t *testing.T) {
	t.Parallel()

	p := New(Config{
		BaseURL:     "http://provider.test/v1",
		TokenSource: auth.NewStatic("k"),
		Model:       "gpt-4",
		HTTPClient:  newTestClient(func(req *http.Request) (*http.Response, error) { return textResponse(http.StatusOK, "text/event-stream", "data: [DONE]\n\n"), nil }),
		MaxRetries:  1,
	})

	// Assistant with interleaved thinking + text + tool_use. Chat
	// completions can't carry thinking or interleaving — verify the
	// flatten behaviour is exactly what the godoc promises.
	msgs := []conversation.Message{
		{Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hi"}}},
		{
			Role: conversation.RoleAssistant,
			Blocks: []conversation.Block{
				{Type: conversation.BlockThinking, Text: "thinking-a"},
				{Type: conversation.BlockText, Text: "one"},
				{Type: conversation.BlockToolUse, ToolCallID: "c1", ToolName: "f", ToolArgsJSON: `{"a":1}`},
				{Type: conversation.BlockThinking, Text: "thinking-b"},
				{Type: conversation.BlockText, Text: "two"},
			},
		},
		{
			Role: conversation.RoleTool,
			Blocks: []conversation.Block{
				{Type: conversation.BlockToolResult, ToolCallID: "c1", Text: "result-text"},
			},
		},
	}

	body := p.buildRequestBody(Request{Messages: msgs, Stream: true})
	wireMsgs := body["messages"].([]map[string]any)

	if len(wireMsgs) != 3 {
		t.Fatalf("got %d wire messages, want 3", len(wireMsgs))
	}

	// Assistant: text blocks concatenate; thinking dropped; tool_use hoisted.
	assistant := wireMsgs[1]
	if got, want := assistant["content"], "onetwo"; got != want {
		t.Errorf("assistant.content = %v, want %v", got, want)
	}
	tcs, ok := assistant["tool_calls"].([]map[string]any)
	if !ok || len(tcs) != 1 {
		t.Fatalf("assistant.tool_calls missing or wrong: %+v", assistant["tool_calls"])
	}
	if tcs[0]["id"] != "c1" {
		t.Errorf("tool_call id = %v", tcs[0]["id"])
	}

	// Tool message carries tool_call_id from the block, not from the message.
	tool := wireMsgs[2]
	if tool["role"] != "tool" {
		t.Errorf("tool role = %v", tool["role"])
	}
	if tool["tool_call_id"] != "c1" {
		t.Errorf("tool_call_id = %v", tool["tool_call_id"])
	}
	if tool["content"] != "result-text" {
		t.Errorf("tool content = %v", tool["content"])
	}
}

func TestOpenAIProvider_BuildRequestBody_MultiPartImage(t *testing.T) {
	t.Parallel()

	p := New(Config{
		BaseURL:     "http://provider.test/v1",
		TokenSource: auth.NewStatic("k"),
		Model:       "gpt-4",
		HTTPClient:  newTestClient(func(req *http.Request) (*http.Response, error) { return textResponse(http.StatusOK, "text/event-stream", "data: [DONE]\n\n"), nil }),
		MaxRetries:  1,
	})

	msgs := []conversation.Message{
		{
			Role: conversation.RoleUser,
			Blocks: []conversation.Block{
				{Type: conversation.BlockText, Text: "what is this?"},
				{Type: conversation.BlockImage, MimeType: "image/png", ImageData: []byte{0x89, 0x50}},
			},
		},
	}

	body := p.buildRequestBody(Request{Messages: msgs, Stream: true})
	wireMsgs := body["messages"].([]map[string]any)

	if len(wireMsgs) != 1 {
		t.Fatalf("wire messages = %d, want 1", len(wireMsgs))
	}
	content, ok := wireMsgs[0]["content"].([]map[string]any)
	if !ok {
		t.Fatalf("user content is not multi-part: %T %+v", wireMsgs[0]["content"], wireMsgs[0]["content"])
	}
	if len(content) != 2 {
		t.Fatalf("multi-part length = %d, want 2", len(content))
	}
	if content[0]["type"] != "text" || content[0]["text"] != "what is this?" {
		t.Errorf("part[0] = %+v", content[0])
	}
	if content[1]["type"] != "image_url" {
		t.Errorf("part[1].type = %v", content[1]["type"])
	}
	imgURL, _ := content[1]["image_url"].(map[string]any)
	if imgURL == nil {
		t.Fatalf("part[1].image_url missing")
	}
	if url, _ := imgURL["url"].(string); !strings.HasPrefix(url, "data:image/png;base64,") {
		t.Errorf("data uri = %q", url)
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
		Messages: []conversation.Message{userText("test")},
		Stream:   true,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var gotContent bool
	for ev := range events {
		if ev.Type == EventTypeBlockEnd && ev.Block != nil && ev.Block.Text == "ok" {
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
		Messages: []conversation.Message{userText("test")},
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

// TestOpenAIProvider_Models_ContextWindowFromCapabilities exercises
// the Copilot-shaped metadata path: capabilities.limits.max_prompt_tokens
// must land on ModelMeta.ContextWindow so profile.ModelInfo can
// propagate it to the TUI without a follow-up lookup.
func TestOpenAIProvider_Models_ContextWindowFromCapabilities(t *testing.T) {
	t.Parallel()

	body := `{"data":[
		{"id":"with-limits","capabilities":{"type":"chat","limits":{"max_prompt_tokens":123456}}},
		{"id":"without-limits","capabilities":{"type":"chat"}}
	]}`
	p := New(Config{
		BaseURL:     "http://provider.test/v1",
		TokenSource: auth.NewStatic("key"),
		HTTPClient: newTestClient(func(req *http.Request) (*http.Response, error) {
			return textResponse(http.StatusOK, "application/json", body), nil
		}),
	})

	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models error: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("got %d models, want 2", len(models))
	}
	byID := map[string]ModelMeta{}
	for _, m := range models {
		byID[m.ID] = m
	}
	if got := byID["with-limits"].ContextWindow; got != 123456 {
		t.Errorf("with-limits ContextWindow = %d, want 123456", got)
	}
	if got := byID["without-limits"].ContextWindow; got != 0 {
		t.Errorf("without-limits ContextWindow = %d, want 0", got)
	}
}

// TestOpenAIProvider_Complete_EmitsUsage verifies that a usage block
// in the terminal SSE chunk is forwarded as EventTypeUsage before
// EventTypeDone, using the numbers the upstream reported verbatim.
// This is the data path the TUI relies on to display actual token
// counts rather than a chars/4 estimate.
func TestOpenAIProvider_Complete_EmitsUsage(t *testing.T) {
	t.Parallel()

	sseData := `data: {"choices":[{"delta":{"content":"Hi"}}]}

data: {"choices":[],"usage":{"prompt_tokens":42,"completion_tokens":7,"total_tokens":49}}

data: [DONE]

`
	p := New(Config{
		BaseURL:     "http://provider.test/v1",
		TokenSource: auth.NewStatic("key"),
		Model:       "gpt-4o",
		HTTPClient: newTestClient(func(req *http.Request) (*http.Response, error) {
			return textResponse(http.StatusOK, "text/event-stream", sseData), nil
		}),
		MaxRetries: 1,
	})

	events, err := p.Complete(context.Background(), Request{
		Messages: []conversation.Message{userText("Hi")},
		Stream:   true,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var usage *TokenUsage
	var sawDoneAfterUsage bool
	var sawDone bool
	for ev := range events {
		switch ev.Type {
		case EventTypeUsage:
			if ev.Usage == nil {
				t.Fatal("EventTypeUsage had nil Usage")
			}
			u := *ev.Usage
			usage = &u
		case EventTypeDone:
			sawDone = true
			if usage != nil {
				sawDoneAfterUsage = true
			}
		case EventTypeError:
			t.Fatalf("unexpected error: %v", ev.Err)
		}
	}
	if !sawDone {
		t.Error("never saw EventTypeDone")
	}
	if usage == nil {
		t.Fatal("never saw EventTypeUsage")
	}
	if !sawDoneAfterUsage {
		t.Error("EventTypeUsage must be emitted before EventTypeDone")
	}
	if usage.PromptTokens != 42 || usage.CompletionTokens != 7 || usage.TotalTokens != 49 {
		t.Errorf("usage = %+v, want {42,7,49}", usage)
	}
}

// TestOpenAIProvider_BuildRequestBody_IncludeUsage asserts that a
// streamed request opts into stream_options.include_usage so OpenAI
// actually emits a usage block. Without this flag the end-of-stream
// is silent for prompt token accounting.
func TestOpenAIProvider_BuildRequestBody_IncludeUsage(t *testing.T) {
	t.Parallel()

	p := New(Config{Model: "gpt-4o", TokenSource: auth.NewStatic("k")})

	body := p.buildRequestBody(Request{
		Messages: []conversation.Message{userText("hi")},
		Stream:   true,
	})
	so, ok := body["stream_options"].(map[string]any)
	if !ok {
		t.Fatalf("stream_options not set or wrong type: %T", body["stream_options"])
	}
	if iu, _ := so["include_usage"].(bool); !iu {
		t.Errorf("stream_options.include_usage = %v, want true", so["include_usage"])
	}

	// Non-streamed requests should omit stream_options — some
	// backends reject it outright on non-stream calls.
	bodyNS := p.buildRequestBody(Request{
		Messages: []conversation.Message{userText("hi")},
		Stream:   false,
	})
	if _, present := bodyNS["stream_options"]; present {
		t.Errorf("stream_options should not be set when Stream=false, got %v", bodyNS["stream_options"])
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

