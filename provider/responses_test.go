package provider

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/stack-bound/stackllm/auth"
	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/tools"
)

// SSE captured from a real Copilot /responses call (text-only). Reasoning
// items, content_part lifecycle, and the response.completed terminator
// are all real-world events that the parser must tolerate.
const responsesSSEText = `event: response.created
data: {"type":"response.created","sequence_number":0}

event: response.in_progress
data: {"type":"response.in_progress","sequence_number":1}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"sequence_number":2,"item":{"type":"reasoning","id":"r1","summary":[]}}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":0,"sequence_number":3,"item":{"type":"reasoning","id":"r1","summary":[{"type":"summary_text","text":"plan"}]}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":1,"sequence_number":4,"item":{"type":"message","role":"assistant","status":"in_progress","content":[]}}

event: response.content_part.added
data: {"type":"response.content_part.added","output_index":1,"content_index":0,"sequence_number":5,"part":{"type":"output_text","text":""}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","item_id":"m1","output_index":1,"content_index":0,"sequence_number":6,"delta":"Hello"}

event: response.output_text.delta
data: {"type":"response.output_text.delta","item_id":"m1","output_index":1,"content_index":0,"sequence_number":7,"delta":" world"}

event: response.output_text.done
data: {"type":"response.output_text.done","item_id":"m1","output_index":1,"content_index":0,"sequence_number":8,"text":"Hello world"}

event: response.content_part.done
data: {"type":"response.content_part.done","output_index":1,"content_index":0,"sequence_number":9,"part":{"type":"output_text","text":"Hello world"}}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":1,"sequence_number":10,"item":{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hello world"}]}}

event: response.completed
data: {"type":"response.completed","sequence_number":11,"response":{"status":"completed"}}

`

// SSE captured from a real Copilot /responses call (tool call). The
// arguments stream as deltas and are finalised on output_item.done with
// item.type == "function_call".
const responsesSSEToolCall = `event: response.created
data: {"type":"response.created","sequence_number":0}

event: response.in_progress
data: {"type":"response.in_progress","sequence_number":1}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"sequence_number":2,"item":{"type":"reasoning","id":"r1","summary":[]}}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":0,"sequence_number":3,"item":{"type":"reasoning","id":"r1","summary":[]}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":1,"sequence_number":4,"item":{"type":"function_call","id":"fc_internal","call_id":"call_abc","name":"current_time","arguments":"","status":"in_progress"}}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","item_id":"fc_internal","output_index":1,"sequence_number":5,"delta":"{}"}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","item_id":"fc_internal","output_index":1,"sequence_number":6,"arguments":"{}"}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":1,"sequence_number":7,"item":{"type":"function_call","id":"fc_internal","call_id":"call_abc","name":"current_time","arguments":"{}","status":"completed"}}

event: response.completed
data: {"type":"response.completed","sequence_number":8,"response":{"status":"completed"}}

`

func TestOpenAIProvider_CompleteResponses_TextStreaming(t *testing.T) {
	t.Parallel()

	var capturedBody []byte
	p := New(Config{
		BaseURL:     "http://provider.test",
		TokenSource: auth.NewStatic("k"),
		Model:       "gpt-5.4-mini",
		Endpoint:    EndpointResponses,
		HTTPClient: newTestClient(func(req *http.Request) (*http.Response, error) {
			if req.URL.Path != "/responses" {
				t.Errorf("path = %q, want /responses", req.URL.Path)
			}
			capturedBody, _ = io.ReadAll(req.Body)
			return textResponse(http.StatusOK, "text/event-stream", responsesSSEText), nil
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
	var done bool
	for ev := range events {
		switch ev.Type {
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

	// Two closed blocks in production order: the reasoning item first
	// (it closed before the message item), then the assistant message.
	if len(blocks) != 2 {
		t.Fatalf("closed blocks = %d, want 2; got %+v", len(blocks), blocks)
	}
	if blocks[0].Type != conversation.BlockThinking {
		t.Errorf("blocks[0].Type = %q, want thinking", blocks[0].Type)
	}
	if blocks[0].Text != "plan" {
		t.Errorf("thinking text = %q, want plan", blocks[0].Text)
	}
	if blocks[1].Type != conversation.BlockText {
		t.Errorf("blocks[1].Type = %q, want text", blocks[1].Type)
	}
	if blocks[1].Text != "Hello world" {
		t.Errorf("text = %q, want %q", blocks[1].Text, "Hello world")
	}

	// Verify the request body uses the Responses API shape, not chat.
	var body map[string]any
	if err := json.Unmarshal(capturedBody, &body); err != nil {
		t.Fatalf("body unmarshal: %v", err)
	}
	if _, ok := body["messages"]; ok {
		t.Error("body should not contain 'messages' field on /responses")
	}
	if _, ok := body["input"]; !ok {
		t.Error("body must contain 'input' field on /responses")
	}
	if got, want := body["model"], "gpt-5.4-mini"; got != want {
		t.Errorf("model = %v, want %v", got, want)
	}
}

func TestOpenAIProvider_CompleteResponses_ToolCall(t *testing.T) {
	t.Parallel()

	p := New(Config{
		BaseURL:     "http://provider.test",
		TokenSource: auth.NewStatic("k"),
		Model:       "gpt-5.4-mini",
		Endpoint:    EndpointResponses,
		HTTPClient: newTestClient(func(req *http.Request) (*http.Response, error) {
			return textResponse(http.StatusOK, "text/event-stream", responsesSSEToolCall), nil
		}),
		MaxRetries: 1,
	})

	events, err := p.Complete(context.Background(), Request{
		Messages: []conversation.Message{userText("what time")},
		Tools:    []tools.Definition{{Name: "current_time", Description: "Get the current time"}},
		Stream:   true,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var blocks []conversation.Block
	var calls []ToolCall
	var done bool
	for ev := range events {
		switch ev.Type {
		case EventTypeBlockEnd:
			if ev.Block != nil {
				blocks = append(blocks, *ev.Block)
			}
		case EventTypeToolCall:
			calls = append(calls, *ev.Call)
		case EventTypeDone:
			done = true
		case EventTypeError:
			t.Fatalf("unexpected error: %v", ev.Err)
		}
	}

	if !done {
		t.Error("expected done event")
	}

	// Should have closed two blocks: the (empty) reasoning item and the
	// function_call. Verify the tool_use block.
	var toolBlock *conversation.Block
	for i := range blocks {
		if blocks[i].Type == conversation.BlockToolUse {
			toolBlock = &blocks[i]
			break
		}
	}
	if toolBlock == nil {
		t.Fatalf("no tool_use block; got %+v", blocks)
	}
	if toolBlock.ToolCallID != "call_abc" || toolBlock.ToolName != "current_time" || toolBlock.ToolArgsJSON != "{}" {
		t.Errorf("tool_use block = %+v", toolBlock)
	}
	if len(calls) != 1 {
		t.Fatalf("convenience ToolCall events = %d, want 1", len(calls))
	}
	if calls[0].ID != "call_abc" || calls[0].Name != "current_time" || calls[0].Arguments != "{}" {
		t.Errorf("tool call = %+v", calls[0])
	}
}

// TestOpenAIProvider_CompleteResponses_InterleavedBlocks is the
// Phase 1 regression-guard for responses.go:173-174: reasoning items
// must no longer be dropped; they must become BlockThinking blocks
// interleaved with text blocks in the order the items close.
func TestOpenAIProvider_CompleteResponses_InterleavedBlocks(t *testing.T) {
	t.Parallel()

	stream := "event: response.output_item.added\n" +
		`data: {"output_index":0,"item":{"type":"reasoning","summary":[]}}` + "\n\n" +
		"event: response.reasoning_summary_text.delta\n" +
		`data: {"output_index":0,"delta":"plan"}` + "\n\n" +
		"event: response.output_item.done\n" +
		`data: {"output_index":0,"item":{"type":"reasoning","summary":[{"type":"summary_text","text":"plan"}]}}` + "\n\n" +
		"event: response.output_item.added\n" +
		`data: {"output_index":1,"item":{"type":"message","role":"assistant","content":[]}}` + "\n\n" +
		"event: response.output_text.delta\n" +
		`data: {"output_index":1,"delta":"answer"}` + "\n\n" +
		"event: response.output_item.done\n" +
		`data: {"output_index":1,"item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}}` + "\n\n" +
		"event: response.output_item.added\n" +
		`data: {"output_index":2,"item":{"type":"reasoning","summary":[]}}` + "\n\n" +
		"event: response.reasoning_summary_text.delta\n" +
		`data: {"output_index":2,"delta":"recheck"}` + "\n\n" +
		"event: response.output_item.done\n" +
		`data: {"output_index":2,"item":{"type":"reasoning","summary":[{"type":"summary_text","text":"recheck"}]}}` + "\n\n" +
		"event: response.output_item.added\n" +
		`data: {"output_index":3,"item":{"type":"function_call","call_id":"c1","name":"do","arguments":""}}` + "\n\n" +
		"event: response.function_call_arguments.delta\n" +
		`data: {"output_index":3,"delta":"{\"x\":1}"}` + "\n\n" +
		"event: response.output_item.done\n" +
		`data: {"output_index":3,"item":{"type":"function_call","call_id":"c1","name":"do","arguments":"{\"x\":1}"}}` + "\n\n" +
		"event: response.completed\n" +
		`data: {"response":{"status":"completed"}}` + "\n\n"

	p := &OpenAIProvider{}
	events := make(chan Event, 64)
	go func() {
		defer close(events)
		p.readResponsesSSE(strings.NewReader(stream), events)
	}()

	var blocks []conversation.Block
	for ev := range events {
		if ev.Type == EventTypeBlockEnd && ev.Block != nil {
			blocks = append(blocks, *ev.Block)
		}
	}

	wantTypes := []conversation.BlockType{
		conversation.BlockThinking, conversation.BlockText,
		conversation.BlockThinking, conversation.BlockToolUse,
	}
	if len(blocks) != len(wantTypes) {
		t.Fatalf("blocks = %d, want %d; got %+v", len(blocks), len(wantTypes), blocks)
	}
	for i, want := range wantTypes {
		if blocks[i].Type != want {
			t.Errorf("blocks[%d].Type = %q, want %q", i, blocks[i].Type, want)
		}
	}
	if blocks[0].Text != "plan" {
		t.Errorf("thinking[0] text = %q", blocks[0].Text)
	}
	if blocks[1].Text != "answer" {
		t.Errorf("text text = %q", blocks[1].Text)
	}
	if blocks[2].Text != "recheck" {
		t.Errorf("thinking[1] text = %q", blocks[2].Text)
	}
	if blocks[3].ToolCallID != "c1" || blocks[3].ToolName != "do" || blocks[3].ToolArgsJSON != `{"x":1}` {
		t.Errorf("tool_use = %+v", blocks[3])
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

func TestOpenAIProvider_CompleteResponses_ToolFormat(t *testing.T) {
	t.Parallel()

	var capturedBody []byte
	p := New(Config{
		BaseURL:     "http://provider.test",
		TokenSource: auth.NewStatic("k"),
		Model:       "gpt-5.4-mini",
		Endpoint:    EndpointResponses,
		HTTPClient: newTestClient(func(req *http.Request) (*http.Response, error) {
			capturedBody, _ = io.ReadAll(req.Body)
			return textResponse(http.StatusOK, "text/event-stream", responsesSSEText), nil
		}),
		MaxRetries: 1,
	})

	events, err := p.Complete(context.Background(), Request{
		Messages: []conversation.Message{userText("Hi")},
		Tools: []tools.Definition{{
			Name:        "read_file",
			Description: "Read a file",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		}},
		Stream: true,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	// Drain events so the request goroutine completes before we
	// inspect capturedBody.
	for range events {
	}

	var body struct {
		Tools []map[string]any `json:"tools"`
	}
	if err := json.Unmarshal(capturedBody, &body); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(body.Tools) != 1 {
		t.Fatalf("got %d tools, want 1", len(body.Tools))
	}
	tool := body.Tools[0]

	// Responses API uses a flat shape: {type,name,description,parameters}
	// — NOT chat completions' {type,function:{name,description,parameters}}.
	if _, ok := tool["function"]; ok {
		t.Error("tool should not have a 'function' wrapper on /responses")
	}
	if got, want := tool["type"], "function"; got != want {
		t.Errorf("tool.type = %v, want %v", got, want)
	}
	if got, want := tool["name"], "read_file"; got != want {
		t.Errorf("tool.name = %v, want %v", got, want)
	}
	if got, want := tool["description"], "Read a file"; got != want {
		t.Errorf("tool.description = %v, want %v", got, want)
	}
	if _, ok := tool["parameters"]; !ok {
		t.Error("tool.parameters missing")
	}
}

// TestOpenAIProvider_CompleteResponses_DisableStore verifies that
// Config.DisableStore forces body["store"] = false on /responses
// calls, and omits the field otherwise. The ChatGPT Codex endpoint
// rejects requests that default to store=true with "Store must be
// set to false".
func TestOpenAIProvider_CompleteResponses_DisableStore(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		disable    bool
		wantStore  bool // whether the field should appear in the body
		wantValue  bool // expected value when present
	}{
		{"disabled", true, true, false},
		{"default omits store", false, false, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var capturedBody []byte
			p := New(Config{
				BaseURL:      "http://provider.test",
				TokenSource:  auth.NewStatic("k"),
				Model:        "gpt-5.4-mini",
				Endpoint:     EndpointResponses,
				DisableStore: tt.disable,
				HTTPClient: newTestClient(func(req *http.Request) (*http.Response, error) {
					capturedBody, _ = io.ReadAll(req.Body)
					return textResponse(http.StatusOK, "text/event-stream", responsesSSEText), nil
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
			for range events {
			}

			var body map[string]any
			if err := json.Unmarshal(capturedBody, &body); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			got, present := body["store"]
			if tt.wantStore {
				if !present {
					t.Fatal("body missing store field")
				}
				if got != tt.wantValue {
					t.Errorf("store = %v, want %v", got, tt.wantValue)
				}
			} else if present {
				t.Errorf("store field present unexpectedly: %v", got)
			}
		})
	}
}

// TestOpenAIProvider_CompleteResponses_Instructions verifies that a
// non-empty Config.Instructions is serialised into the top-level
// body["instructions"] on /responses calls. This is required by the
// ChatGPT Codex endpoint (chatgpt.com/backend-api/codex/responses)
// which rejects requests without it.
func TestOpenAIProvider_CompleteResponses_Instructions(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		instructions string
		wantPresent  bool
	}{
		{"set", "You are a helpful assistant.", true},
		{"empty omitted", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var capturedBody []byte
			p := New(Config{
				BaseURL:      "http://provider.test",
				TokenSource:  auth.NewStatic("k"),
				Model:        "gpt-5.4-mini",
				Endpoint:     EndpointResponses,
				Instructions: tt.instructions,
				HTTPClient: newTestClient(func(req *http.Request) (*http.Response, error) {
					capturedBody, _ = io.ReadAll(req.Body)
					return textResponse(http.StatusOK, "text/event-stream", responsesSSEText), nil
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
			for range events {
			}

			var body map[string]any
			if err := json.Unmarshal(capturedBody, &body); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			got, present := body["instructions"]
			if tt.wantPresent {
				if !present {
					t.Fatal("body missing instructions field")
				}
				if got != tt.instructions {
					t.Errorf("instructions = %v, want %q", got, tt.instructions)
				}
			} else if present {
				t.Errorf("instructions field present unexpectedly: %v", got)
			}
		})
	}
}

func TestOpenAIProvider_EndpointDispatch(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		endpoint string
		wantPath string
	}{
		{"default routes to chat completions", EndpointChatCompletions, "/chat/completions"},
		{"explicit empty routes to chat completions", "", "/chat/completions"},
		{"responses routes to /responses", EndpointResponses, "/responses"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var gotPath string
			p := New(Config{
				BaseURL:     "http://provider.test",
				TokenSource: auth.NewStatic("k"),
				Model:       "m",
				Endpoint:    tt.endpoint,
				HTTPClient: newTestClient(func(req *http.Request) (*http.Response, error) {
					gotPath = req.URL.Path
					// Both endpoints terminate cleanly with their own
					// done sentinel; an empty body works for the test
					// because we're only checking the request path.
					if tt.endpoint == EndpointResponses {
						return textResponse(http.StatusOK, "text/event-stream", "event: response.completed\ndata: {}\n\n"), nil
					}
					return textResponse(http.StatusOK, "text/event-stream", "data: [DONE]\n\n"), nil
				}),
				MaxRetries: 1,
			})

			events, err := p.Complete(context.Background(), Request{
				Messages: []conversation.Message{userText("x")},
				Stream:   true,
			})
			if err != nil {
				t.Fatalf("Complete error: %v", err)
			}
			// Drain to trigger the request.
			for range events {
			}

			if gotPath != tt.wantPath {
				t.Errorf("path = %q, want %q", gotPath, tt.wantPath)
			}
		})
	}
}

func TestMessagesToInput(t *testing.T) {
	t.Parallel()

	msgs := []conversation.Message{
		{Role: conversation.RoleSystem, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "you are helpful"}}},
		{Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "what time is it"}}},
		{
			Role: conversation.RoleAssistant,
			Blocks: []conversation.Block{
				{Type: conversation.BlockText, Text: "let me check"},
				{Type: conversation.BlockToolUse, ToolCallID: "call_1", ToolName: "current_time", ToolArgsJSON: `{}`},
			},
		},
		{
			Role: conversation.RoleTool,
			Blocks: []conversation.Block{
				{Type: conversation.BlockToolResult, ToolCallID: "call_1", Text: "2024-01-15"},
			},
		},
		{
			Role:   conversation.RoleAssistant,
			Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "It's January 15."}},
		},
	}

	input, err := messagesToInput(msgs)
	if err != nil {
		t.Fatalf("messagesToInput: %v", err)
	}

	want := []map[string]any{
		{
			"type": "message",
			"role": "system",
			"content": []map[string]any{
				{"type": "input_text", "text": "you are helpful"},
			},
		},
		{
			"type": "message",
			"role": "user",
			"content": []map[string]any{
				{"type": "input_text", "text": "what time is it"},
			},
		},
		{
			"type": "message",
			"role": "assistant",
			"content": []map[string]any{
				{"type": "output_text", "text": "let me check"},
			},
		},
		{
			"type":      "function_call",
			"call_id":   "call_1",
			"name":      "current_time",
			"arguments": `{}`,
		},
		{
			"type":    "function_call_output",
			"call_id": "call_1",
			"output":  "2024-01-15",
		},
		{
			"type": "message",
			"role": "assistant",
			"content": []map[string]any{
				{"type": "output_text", "text": "It's January 15."},
			},
		},
	}

	if !reflect.DeepEqual(input, want) {
		// Marshal both to JSON for a readable diff.
		gotJSON, _ := json.MarshalIndent(input, "", "  ")
		wantJSON, _ := json.MarshalIndent(want, "", "  ")
		t.Errorf("messagesToInput mismatch:\n--- got ---\n%s\n--- want ---\n%s", gotJSON, wantJSON)
	}
}

// TestMessagesToInput_InterleavedBlocksPreservesOrder verifies the
// Responses API serializer preserves the interleaved thinking/text/
// tool_use ordering a model produced. The /responses wire format is
// the only provider format that can faithfully round-trip this, so
// this test is the Phase 1 replay-fidelity gate.
func TestMessagesToInput_InterleavedBlocksPreservesOrder(t *testing.T) {
	t.Parallel()

	msgs := []conversation.Message{
		{Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "go"}}},
		{
			Role: conversation.RoleAssistant,
			Blocks: []conversation.Block{
				{Type: conversation.BlockThinking, Text: "plan"},
				{Type: conversation.BlockText, Text: "Starting."},
				{Type: conversation.BlockToolUse, ToolCallID: "c1", ToolName: "read_file", ToolArgsJSON: `{"p":"a"}`},
				{Type: conversation.BlockThinking, Text: "continue"},
				{Type: conversation.BlockText, Text: "Done."},
			},
		},
	}

	input, err := messagesToInput(msgs)
	if err != nil {
		t.Fatalf("messagesToInput: %v", err)
	}

	// Expect user message, then five items for the assistant turn in
	// block order: reasoning, message, function_call, reasoning, message.
	if len(input) != 6 {
		t.Fatalf("input items = %d, want 6", len(input))
	}
	wantTypes := []string{"message", "reasoning", "message", "function_call", "reasoning", "message"}
	for i, want := range wantTypes {
		if got, _ := input[i]["type"].(string); got != want {
			t.Errorf("input[%d].type = %q, want %q", i, got, want)
		}
	}
}

func TestMessagesToInput_UserImageBlock(t *testing.T) {
	t.Parallel()

	msgs := []conversation.Message{
		{
			Role: conversation.RoleUser,
			Blocks: []conversation.Block{
				{Type: conversation.BlockText, Text: "see this"},
				{Type: conversation.BlockImage, MimeType: "image/png", ImageData: []byte{0x01, 0x02}},
			},
		},
	}
	input, err := messagesToInput(msgs)
	if err != nil {
		t.Fatalf("messagesToInput: %v", err)
	}
	if len(input) != 1 {
		t.Fatalf("input len = %d, want 1", len(input))
	}
	content, _ := input[0]["content"].([]map[string]any)
	if len(content) != 2 {
		t.Fatalf("content parts = %d, want 2", len(content))
	}
	if content[0]["type"] != "input_text" {
		t.Errorf("content[0].type = %v", content[0]["type"])
	}
	if content[1]["type"] != "input_image" {
		t.Errorf("content[1].type = %v", content[1]["type"])
	}
	if url, _ := content[1]["image_url"].(string); !strings.HasPrefix(url, "data:image/png;base64,") {
		t.Errorf("image_url = %q", url)
	}
}

func TestMessagesToInput_AssistantToolCallNoText(t *testing.T) {
	t.Parallel()

	// Assistant message that's pure tool call (no text) should NOT
	// emit an empty message item — the API rejects empty content.
	msgs := []conversation.Message{
		{Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "do it"}}},
		{
			Role: conversation.RoleAssistant,
			Blocks: []conversation.Block{
				{Type: conversation.BlockToolUse, ToolCallID: "call_1", ToolName: "act", ToolArgsJSON: `{}`},
			},
		},
	}

	input, err := messagesToInput(msgs)
	if err != nil {
		t.Fatalf("messagesToInput: %v", err)
	}

	if len(input) != 2 {
		t.Fatalf("got %d items, want 2 (user + function_call)", len(input))
	}
	if input[1]["type"] != "function_call" {
		t.Errorf("item[1].type = %v, want function_call", input[1]["type"])
	}
}

func TestMessagesToInput_ToolMessageMissingID(t *testing.T) {
	t.Parallel()

	msgs := []conversation.Message{
		{
			Role: conversation.RoleTool,
			Blocks: []conversation.Block{
				{Type: conversation.BlockToolResult, Text: "result"}, // missing ToolCallID
			},
		},
	}
	if _, err := messagesToInput(msgs); err == nil {
		t.Error("expected error for tool_result block without ToolCallID")
	}
}

// TestReadResponsesSSE_EmitsUsage asserts that the usage block on a
// response.completed event is emitted as EventTypeUsage before
// EventTypeDone, with /responses' input_tokens/output_tokens mapped
// onto the canonical PromptTokens/CompletionTokens shape.
func TestReadResponsesSSE_EmitsUsage(t *testing.T) {
	t.Parallel()

	stream := "event: response.output_text.delta\n" +
		`data: {"output_index":0,"delta":"hi"}` + "\n\n" +
		"event: response.completed\n" +
		`data: {"response":{"status":"completed","usage":{"input_tokens":111,"output_tokens":22,"total_tokens":133}}}` + "\n\n"

	p := &OpenAIProvider{}
	events := make(chan Event, 16)
	go func() {
		defer close(events)
		p.readResponsesSSE(strings.NewReader(stream), events)
	}()

	var usage *TokenUsage
	var sawDone bool
	var doneAfterUsage bool
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
				doneAfterUsage = true
			}
		case EventTypeError:
			t.Fatalf("unexpected error: %v", ev.Err)
		}
	}
	if !sawDone {
		t.Error("expected done event")
	}
	if usage == nil {
		t.Fatal("expected EventTypeUsage")
	}
	if !doneAfterUsage {
		t.Error("EventTypeUsage must be emitted before EventTypeDone")
	}
	if usage.PromptTokens != 111 || usage.CompletionTokens != 22 || usage.TotalTokens != 133 {
		t.Errorf("usage = %+v, want {111,22,133}", usage)
	}
}

func TestReadResponsesSSE_HandlesLargeCompletedPayload(t *testing.T) {
	t.Parallel()

	// response.completed payloads in real life carry the full response
	// object and frequently exceed bufio.Scanner's default 64 KiB
	// limit. Construct a >64 KiB payload to verify the parser tolerates
	// it without truncation or hanging.
	bigText := strings.Repeat("x", 80*1024)
	stream := "event: response.output_text.delta\n" +
		`data: {"output_index":0,"delta":"hi"}` + "\n\n" +
		"event: response.completed\n" +
		`data: {"response":{"status":"completed","blob":"` + bigText + `"}}` + "\n\n"

	p := &OpenAIProvider{}
	events := make(chan Event, 16)
	go func() {
		defer close(events)
		p.readResponsesSSE(strings.NewReader(stream), events)
	}()

	var sawDelta, sawDone bool
	for ev := range events {
		switch ev.Type {
		case EventTypeBlockDelta:
			if ev.BlockType == conversation.BlockText {
				sawDelta = true
			}
		case EventTypeDone:
			sawDone = true
		case EventTypeError:
			t.Fatalf("unexpected error: %v", ev.Err)
		}
	}

	if !sawDelta {
		t.Error("expected text block delta")
	}
	if !sawDone {
		t.Error("expected done event after >64 KiB completed payload")
	}
}
