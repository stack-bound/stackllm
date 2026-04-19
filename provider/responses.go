package provider

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/stack-bound/stackllm/conversation"
)

// completeResponses dispatches the request to the OpenAI Responses API
// (POST /responses), used for Copilot models that are not accessible via
// /chat/completions (e.g. gpt-5.4-mini, gpt-5.x-codex).
//
// The wire format differs from chat completions in three important ways:
//   - request body uses "input" (an array of typed items), not "messages"
//   - tools omit the "function" wrapper: {"type":"function","name":...}
//   - the SSE stream uses named events (response.output_text.delta,
//     response.function_call_arguments.delta, response.completed, …)
//     and does not emit a [DONE] sentinel.
func (p *OpenAIProvider) completeResponses(ctx context.Context, req Request) (<-chan Event, error) {
	body, err := p.buildResponsesBody(req)
	if err != nil {
		return nil, fmt.Errorf("provider: build responses request: %w", err)
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("provider: marshal responses request: %w", err)
	}

	url := p.cfg.BaseURL + "/responses"
	if p.cfg.APIVersion != "" {
		url += "?api-version=" + p.cfg.APIVersion
	}

	events := make(chan Event, 64)

	go func() {
		defer close(events)
		p.doStreamingPOST(ctx, url, jsonBody, events, p.readResponsesSSE)
	}()

	return events, nil
}

// buildResponsesBody constructs the JSON body for a /responses call.
func (p *OpenAIProvider) buildResponsesBody(req Request) (map[string]any, error) {
	input, err := messagesToInput(req.Messages)
	if err != nil {
		return nil, err
	}

	body := map[string]any{
		"input":  input,
		"stream": req.Stream,
	}

	if req.Model != "" {
		body["model"] = req.Model
	} else {
		body["model"] = p.cfg.Model
	}

	if p.cfg.Instructions != "" {
		body["instructions"] = p.cfg.Instructions
	}

	if p.cfg.DisableStore {
		body["store"] = false
	}

	if req.MaxTokens > 0 {
		body["max_output_tokens"] = req.MaxTokens
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}

	if len(req.Tools) > 0 {
		oaiTools := make([]map[string]any, len(req.Tools))
		for i, t := range req.Tools {
			// Responses API uses a flat tool shape — no nested
			// "function": {...} wrapper.
			oaiTools[i] = map[string]any{
				"type":        "function",
				"name":        t.Name,
				"description": t.Description,
				"parameters":  t.Parameters,
			}
		}
		body["tools"] = oaiTools
	}

	return body, nil
}

// messagesToInput converts a block-shaped message slice into the
// Responses API "input" array. The Responses API is the only provider
// wire format that can faithfully round-trip interleaved
// thinking/text/tool_use blocks; this function preserves block order
// within each message and across the conversation.
//
// Mapping:
//   - system / user BlockText    → {"type":"message","role":"...",
//                                    "content":[{"type":"input_text","text":...}]}
//   - user BlockImage            → {"type":"message","role":"user",
//                                    "content":[{"type":"input_image","image_url":...}]}
//   - assistant BlockText        → {"type":"message","role":"assistant",
//                                    "content":[{"type":"output_text","text":...}]}
//   - assistant BlockThinking    → {"type":"reasoning",
//                                    "summary":[{"type":"summary_text","text":...}]}
//   - assistant BlockToolUse     → {"type":"function_call","call_id":...,
//                                    "name":...,"arguments":...}
//   - tool BlockToolResult       → {"type":"function_call_output","call_id":...,
//                                    "output":...}
func messagesToInput(msgs []conversation.Message) ([]map[string]any, error) {
	out := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		switch m.Role {
		case conversation.RoleSystem:
			text := m.TextContent()
			out = append(out, map[string]any{
				"type": "message",
				"role": "system",
				"content": []map[string]any{
					{"type": "input_text", "text": text},
				},
			})

		case conversation.RoleUser:
			parts := make([]map[string]any, 0, len(m.Blocks))
			for _, b := range m.Blocks {
				switch b.Type {
				case conversation.BlockText:
					parts = append(parts, map[string]any{
						"type": "input_text",
						"text": b.Text,
					})
				case conversation.BlockImage:
					url := b.ImageURL
					if url == "" && len(b.ImageData) > 0 {
						mime := b.MimeType
						if mime == "" {
							mime = "image/png"
						}
						url = "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(b.ImageData)
					}
					parts = append(parts, map[string]any{
						"type":      "input_image",
						"image_url": url,
					})
				}
			}
			if len(parts) == 0 {
				parts = append(parts, map[string]any{
					"type": "input_text",
					"text": "",
				})
			}
			out = append(out, map[string]any{
				"type":    "message",
				"role":    "user",
				"content": parts,
			})

		case conversation.RoleAssistant:
			for _, b := range m.Blocks {
				switch b.Type {
				case conversation.BlockText:
					out = append(out, map[string]any{
						"type": "message",
						"role": "assistant",
						"content": []map[string]any{
							{"type": "output_text", "text": b.Text},
						},
					})
				case conversation.BlockThinking:
					out = append(out, map[string]any{
						"type": "reasoning",
						"summary": []map[string]any{
							{"type": "summary_text", "text": b.Text},
						},
					})
				case conversation.BlockToolUse:
					args := b.ToolArgsJSON
					if args == "" {
						args = "{}"
					}
					out = append(out, map[string]any{
						"type":      "function_call",
						"call_id":   b.ToolCallID,
						"name":      b.ToolName,
						"arguments": args,
					})
				}
			}

		case conversation.RoleTool:
			results := m.ToolResults()
			if len(results) == 0 {
				return nil, fmt.Errorf("provider: tool message has no tool_result blocks")
			}
			for _, r := range results {
				if r.ToolCallID == "" {
					return nil, fmt.Errorf("provider: tool_result block missing ToolCallID")
				}
				out = append(out, map[string]any{
					"type":    "function_call_output",
					"call_id": r.ToolCallID,
					"output":  r.Text,
				})
			}

		default:
			return nil, fmt.Errorf("provider: unknown role %q", m.Role)
		}
	}
	return out, nil
}

// responsesItemAcc accumulates a single output_item as its deltas arrive.
// output_index identifies the slot in the model's output array; we close
// items in the order response.output_item.done fires, preserving the
// natural production order of interleaved reasoning/text/tool_use.
type responsesItemAcc struct {
	kind        string // "reasoning" | "message" | "function_call"
	blockType   conversation.BlockType
	text        strings.Builder
	callID      string
	name        string
	arguments   strings.Builder
	blockStarted bool
}

// readResponsesSSE parses an SSE stream from POST /responses and emits
// ordered block events. Each output_item (reasoning, message,
// function_call) becomes one block. Items are closed in the order the
// response.output_item.done events fire, which matches the order the
// model produced them — so an interleaved
// reasoning → message → reasoning → function_call turn is captured
// verbatim.
//
// The stream is terminated by a `response.completed` event; there is
// no `[DONE]` sentinel.
func (p *OpenAIProvider) readResponsesSSE(body io.Reader, events chan<- Event) {
	scanner := bufio.NewScanner(body)
	// Some events (especially response.completed) carry the full response
	// object, which can exceed bufio's default 64 KiB line limit.
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	items := make(map[int]*responsesItemAcc)

	getOrCreate := func(index int, kind, itemType string) *responsesItemAcc {
		acc, ok := items[index]
		if ok {
			return acc
		}
		acc = &responsesItemAcc{kind: kind}
		switch itemType {
		case "reasoning":
			acc.blockType = conversation.BlockThinking
		case "message":
			acc.blockType = conversation.BlockText
		case "function_call":
			acc.blockType = conversation.BlockToolUse
		}
		items[index] = acc
		return acc
	}

	ensureStart := func(acc *responsesItemAcc) {
		if acc.blockStarted {
			return
		}
		acc.blockStarted = true
		events <- Event{Type: EventTypeBlockStart, BlockType: acc.blockType}
	}

	var currentEvent string

	for scanner.Scan() {
		line := scanner.Text()

		if line == "" {
			currentEvent = ""
			continue
		}

		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimPrefix(line, "event: ")
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		switch currentEvent {
		case "response.output_item.added":
			var payload struct {
				OutputIndex int `json:"output_index"`
				Item        struct {
					Type   string `json:"type"`
					CallID string `json:"call_id"`
					Name   string `json:"name"`
				} `json:"item"`
			}
			if err := json.Unmarshal([]byte(data), &payload); err != nil {
				continue
			}
			acc := getOrCreate(payload.OutputIndex, payload.Item.Type, payload.Item.Type)
			if payload.Item.CallID != "" {
				acc.callID = payload.Item.CallID
			}
			if payload.Item.Name != "" {
				acc.name = payload.Item.Name
			}
			ensureStart(acc)

		case "response.output_text.delta":
			var payload struct {
				OutputIndex int    `json:"output_index"`
				Delta       string `json:"delta"`
			}
			if err := json.Unmarshal([]byte(data), &payload); err != nil {
				continue
			}
			if payload.Delta == "" {
				continue
			}
			acc := getOrCreate(payload.OutputIndex, "message", "message")
			ensureStart(acc)
			acc.text.WriteString(payload.Delta)
			events <- Event{
				Type:      EventTypeBlockDelta,
				BlockType: conversation.BlockText,
				Content:   payload.Delta,
			}

		case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
			var payload struct {
				OutputIndex int    `json:"output_index"`
				Delta       string `json:"delta"`
			}
			if err := json.Unmarshal([]byte(data), &payload); err != nil {
				continue
			}
			if payload.Delta == "" {
				continue
			}
			acc := getOrCreate(payload.OutputIndex, "reasoning", "reasoning")
			ensureStart(acc)
			acc.text.WriteString(payload.Delta)
			events <- Event{
				Type:      EventTypeBlockDelta,
				BlockType: conversation.BlockThinking,
				Content:   payload.Delta,
			}

		case "response.function_call_arguments.delta":
			var payload struct {
				OutputIndex int    `json:"output_index"`
				Delta       string `json:"delta"`
			}
			if err := json.Unmarshal([]byte(data), &payload); err != nil {
				continue
			}
			if payload.Delta == "" {
				continue
			}
			acc := getOrCreate(payload.OutputIndex, "function_call", "function_call")
			ensureStart(acc)
			acc.arguments.WriteString(payload.Delta)
			events <- Event{
				Type:      EventTypeBlockDelta,
				BlockType: conversation.BlockToolUse,
				Content:   payload.Delta,
			}

		case "response.output_item.done":
			var payload struct {
				OutputIndex int `json:"output_index"`
				Item        struct {
					Type      string `json:"type"`
					CallID    string `json:"call_id"`
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
					Content   []struct {
						Type string `json:"type"`
						Text string `json:"text"`
					} `json:"content"`
					Summary []struct {
						Type string `json:"type"`
						Text string `json:"text"`
					} `json:"summary"`
				} `json:"item"`
			}
			if err := json.Unmarshal([]byte(data), &payload); err != nil {
				continue
			}
			acc := getOrCreate(payload.OutputIndex, payload.Item.Type, payload.Item.Type)

			// Prefer the canonical values from the done payload; fall
			// back to whatever we accumulated from deltas.
			switch payload.Item.Type {
			case "message":
				if len(payload.Item.Content) > 0 {
					acc.text.Reset()
					for _, c := range payload.Item.Content {
						if c.Type == "output_text" {
							acc.text.WriteString(c.Text)
						}
					}
				}
			case "reasoning":
				if len(payload.Item.Summary) > 0 {
					acc.text.Reset()
					for _, s := range payload.Item.Summary {
						if s.Type == "summary_text" {
							acc.text.WriteString(s.Text)
						}
					}
				}
			case "function_call":
				if payload.Item.CallID != "" {
					acc.callID = payload.Item.CallID
				}
				if payload.Item.Name != "" {
					acc.name = payload.Item.Name
				}
				if payload.Item.Arguments != "" {
					acc.arguments.Reset()
					acc.arguments.WriteString(payload.Item.Arguments)
				}
			}

			ensureStart(acc)

			switch acc.blockType {
			case conversation.BlockText:
				text := acc.text.String()
				blk := conversation.Block{ID: conversation.NewID(), Type: conversation.BlockText, Text: text}
				events <- Event{Type: EventTypeBlockEnd, BlockType: conversation.BlockText, Block: &blk}
			case conversation.BlockThinking:
				text := acc.text.String()
				blk := conversation.Block{ID: conversation.NewID(), Type: conversation.BlockThinking, Text: text}
				events <- Event{Type: EventTypeBlockEnd, BlockType: conversation.BlockThinking, Block: &blk}
			case conversation.BlockToolUse:
				args := acc.arguments.String()
				blk := conversation.Block{
					ID:           conversation.NewID(),
					Type:         conversation.BlockToolUse,
					ToolCallID:   acc.callID,
					ToolName:     acc.name,
					ToolArgsJSON: args,
				}
				events <- Event{Type: EventTypeBlockEnd, BlockType: conversation.BlockToolUse, Block: &blk}
				events <- Event{
					Type: EventTypeToolCall,
					Call: &conversation.ToolCall{
						ID:        acc.callID,
						Name:      acc.name,
						Arguments: args,
					},
				}
			}

			delete(items, payload.OutputIndex)

		case "response.completed":
			// response.completed carries the full response object,
			// including a usage block with input/output/total token
			// counts. Forward it as EventTypeUsage before the final
			// EventTypeDone so downstream consumers see a consistent
			// ordering with the chat completions path.
			var payload struct {
				Response struct {
					Usage struct {
						InputTokens  int `json:"input_tokens"`
						OutputTokens int `json:"output_tokens"`
						TotalTokens  int `json:"total_tokens"`
					} `json:"usage"`
				} `json:"response"`
			}
			if err := json.Unmarshal([]byte(data), &payload); err == nil {
				u := payload.Response.Usage
				if u.InputTokens > 0 || u.OutputTokens > 0 || u.TotalTokens > 0 {
					events <- Event{Type: EventTypeUsage, Usage: &TokenUsage{
						PromptTokens:     u.InputTokens,
						CompletionTokens: u.OutputTokens,
						TotalTokens:      u.TotalTokens,
					}}
				}
			}
			events <- Event{Type: EventTypeDone}
			return

		case "response.failed", "response.error":
			// Best-effort: surface the upstream error message if present.
			var payload struct {
				Response struct {
					Error *struct {
						Message string `json:"message"`
						Code    string `json:"code"`
					} `json:"error"`
				} `json:"response"`
				Error *struct {
					Message string `json:"message"`
					Code    string `json:"code"`
				} `json:"error"`
			}
			_ = json.Unmarshal([]byte(data), &payload)
			msg := "responses stream error"
			if payload.Response.Error != nil && payload.Response.Error.Message != "" {
				msg = payload.Response.Error.Message
			} else if payload.Error != nil && payload.Error.Message != "" {
				msg = payload.Error.Message
			}
			events <- Event{Type: EventTypeError, Err: fmt.Errorf("provider: %s", msg)}
			return
		}
	}
}
