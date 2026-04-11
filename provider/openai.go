package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
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

	// Endpoint selects the API path: "" (default) → /chat/completions,
	// EndpointResponses ("/responses") → OpenAI Responses API. Used for
	// Copilot models that are not accessible via /chat/completions.
	Endpoint string
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

// Model returns the model name this provider is configured for. It's
// the value baked into Config.Model at construction — callers that
// override it per-request via Request.Model still win at wire time,
// but this is the stable default surface (e.g. for the TUI status
// line when the embedder didn't pass agent.WithModel).
func (p *OpenAIProvider) Model() string { return p.cfg.Model }

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

// Complete makes a streaming completion request and returns a channel of events.
//
// The request path is selected by p.cfg.Endpoint:
//   - ""                  → /chat/completions (default)
//   - EndpointResponses   → /responses (used for Copilot responses-only models)
func (p *OpenAIProvider) Complete(ctx context.Context, req Request) (<-chan Event, error) {
	switch p.cfg.Endpoint {
	case EndpointResponses:
		return p.completeResponses(ctx, req)
	default:
		return p.completeChat(ctx, req)
	}
}

// completeChat dispatches the request to the legacy /chat/completions endpoint.
func (p *OpenAIProvider) completeChat(ctx context.Context, req Request) (<-chan Event, error) {
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
		p.doStreamingPOST(ctx, url, jsonBody, events, p.readChatSSE)
	}()

	return events, nil
}

// Models returns model metadata from the provider's /models endpoint.
//
// The returned ModelMeta entries include any per-model SupportedEndpoints
// and capabilities.type fields when the upstream API exposes them
// (currently only Copilot). For OpenAI/Gemini/Ollama those fields are
// nil/empty and callers should treat the model as compatible with the
// provider's default endpoint.
func (p *OpenAIProvider) Models(ctx context.Context) ([]ModelMeta, error) {
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
			ID                 string   `json:"id"`
			SupportedEndpoints []string `json:"supported_endpoints"`
			ModelPickerEnabled *bool    `json:"model_picker_enabled"`
			Capabilities       struct {
				Type   string `json:"type"`
				Limits struct {
					MaxPromptTokens  int `json:"max_prompt_tokens"`
					MaxContextWindow int `json:"max_context_window_tokens"`
					MaxInputTokens   int `json:"max_input_tokens"`
				} `json:"limits"`
			} `json:"capabilities"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("provider: models decode: %w", err)
	}

	models := make([]ModelMeta, len(result.Data))
	for i, m := range result.Data {
		// Prefer the most specific limit upstream reports. Copilot
		// exposes max_prompt_tokens for chat models; a few historic
		// variants have used max_context_window_tokens or
		// max_input_tokens. Take whichever is non-zero first.
		cw := m.Capabilities.Limits.MaxPromptTokens
		if cw == 0 {
			cw = m.Capabilities.Limits.MaxContextWindow
		}
		if cw == 0 {
			cw = m.Capabilities.Limits.MaxInputTokens
		}
		models[i] = ModelMeta{
			ID:                 m.ID,
			SupportedEndpoints: m.SupportedEndpoints,
			Type:               m.Capabilities.Type,
			ModelPickerEnabled: m.ModelPickerEnabled,
			ContextWindow:      cw,
		}
	}
	return models, nil
}

// buildRequestBody flattens the block-shaped Messages into the legacy
// OpenAI chat completions wire format.
//
// Chat completions cannot represent interleaved thinking/text/tool_use
// in a single assistant turn. This method is intentionally lossy:
//
//   - BlockThinking blocks are dropped on serialization. The /responses
//     API preserves reasoning; the legacy chat API has no field for it.
//   - A single assistant message containing multiple text and tool_use
//     blocks is collapsed to one chat message: text blocks concatenate
//     into "content", and tool_use blocks are hoisted into tool_calls.
//   - A tool-role message that carries multiple tool_result blocks
//     emits one chat "tool" message per block (chat completions
//     requires a 1:1 mapping of tool_call_id to tool message).
//   - User messages with image blocks emit OpenAI's multi-part content
//     array (text parts + image_url parts with data URIs for inline
//     bytes).
//
// Callers that need faithful replay of interleaved output should use
// the Responses API endpoint (Endpoint=EndpointResponses) instead.
func (p *OpenAIProvider) buildRequestBody(req Request) map[string]any {
	msgs := make([]map[string]any, 0, len(req.Messages))
	for _, m := range req.Messages {
		msgs = append(msgs, messageToChatCompletions(m)...)
	}

	body := map[string]any{
		"model":    req.Model,
		"messages": msgs,
		"stream":   req.Stream,
	}

	// OpenAI only emits the usage block on streamed responses when the
	// caller opts in via stream_options.include_usage. Copilot, Gemini
	// (OpenAI-compat) and Ollama tolerate the field — they either honour
	// it or ignore it. Only set it on the chat completions path; the
	// /responses endpoint carries usage on response.completed by default
	// and rejects stream_options.
	if req.Stream {
		body["stream_options"] = map[string]any{"include_usage": true}
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

// messageToChatCompletions converts one block-shaped Message into one
// or more chat-completions message dicts. Most messages yield a single
// dict; a tool-role message with N tool_result blocks yields N dicts
// (one per block) because the chat completions API requires a 1:1
// mapping between tool_call_id and tool message.
func messageToChatCompletions(m conversation.Message) []map[string]any {
	switch m.Role {
	case conversation.RoleTool:
		results := m.ToolResults()
		if len(results) == 0 {
			// Fallback: a tool-role message without explicit
			// tool_result blocks still needs to travel as a tool
			// message so the assistant doesn't see a broken turn.
			return []map[string]any{{
				"role":    "tool",
				"content": m.TextContent(),
			}}
		}
		out := make([]map[string]any, 0, len(results))
		for _, r := range results {
			out = append(out, map[string]any{
				"role":         "tool",
				"tool_call_id": r.ToolCallID,
				"content":      r.Text,
			})
		}
		return out

	case conversation.RoleUser:
		content := userContentForChat(m)
		return []map[string]any{{
			"role":    "user",
			"content": content,
		}}

	case conversation.RoleSystem:
		return []map[string]any{{
			"role":    "system",
			"content": m.TextContent(),
		}}

	case conversation.RoleAssistant:
		msg := map[string]any{
			"role":    "assistant",
			"content": m.TextContent(),
		}
		var toolCalls []map[string]any
		for _, b := range m.Blocks {
			if b.Type != conversation.BlockToolUse {
				continue
			}
			args := b.ToolArgsJSON
			if args == "" {
				args = "{}"
			}
			toolCalls = append(toolCalls, map[string]any{
				"id":   b.ToolCallID,
				"type": "function",
				"function": map[string]any{
					"name":      b.ToolName,
					"arguments": args,
				},
			})
		}
		if len(toolCalls) > 0 {
			msg["tool_calls"] = toolCalls
		}
		return []map[string]any{msg}
	}

	return []map[string]any{{
		"role":    string(m.Role),
		"content": m.TextContent(),
	}}
}

// userContentForChat returns either a plain string (when the user
// message is pure text) or OpenAI's multi-part content array (when the
// message carries image blocks).
func userContentForChat(m conversation.Message) any {
	hasImage := false
	for _, b := range m.Blocks {
		if b.Type == conversation.BlockImage {
			hasImage = true
			break
		}
	}
	if !hasImage {
		return m.TextContent()
	}

	parts := make([]map[string]any, 0, len(m.Blocks))
	for _, b := range m.Blocks {
		switch b.Type {
		case conversation.BlockText:
			parts = append(parts, map[string]any{
				"type": "text",
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
				"type": "image_url",
				"image_url": map[string]any{
					"url": url,
				},
			})
		}
	}
	return parts
}

func (p *OpenAIProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	for k, v := range p.cfg.ExtraHeaders {
		req.Header.Set(k, v)
	}
}

// sseReader reads a streaming response body and emits provider events.
type sseReader func(io.Reader, chan<- Event)

// doStreamingPOST POSTs body to url, retries on 429/5xx, and on success
// hands the response body to reader for SSE parsing. Errors are emitted
// to events as Event{Type: EventTypeError}.
func (p *OpenAIProvider) doStreamingPOST(ctx context.Context, url string, body []byte, events chan<- Event, reader sseReader) {
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

		reader(resp.Body, events)
		resp.Body.Close()
		return
	}

	events <- Event{Type: EventTypeError, Err: fmt.Errorf("provider: max retries exceeded: %w", lastErr)}
}

type toolCallAcc struct {
	ID        string
	Name      string
	Arguments strings.Builder
	started   bool // whether a BlockStart was already emitted
}

// chatBlockKind tracks which kind of text-bearing block is currently
// open in the chat-completions stream so that we can emit a clean
// BlockEnd / BlockStart pair when the model switches between
// reasoning_content and content deltas.
type chatBlockKind int

const (
	chatBlockNone chatBlockKind = iota
	chatBlockText
	chatBlockThinking
)

// readChatSSE parses the chat completions SSE stream and emits ordered
// block events. Reasoning deltas (delta.reasoning_content, exposed by
// several OpenAI-compatible backends) are mapped to BlockThinking
// blocks; normal content deltas become BlockText. Tool call argument
// deltas are streamed as BlockDelta events on the corresponding
// BlockToolUse block.
//
// On [DONE] any still-open text / thinking block is closed, all
// accumulated tool_use blocks are finalised in index order, and a
// single EventTypeDone is emitted.
func (p *OpenAIProvider) readChatSSE(body io.Reader, events chan<- Event) {
	scanner := bufio.NewScanner(body)
	// Single delta payloads can exceed 64 KiB for multi-part or image
	// responses; give the scanner headroom.
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	toolCalls := make(map[int]*toolCallAcc)
	var toolOrder []int

	currentKind := chatBlockNone
	var currentText strings.Builder
	var lastUsage *TokenUsage

	closeCurrent := func() {
		if currentKind == chatBlockNone {
			return
		}
		var blk conversation.Block
		switch currentKind {
		case chatBlockText:
			blk = conversation.Block{ID: conversation.NewID(), Type: conversation.BlockText, Text: currentText.String()}
		case chatBlockThinking:
			blk = conversation.Block{ID: conversation.NewID(), Type: conversation.BlockThinking, Text: currentText.String()}
		}
		events <- Event{Type: EventTypeBlockEnd, BlockType: blk.Type, Block: &blk}
		currentKind = chatBlockNone
		currentText.Reset()
	}

	switchTo := func(kind chatBlockKind) conversation.BlockType {
		if currentKind != kind {
			closeCurrent()
			currentKind = kind
			var bt conversation.BlockType
			if kind == chatBlockText {
				bt = conversation.BlockText
			} else {
				bt = conversation.BlockThinking
			}
			events <- Event{Type: EventTypeBlockStart, BlockType: bt}
			return bt
		}
		if kind == chatBlockText {
			return conversation.BlockText
		}
		return conversation.BlockThinking
	}

	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		if data == "[DONE]" {
			closeCurrent()

			// Emit accumulated tool calls in index order. If we
			// never saw a BlockStart for a given tool call (arguments
			// arrived in a single chunk on finish), synthesise one
			// here so the event sequence remains well-formed.
			for _, i := range toolOrder {
				tc := toolCalls[i]
				if !tc.started {
					events <- Event{Type: EventTypeBlockStart, BlockType: conversation.BlockToolUse}
				}
				args := tc.Arguments.String()
				blk := conversation.Block{
					ID:           conversation.NewID(),
					Type:         conversation.BlockToolUse,
					ToolCallID:   tc.ID,
					ToolName:     tc.Name,
					ToolArgsJSON: args,
				}
				events <- Event{Type: EventTypeBlockEnd, BlockType: conversation.BlockToolUse, Block: &blk}
				events <- Event{
					Type: EventTypeToolCall,
					Call: &conversation.ToolCall{
						ID:        tc.ID,
						Name:      tc.Name,
						Arguments: args,
					},
				}
			}
			if lastUsage != nil {
				events <- Event{Type: EventTypeUsage, Usage: lastUsage}
			}
			events <- Event{Type: EventTypeDone}
			return
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
					Reasoning        string `json:"reasoning"`
					ToolCalls        []struct {
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
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			} `json:"usage"`
		}

		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		// The final chunk from an OpenAI-compatible stream carries
		// usage in a standalone message with no choices — accept it
		// from any chunk that has it, to be resilient to providers
		// that interleave usage on the last choice chunk too.
		if chunk.Usage != nil && (chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0 || chunk.Usage.TotalTokens > 0) {
			lastUsage = &TokenUsage{
				PromptTokens:     chunk.Usage.PromptTokens,
				CompletionTokens: chunk.Usage.CompletionTokens,
				TotalTokens:      chunk.Usage.TotalTokens,
			}
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		delta := chunk.Choices[0].Delta

		// Some backends expose reasoning under different field names.
		reasoning := delta.ReasoningContent
		if reasoning == "" {
			reasoning = delta.Reasoning
		}

		if reasoning != "" {
			bt := switchTo(chatBlockThinking)
			currentText.WriteString(reasoning)
			events <- Event{Type: EventTypeBlockDelta, BlockType: bt, Content: reasoning}
		}

		if delta.Content != "" {
			bt := switchTo(chatBlockText)
			currentText.WriteString(delta.Content)
			events <- Event{Type: EventTypeBlockDelta, BlockType: bt, Content: delta.Content}
		}

		for _, tc := range delta.ToolCalls {
			acc, ok := toolCalls[tc.Index]
			if !ok {
				acc = &toolCallAcc{}
				toolCalls[tc.Index] = acc
				toolOrder = append(toolOrder, tc.Index)
				// A new tool_use block has opened. Close any still-open
				// text/thinking block first so ordering is preserved.
				closeCurrent()
				events <- Event{Type: EventTypeBlockStart, BlockType: conversation.BlockToolUse}
				acc.started = true
			}
			if tc.ID != "" {
				acc.ID = tc.ID
			}
			if tc.Function.Name != "" {
				acc.Name = tc.Function.Name
			}
			if tc.Function.Arguments != "" {
				acc.Arguments.WriteString(tc.Function.Arguments)
				events <- Event{
					Type:      EventTypeBlockDelta,
					BlockType: conversation.BlockToolUse,
					Content:   tc.Function.Arguments,
				}
			}
		}
	}
	closeCurrent()
}
