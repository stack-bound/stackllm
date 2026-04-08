package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// MCPTool wraps a single tool from an MCP server as a Tool.
type MCPTool struct {
	ServerURL  string
	ToolName   string
	HTTPClient *http.Client
	def        Definition
}

type mcpRPCRequest struct {
	JSONRPC string         `json:"jsonrpc"`
	ID      int            `json:"id"`
	Method  string         `json:"method"`
	Params  map[string]any `json:"params,omitempty"`
}

type mcpRPCResponse struct {
	Result json.RawMessage `json:"result"`
	Error  *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type mcpToolDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

// NewMCPTool connects to an MCP server and retrieves the named tool's schema.
func NewMCPTool(ctx context.Context, serverURL, toolName string) (*MCPTool, error) {
	client := &MCPTool{ServerURL: serverURL}
	defs, err := client.listTools(ctx)
	if err != nil {
		return nil, err
	}
	for _, def := range defs {
		if def.Name == toolName {
			client.ToolName = toolName
			client.def = def
			return client, nil
		}
	}
	return nil, fmt.Errorf("tools: mcp tool %q not found", toolName)
}

func (t *MCPTool) httpClient() *http.Client {
	if t.HTTPClient != nil {
		return t.HTTPClient
	}
	return http.DefaultClient
}

func (t *MCPTool) Definition() Definition {
	return t.def
}

func (t *MCPTool) Call(ctx context.Context, arguments string) (string, error) {
	var args map[string]any
	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return "", fmt.Errorf("tools: mcp decode arguments: %w", err)
	}

	var result struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		StructuredContent any  `json:"structuredContent"`
		IsError           bool `json:"isError"`
	}
	if err := t.rpc(ctx, "tools/call", map[string]any{
		"name":      t.ToolName,
		"arguments": args,
	}, &result); err != nil {
		return "", err
	}
	if result.IsError {
		return "", fmt.Errorf("tools: mcp tool %q returned error", t.ToolName)
	}
	if len(result.Content) > 0 {
		var out string
		for i, item := range result.Content {
			if item.Type != "text" {
				continue
			}
			if i > 0 && out != "" {
				out += "\n"
			}
			out += item.Text
		}
		if out != "" {
			return out, nil
		}
	}
	if result.StructuredContent != nil {
		data, err := json.Marshal(result.StructuredContent)
		if err != nil {
			return "", fmt.Errorf("tools: mcp encode structured content: %w", err)
		}
		return string(data), nil
	}
	return "", nil
}

// NewMCPRegistry discovers all tools from an MCP server and registers them.
func NewMCPRegistry(ctx context.Context, serverURL string) (*Registry, error) {
	client := &MCPTool{ServerURL: serverURL}
	defs, err := client.listTools(ctx)
	if err != nil {
		return nil, err
	}

	reg := NewRegistry()
	for _, def := range defs {
		reg.Add(&MCPTool{
			ServerURL: serverURL,
			ToolName:  def.Name,
			def:       def,
		})
	}
	return reg, nil
}

func (t *MCPTool) listTools(ctx context.Context) ([]Definition, error) {
	var result struct {
		Tools []mcpToolDefinition `json:"tools"`
	}
	if err := t.rpc(ctx, "tools/list", nil, &result); err != nil {
		return nil, err
	}

	defs := make([]Definition, 0, len(result.Tools))
	for _, tool := range result.Tools {
		defs = append(defs, Definition{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  tool.InputSchema,
		})
	}
	return defs, nil
}

func (t *MCPTool) rpc(ctx context.Context, method string, params map[string]any, target any) error {
	body, err := json.Marshal(mcpRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  method,
		Params:  params,
	})
	if err != nil {
		return fmt.Errorf("tools: mcp encode request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.ServerURL, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("tools: mcp request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := t.httpClient().Do(req)
	if err != nil {
		return fmt.Errorf("tools: mcp call %q: %w", method, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("tools: mcp call %q: status %d: %s", method, resp.StatusCode, data)
	}

	var rpcResp mcpRPCResponse
	if err := json.NewDecoder(resp.Body).Decode(&rpcResp); err != nil {
		return fmt.Errorf("tools: mcp decode response: %w", err)
	}
	if rpcResp.Error != nil {
		return fmt.Errorf("tools: mcp call %q: %s", method, rpcResp.Error.Message)
	}
	if err := json.Unmarshal(rpcResp.Result, target); err != nil {
		return fmt.Errorf("tools: mcp decode result: %w", err)
	}
	return nil
}
