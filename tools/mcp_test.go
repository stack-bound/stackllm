package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"
)

func TestNewMCPToolAndCall(t *testing.T) {
	orig := http.DefaultClient
	http.DefaultClient = &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		body, err := io.ReadAll(req.Body)
		if err != nil {
			t.Fatalf("ReadAll: %v", err)
		}

		var rpcReq map[string]any
		if err := json.Unmarshal(body, &rpcReq); err != nil {
			t.Fatalf("Unmarshal: %v", err)
		}

		var payload any
		switch rpcReq["method"] {
		case "tools/list":
			payload = map[string]any{
				"jsonrpc": "2.0",
				"id":      1,
				"result": map[string]any{
					"tools": []map[string]any{
						{
							"name":        "echo",
							"description": "Echo text",
							"inputSchema": map[string]any{"type": "object"},
						},
					},
				},
			}
		case "tools/call":
			payload = map[string]any{
				"jsonrpc": "2.0",
				"id":      1,
				"result": map[string]any{
					"content": []map[string]any{
						{"type": "text", "text": "hello"},
					},
				},
			}
		default:
			t.Fatalf("unexpected method: %v", rpcReq["method"])
		}

		data, _ := json.Marshal(payload)
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(bytes.NewReader(data)),
		}, nil
	})}
	defer func() { http.DefaultClient = orig }()

	tool, err := NewMCPTool(context.Background(), "http://mcp.test", "echo")
	if err != nil {
		t.Fatalf("NewMCPTool error: %v", err)
	}

	result, err := tool.Call(context.Background(), `{"text":"hello"}`)
	if err != nil {
		t.Fatalf("Call error: %v", err)
	}
	if result != "hello" {
		t.Fatalf("result = %q, want hello", result)
	}
}

func TestNewMCPRegistry(t *testing.T) {
	orig := http.DefaultClient
	http.DefaultClient = &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		data, _ := json.Marshal(map[string]any{
			"jsonrpc": "2.0",
			"id":      1,
			"result": map[string]any{
				"tools": []map[string]any{
					{"name": "one", "description": "First", "inputSchema": map[string]any{"type": "object"}},
					{"name": "two", "description": "Second", "inputSchema": map[string]any{"type": "object"}},
				},
			},
		})
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(bytes.NewReader(data)),
		}, nil
	})}
	defer func() { http.DefaultClient = orig }()

	reg, err := NewMCPRegistry(context.Background(), "http://mcp.test")
	if err != nil {
		t.Fatalf("NewMCPRegistry error: %v", err)
	}
	if len(reg.Definitions()) != 2 {
		t.Fatalf("Definitions len = %d, want 2", len(reg.Definitions()))
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}
