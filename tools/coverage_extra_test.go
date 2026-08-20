package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// --- registry: pointer-argument tools and signature validation ---

type ptrArgs struct {
	Name string `json:"name"`
}

func TestRegister_PointerStructArg(t *testing.T) {
	t.Parallel()

	r := NewRegistry()
	err := r.Register("greet", "Greets by name", func(_ context.Context, args *ptrArgs) (string, error) {
		return "hello " + args.Name, nil
	})
	if err != nil {
		t.Fatalf("Register with *struct arg: %v", err)
	}

	out, err := r.Dispatch(context.Background(), "greet", `{"name":"matt"}`)
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	if out != "hello matt" {
		t.Errorf("Dispatch = %q, want %q", out, "hello matt")
	}
}

func TestRegister_InvalidSignatures(t *testing.T) {
	t.Parallel()

	type args struct{}
	tests := []struct {
		name    string
		fn      any
		wantErr string
	}{
		{
			name:    "first return not string",
			fn:      func(_ context.Context, _ args) (int, error) { return 0, nil },
			wantErr: "first return value must be string",
		},
		{
			name:    "second return not error",
			fn:      func(_ context.Context, _ args) (string, string) { return "", "" },
			wantErr: "second return value must be error",
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			r := NewRegistry()
			err := r.Register("bad", "bad", tt.fn)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error = %v, want containing %q", err, tt.wantErr)
			}
		})
	}
}

// --- schema: pointer deref, uint, and fallback kinds ---

func TestSchemaOf_PointerAndFallbackKinds(t *testing.T) {
	t.Parallel()

	type shape struct {
		Count uint           `json:"count"`
		Meta  map[string]any `json:"meta"`
	}

	schema := SchemaOf(&shape{})
	if schema["type"] != "object" {
		t.Fatalf("schema type = %v, want object", schema["type"])
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties missing: %+v", schema)
	}
	count, ok := props["count"].(map[string]any)
	if !ok || count["type"] != "number" {
		t.Errorf("uint field schema = %+v, want number", props["count"])
	}
	meta, ok := props["meta"].(map[string]any)
	if !ok || meta["type"] != "string" {
		t.Errorf("map field schema = %+v, want string fallback", props["meta"])
	}
}

// --- MCP ---

// mcpServer builds an httptest server implementing the minimal JSON-RPC
// surface MCPTool speaks. handle receives the decoded request and
// returns the raw JSON to write as the "result" (or an rpc error).
func mcpServer(t *testing.T, handle func(method string, params map[string]any) (result string, rpcErr string)) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Method string         `json:"method"`
			Params map[string]any `json:"params"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode rpc request: %v", err)
		}
		result, rpcErr := handle(req.Method, req.Params)
		w.Header().Set("Content-Type", "application/json")
		if rpcErr != "" {
			fmt.Fprintf(w, `{"error":{"code":-1,"message":%q}}`, rpcErr)
			return
		}
		fmt.Fprintf(w, `{"result":%s}`, result)
	}))
	t.Cleanup(srv.Close)
	return srv
}

const toolListJSON = `{"tools":[{"name":"echo","description":"echoes","inputSchema":{"type":"object"}}]}`

func TestNewMCPTool_NotFound(t *testing.T) {
	t.Parallel()
	srv := mcpServer(t, func(method string, _ map[string]any) (string, string) {
		return toolListJSON, ""
	})

	_, err := NewMCPTool(context.Background(), srv.URL, "missing")
	if err == nil || !strings.Contains(err.Error(), `"missing" not found`) {
		t.Errorf("error = %v, want not-found error", err)
	}
}

func TestNewMCPTool_ServerError(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)

	if _, err := NewMCPTool(context.Background(), srv.URL, "echo"); err == nil {
		t.Error("expected error from 500 during discovery")
	}
	if _, err := NewMCPRegistry(context.Background(), srv.URL); err == nil {
		t.Error("expected error from 500 during registry discovery")
	}
}

func TestMCPTool_Call_TextJoining(t *testing.T) {
	t.Parallel()

	// Content mixes a non-text item with two text items: the result must
	// join the text items with a newline and skip the image.
	srv := mcpServer(t, func(method string, params map[string]any) (string, string) {
		if method != "tools/call" {
			return toolListJSON, ""
		}
		if params["name"] != "echo" {
			t.Errorf("tool name = %v, want echo", params["name"])
		}
		return `{"content":[
			{"type":"image","text":"IGNORED"},
			{"type":"text","text":"line one"},
			{"type":"text","text":"line two"}
		]}`, ""
	})

	tool := &MCPTool{ServerURL: srv.URL, ToolName: "echo", HTTPClient: srv.Client()}
	out, err := tool.Call(context.Background(), `{"msg":"hi"}`)
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if out != "line one\nline two" {
		t.Errorf("Call = %q, want joined text lines", out)
	}
}

func TestMCPTool_Call_StructuredContentFallback(t *testing.T) {
	t.Parallel()
	srv := mcpServer(t, func(method string, _ map[string]any) (string, string) {
		return `{"content":[],"structuredContent":{"answer":42}}`, ""
	})

	tool := &MCPTool{ServerURL: srv.URL, ToolName: "calc"}
	out, err := tool.Call(context.Background(), `{}`)
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal([]byte(out), &decoded); err != nil {
		t.Fatalf("output not JSON: %q", out)
	}
	if decoded["answer"] != float64(42) {
		t.Errorf("structured output = %q, want answer 42", out)
	}
}

func TestMCPTool_Call_EmptyResult(t *testing.T) {
	t.Parallel()
	srv := mcpServer(t, func(method string, _ map[string]any) (string, string) {
		return `{"content":[]}`, ""
	})

	tool := &MCPTool{ServerURL: srv.URL, ToolName: "noop"}
	out, err := tool.Call(context.Background(), `{}`)
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if out != "" {
		t.Errorf("Call = %q, want empty string", out)
	}
}

func TestMCPTool_Call_IsError(t *testing.T) {
	t.Parallel()
	srv := mcpServer(t, func(method string, _ map[string]any) (string, string) {
		return `{"isError":true}`, ""
	})

	tool := &MCPTool{ServerURL: srv.URL, ToolName: "boom"}
	if _, err := tool.Call(context.Background(), `{}`); err == nil || !strings.Contains(err.Error(), "returned error") {
		t.Errorf("error = %v, want tool-returned-error", err)
	}
}

func TestMCPTool_Call_BadArguments(t *testing.T) {
	t.Parallel()
	tool := &MCPTool{ServerURL: "http://unused.test", ToolName: "echo"}
	if _, err := tool.Call(context.Background(), `not json`); err == nil || !strings.Contains(err.Error(), "decode arguments") {
		t.Errorf("error = %v, want decode arguments error", err)
	}
}

func TestMCPTool_Call_RPCFailure(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "down", http.StatusServiceUnavailable)
	}))
	t.Cleanup(srv.Close)

	tool := &MCPTool{ServerURL: srv.URL, ToolName: "echo"}
	if _, err := tool.Call(context.Background(), `{}`); err == nil || !strings.Contains(err.Error(), "status 503") {
		t.Errorf("error = %v, want status 503 error", err)
	}
}

func TestMCPTool_RPC_ErrorBranches(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		setup   func(t *testing.T) *MCPTool
		wantErr string
	}{
		{
			name: "invalid server URL",
			setup: func(t *testing.T) *MCPTool {
				return &MCPTool{ServerURL: "http://bad url with spaces\x7f", ToolName: "x"}
			},
			wantErr: "mcp request",
		},
		{
			name: "unreachable server",
			setup: func(t *testing.T) *MCPTool {
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
				srv.Close() // immediately dead
				return &MCPTool{ServerURL: srv.URL, ToolName: "x"}
			},
			wantErr: "mcp call",
		},
		{
			name: "invalid JSON response",
			setup: func(t *testing.T) *MCPTool {
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					fmt.Fprint(w, "not json")
				}))
				t.Cleanup(srv.Close)
				return &MCPTool{ServerURL: srv.URL, ToolName: "x"}
			},
			wantErr: "decode response",
		},
		{
			name: "rpc error object",
			setup: func(t *testing.T) *MCPTool {
				srv := mcpServer(t, func(string, map[string]any) (string, string) {
					return "", "tool exploded"
				})
				return &MCPTool{ServerURL: srv.URL, ToolName: "x"}
			},
			wantErr: "tool exploded",
		},
		{
			name: "result shape mismatch",
			setup: func(t *testing.T) *MCPTool {
				srv := mcpServer(t, func(string, map[string]any) (string, string) {
					return `"a plain string"`, ""
				})
				return &MCPTool{ServerURL: srv.URL, ToolName: "x"}
			},
			wantErr: "decode result",
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			tool := tt.setup(t)
			_, err := tool.Call(context.Background(), `{}`)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error = %v, want containing %q", err, tt.wantErr)
			}
		})
	}
}
