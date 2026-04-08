package tools

import (
	"context"
	"fmt"
	"testing"
)

func TestRegistry_Register(t *testing.T) {
	t.Parallel()

	type ReadArgs struct {
		Path string `json:"path" jsonschema:"description=File path,required"`
	}

	r := NewRegistry()
	err := r.Register("read_file", "Read a file", func(ctx context.Context, args ReadArgs) (string, error) {
		return "contents of " + args.Path, nil
	})
	if err != nil {
		t.Fatalf("Register error: %v", err)
	}

	// Verify definition.
	defs := r.Definitions()
	if len(defs) != 1 {
		t.Fatalf("Definitions() len = %d, want 1", len(defs))
	}
	if defs[0].Name != "read_file" {
		t.Errorf("Name = %q, want %q", defs[0].Name, "read_file")
	}
	if defs[0].Description != "Read a file" {
		t.Errorf("Description = %q, want %q", defs[0].Description, "Read a file")
	}

	// Verify dispatch.
	result, err := r.Dispatch(context.Background(), "read_file", `{"path":"/tmp/test"}`)
	if err != nil {
		t.Fatalf("Dispatch error: %v", err)
	}
	if result != "contents of /tmp/test" {
		t.Errorf("result = %q, want %q", result, "contents of /tmp/test")
	}
}

func TestRegistry_Register_InvalidSignatures(t *testing.T) {
	t.Parallel()

	r := NewRegistry()

	tests := []struct {
		name string
		fn   any
	}{
		{"not a function", "hello"},
		{"no args", func() (string, error) { return "", nil }},
		{"wrong first arg", func(s string, args struct{}) (string, error) { return "", nil }},
		{"wrong arg type", func(ctx context.Context, s string) (string, error) { return "", nil }},
		{"wrong return count", func(ctx context.Context, args struct{}) string { return "" }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := r.Register("test", "test", tt.fn)
			if err == nil {
				t.Error("expected error for invalid signature")
			}
		})
	}
}

func TestRegistry_Dispatch_UnknownTool(t *testing.T) {
	t.Parallel()

	r := NewRegistry()
	_, err := r.Dispatch(context.Background(), "nonexistent", "{}")
	if err == nil {
		t.Fatal("expected error for unknown tool")
	}
}

func TestRegistry_Dispatch_InvalidJSON(t *testing.T) {
	t.Parallel()

	type Args struct {
		X int `json:"x"`
	}

	r := NewRegistry()
	r.Register("test", "test", func(ctx context.Context, args Args) (string, error) {
		return "", nil
	})

	_, err := r.Dispatch(context.Background(), "test", "not json")
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestRegistry_Dispatch_ToolError(t *testing.T) {
	t.Parallel()

	type Args struct{}

	r := NewRegistry()
	r.Register("failing", "always fails", func(ctx context.Context, args Args) (string, error) {
		return "", fmt.Errorf("something went wrong")
	})

	_, err := r.Dispatch(context.Background(), "failing", "{}")
	if err == nil {
		t.Fatal("expected error from failing tool")
	}
}

func TestRegistry_Add(t *testing.T) {
	t.Parallel()

	r := NewRegistry()
	r.Add(&staticTool{
		def: Definition{Name: "custom", Description: "custom tool"},
	})

	defs := r.Definitions()
	if len(defs) != 1 || defs[0].Name != "custom" {
		t.Errorf("expected custom tool, got %v", defs)
	}
}

// staticTool is a test helper implementing Tool.
type staticTool struct {
	def    Definition
	result string
}

func (s *staticTool) Definition() Definition {
	return s.def
}

func (s *staticTool) Call(_ context.Context, _ string) (string, error) {
	return s.result, nil
}
