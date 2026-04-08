package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
)

// Registry holds registered tools and dispatches calls by name.
type Registry struct {
	mu    sync.RWMutex
	tools map[string]Tool
}

// NewRegistry creates a new empty tool registry.
func NewRegistry() *Registry {
	return &Registry{tools: make(map[string]Tool)}
}

// Register adds a function-backed tool. fn must be a function that takes a
// context.Context and a single struct argument and returns (string, error).
// The struct's fields become the tool's JSON Schema parameters.
//
// Example:
//
//	type ReadFileArgs struct {
//	    Path string `json:"path" jsonschema:"description=Absolute path to read,required"`
//	}
//	registry.Register("read_file", "Read a file from disk", func(ctx context.Context, args ReadFileArgs) (string, error) {
//	    data, err := os.ReadFile(args.Path)
//	    return string(data), err
//	})
func (r *Registry) Register(name, description string, fn any) error {
	ft := reflect.TypeOf(fn)
	if ft.Kind() != reflect.Func {
		return fmt.Errorf("tools: Register %q: expected function, got %s", name, ft.Kind())
	}

	// Validate signature: func(context.Context, T) (string, error)
	if ft.NumIn() != 2 {
		return fmt.Errorf("tools: Register %q: function must take 2 arguments (context.Context, struct), got %d", name, ft.NumIn())
	}
	if !ft.In(0).Implements(reflect.TypeOf((*context.Context)(nil)).Elem()) {
		return fmt.Errorf("tools: Register %q: first argument must be context.Context", name)
	}
	argType := ft.In(1)
	if argType.Kind() == reflect.Ptr {
		argType = argType.Elem()
	}
	if argType.Kind() != reflect.Struct {
		return fmt.Errorf("tools: Register %q: second argument must be a struct, got %s", name, argType.Kind())
	}
	if ft.NumOut() != 2 {
		return fmt.Errorf("tools: Register %q: function must return (string, error), got %d return values", name, ft.NumOut())
	}
	if ft.Out(0).Kind() != reflect.String {
		return fmt.Errorf("tools: Register %q: first return value must be string", name)
	}
	if !ft.Out(1).Implements(reflect.TypeOf((*error)(nil)).Elem()) {
		return fmt.Errorf("tools: Register %q: second return value must be error", name)
	}

	// Generate schema from the argument struct.
	schema := schemaForType(argType)

	def := Definition{
		Name:        name,
		Description: description,
		Parameters:  schema,
	}

	fv := reflect.ValueOf(fn)
	tool := &funcTool{
		def:     def,
		fn:      fv,
		argType: ft.In(1), // preserve original (may be pointer)
	}

	r.Add(tool)
	return nil
}

// Add adds a pre-constructed Tool directly.
func (r *Registry) Add(tool Tool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tools[tool.Definition().Name] = tool
}

// Definitions returns all tool definitions for sending to the LLM.
func (r *Registry) Definitions() []Definition {
	r.mu.RLock()
	defer r.mu.RUnlock()
	defs := make([]Definition, 0, len(r.tools))
	for _, t := range r.tools {
		defs = append(defs, t.Definition())
	}
	return defs
}

// Dispatch calls the named tool with the given JSON arguments string.
func (r *Registry) Dispatch(ctx context.Context, name, arguments string) (string, error) {
	r.mu.RLock()
	tool, ok := r.tools[name]
	r.mu.RUnlock()
	if !ok {
		return "", fmt.Errorf("tools: unknown tool %q", name)
	}
	return tool.Call(ctx, arguments)
}

// funcTool wraps a Go function as a Tool.
type funcTool struct {
	def     Definition
	fn      reflect.Value
	argType reflect.Type
}

func (t *funcTool) Definition() Definition {
	return t.def
}

func (t *funcTool) Call(ctx context.Context, arguments string) (string, error) {
	// Create a new instance of the argument type.
	isPtr := t.argType.Kind() == reflect.Ptr
	var argVal reflect.Value
	if isPtr {
		argVal = reflect.New(t.argType.Elem())
	} else {
		argVal = reflect.New(t.argType)
	}

	if err := json.Unmarshal([]byte(arguments), argVal.Interface()); err != nil {
		return "", fmt.Errorf("tools: unmarshal arguments for %q: %w", t.def.Name, err)
	}

	var callArg reflect.Value
	if isPtr {
		callArg = argVal
	} else {
		callArg = argVal.Elem()
	}

	results := t.fn.Call([]reflect.Value{reflect.ValueOf(ctx), callArg})

	// Extract return values.
	result := results[0].String()
	var err error
	if !results[1].IsNil() {
		err = results[1].Interface().(error)
	}
	return result, err
}
