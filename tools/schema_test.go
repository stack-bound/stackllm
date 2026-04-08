package tools

import (
	"testing"
)

func TestSchemaOf_String(t *testing.T) {
	t.Parallel()
	s := SchemaOf("")
	if s["type"] != "string" {
		t.Errorf("type = %v, want string", s["type"])
	}
}

func TestSchemaOf_Number(t *testing.T) {
	t.Parallel()

	for _, v := range []any{0, int64(0), float64(0)} {
		s := SchemaOf(v)
		if s["type"] != "number" {
			t.Errorf("type for %T = %v, want number", v, s["type"])
		}
	}
}

func TestSchemaOf_Bool(t *testing.T) {
	t.Parallel()
	s := SchemaOf(false)
	if s["type"] != "boolean" {
		t.Errorf("type = %v, want boolean", s["type"])
	}
}

func TestSchemaOf_Slice(t *testing.T) {
	t.Parallel()
	s := SchemaOf([]string{})
	if s["type"] != "array" {
		t.Errorf("type = %v, want array", s["type"])
	}
	items, ok := s["items"].(map[string]any)
	if !ok {
		t.Fatal("items not a map")
	}
	if items["type"] != "string" {
		t.Errorf("items type = %v, want string", items["type"])
	}
}

func TestSchemaOf_Struct(t *testing.T) {
	t.Parallel()

	type Args struct {
		Path    string `json:"path" jsonschema:"description=File path,required"`
		Lines   int    `json:"lines"`
		Verbose bool   `json:"verbose"`
	}

	s := SchemaOf(Args{})
	if s["type"] != "object" {
		t.Fatalf("type = %v, want object", s["type"])
	}

	props, ok := s["properties"].(map[string]any)
	if !ok {
		t.Fatal("properties not a map")
	}

	// Check path field.
	pathProp, ok := props["path"].(map[string]any)
	if !ok {
		t.Fatal("path property not found")
	}
	if pathProp["type"] != "string" {
		t.Errorf("path type = %v, want string", pathProp["type"])
	}
	if pathProp["description"] != "File path" {
		t.Errorf("path description = %v, want 'File path'", pathProp["description"])
	}

	// Check required.
	req, ok := s["required"].([]string)
	if !ok {
		t.Fatal("required not a string slice")
	}
	if len(req) != 1 || req[0] != "path" {
		t.Errorf("required = %v, want [path]", req)
	}

	// Check other fields exist.
	if _, ok := props["lines"]; !ok {
		t.Error("lines property missing")
	}
	if _, ok := props["verbose"]; !ok {
		t.Error("verbose property missing")
	}
}

func TestSchemaOf_NestedStruct(t *testing.T) {
	t.Parallel()

	type Inner struct {
		Value string `json:"value"`
	}
	type Outer struct {
		Inner Inner `json:"inner"`
	}

	s := SchemaOf(Outer{})
	props := s["properties"].(map[string]any)
	innerProp := props["inner"].(map[string]any)
	if innerProp["type"] != "object" {
		t.Errorf("inner type = %v, want object", innerProp["type"])
	}
	innerProps := innerProp["properties"].(map[string]any)
	if _, ok := innerProps["value"]; !ok {
		t.Error("inner.value property missing")
	}
}

func TestSchemaOf_Pointer(t *testing.T) {
	t.Parallel()

	type Args struct {
		Name *string `json:"name"`
	}

	s := SchemaOf(Args{})
	props := s["properties"].(map[string]any)
	nameProp := props["name"].(map[string]any)
	if nameProp["type"] != "string" {
		t.Errorf("pointer string type = %v, want string", nameProp["type"])
	}
}

func TestSchemaOf_Enum(t *testing.T) {
	t.Parallel()

	type Args struct {
		Mode string `json:"mode" jsonschema:"enum=fast|slow|auto"`
	}

	s := SchemaOf(Args{})
	props := s["properties"].(map[string]any)
	modeProp := props["mode"].(map[string]any)
	enumVals, ok := modeProp["enum"].([]any)
	if !ok {
		t.Fatal("enum not found")
	}
	if len(enumVals) != 3 {
		t.Fatalf("enum length = %d, want 3", len(enumVals))
	}
	if enumVals[0] != "fast" || enumVals[1] != "slow" || enumVals[2] != "auto" {
		t.Errorf("enum = %v, want [fast slow auto]", enumVals)
	}
}

func TestSchemaOf_JSONTagDash(t *testing.T) {
	t.Parallel()

	type Args struct {
		Public  string `json:"public"`
		Private string `json:"-"`
	}

	s := SchemaOf(Args{})
	props := s["properties"].(map[string]any)
	if _, ok := props["Private"]; ok {
		t.Error("json:\"-\" field should be excluded")
	}
	if _, ok := props["public"]; !ok {
		t.Error("public field should be included")
	}
}

func TestSchemaOf_UnexportedFields(t *testing.T) {
	t.Parallel()

	type Args struct {
		Public  string `json:"public"`
		private string //nolint:unused
	}

	s := SchemaOf(Args{})
	props := s["properties"].(map[string]any)
	if len(props) != 1 {
		t.Errorf("expected 1 property, got %d", len(props))
	}
}
