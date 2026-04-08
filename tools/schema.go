package tools

import (
	"reflect"
	"strings"
)

// SchemaOf returns a JSON Schema map for the given value's type.
// Supports struct tags:
//   - json:"name"         — field name in schema
//   - jsonschema:"description=..." — field description
//   - jsonschema:"required"        — mark field as required
//   - jsonschema:"enum=a,b,c"      — enum values
func SchemaOf(v any) map[string]any {
	t := reflect.TypeOf(v)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	return schemaForType(t)
}

func schemaForType(t reflect.Type) map[string]any {
	if t.Kind() == reflect.Ptr {
		return schemaForType(t.Elem())
	}

	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return map[string]any{"type": "number"}
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "number"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.Slice, reflect.Array:
		return map[string]any{
			"type":  "array",
			"items": schemaForType(t.Elem()),
		}
	case reflect.Struct:
		return schemaForStruct(t)
	default:
		return map[string]any{"type": "string"}
	}
}

func schemaForStruct(t reflect.Type) map[string]any {
	properties := make(map[string]any)
	var required []string

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}

		name := field.Name
		if jsonTag := field.Tag.Get("json"); jsonTag != "" {
			parts := strings.Split(jsonTag, ",")
			if parts[0] == "-" {
				continue
			}
			if parts[0] != "" {
				name = parts[0]
			}
		}

		prop := schemaForType(field.Type)

		// Parse jsonschema tag.
		if jsTag := field.Tag.Get("jsonschema"); jsTag != "" {
			parts := strings.Split(jsTag, ",")
			for _, part := range parts {
				part = strings.TrimSpace(part)
				if part == "required" {
					required = append(required, name)
				} else if strings.HasPrefix(part, "description=") {
					prop["description"] = strings.TrimPrefix(part, "description=")
				} else if strings.HasPrefix(part, "enum=") {
					enumStr := strings.TrimPrefix(part, "enum=")
					enumVals := strings.Split(enumStr, "|")
					anyVals := make([]any, len(enumVals))
					for i, v := range enumVals {
						anyVals[i] = v
					}
					prop["enum"] = anyVals
				}
			}
		}

		properties[name] = prop
	}

	schema := map[string]any{
		"type":       "object",
		"properties": properties,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema
}
