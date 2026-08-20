package tui

import (
	"testing"
)

func TestFilterCommands(t *testing.T) {
	t.Parallel()

	allNames := make([]string, len(commands))
	for i, c := range commands {
		allNames[i] = c.Name
	}

	tests := []struct {
		name  string
		query string
		want  []string
	}{
		{"empty returns all in registry order", "", allNames},
		{"bare slash returns all", "/", allNames},
		{"whitespace-only returns all", "   ", allNames},
		{"prefix match", "/mo", []string{"/models"}},
		{"case-insensitive", "/MODELS", []string{"/models"}},
		{"substring without slash", "ses", []string{"/sessions"}},
		{"substring mid-word", "ort", []string{"/export"}},
		{"no match", "/zzz", nil},
		{"slash-anchored substring", "/e", []string{"/export"}},
		{"multi match", "e", []string{"/help", "/sessions", "/new", "/rename", "/delete", "/export", "/models"}},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := filterCommands(tc.query)
			if len(got) != len(tc.want) {
				t.Fatalf("filterCommands(%q) returned %d commands, want %d: %+v", tc.query, len(got), len(tc.want), got)
			}
			for i, c := range got {
				if c.Name != tc.want[i] {
					t.Errorf("filterCommands(%q)[%d] = %q, want %q", tc.query, i, c.Name, tc.want[i])
				}
			}
		})
	}
}

// TestFilterCommands_ReturnsCopy verifies the empty-query fast path
// hands back a copy, so callers mutating the result cannot corrupt the
// canonical registry.
func TestFilterCommands_ReturnsCopy(t *testing.T) {
	t.Parallel()

	got := filterCommands("")
	if len(got) == 0 {
		t.Fatal("expected non-empty command list")
	}
	original := commands[0].Name
	got[0].Name = "/mutated"
	if commands[0].Name != original {
		t.Errorf("mutating filterCommands result leaked into registry: %q", commands[0].Name)
	}
}
