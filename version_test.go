package stackllm

import (
	"os"
	"strings"
	"testing"
)

func TestVersion(t *testing.T) {
	v := Version()
	if v == "" {
		t.Fatal("Version() returned empty string")
	}

	raw, err := os.ReadFile("VERSION")
	if err != nil {
		t.Fatalf("reading VERSION file: %v", err)
	}
	want := strings.TrimSpace(string(raw))
	if v != want {
		t.Fatalf("Version() = %q, want %q (from VERSION file)", v, want)
	}
}
