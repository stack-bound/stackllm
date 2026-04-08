package stackllm

import (
	_ "embed"
	"strings"
)

//go:embed VERSION
var version string

// Version returns the library version.
func Version() string {
	return strings.TrimSpace(version)
}
