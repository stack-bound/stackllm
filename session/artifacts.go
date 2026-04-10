package session

import (
	"crypto/sha256"
	"encoding/hex"
	"unicode/utf8"
)

// sha256Hex returns the lowercase-hex SHA-256 digest of b. Used both
// as the content hash for artifact dedupe and as a stable identifier
// embedded in the search path.
func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// extractPreview returns the first artifactPreviewBytes of s, truncated
// at a safe UTF-8 boundary. If s contains a newline inside the
// threshold, truncation is pulled back to the last newline so that
// scrollback never shows half a line. If the string is shorter than
// the threshold it is returned unchanged.
//
// The preview is what goes into stackllm_blocks.text_content for
// artifact-backed tool_result blocks; it is also what feeds FTS for
// those blocks. Callers that want to search the full artifact payload
// must hydrate it explicitly.
func extractPreview(s string) string {
	if len(s) <= artifactPreviewBytes {
		return s
	}

	cut := artifactPreviewBytes

	// Pull back to a UTF-8 rune boundary. utf8.RuneStart is true at
	// the first byte of a rune, so we walk back at most 3 bytes.
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}

	// Prefer a newline boundary if one exists reasonably close to the
	// cut. This keeps previews whole-line for readability. Don't pull
	// back further than half the preview — better to show more text
	// than to aggressively trim.
	minNewline := cut / 2
	for i := cut - 1; i >= minNewline; i-- {
		if s[i] == '\n' {
			return s[:i+1]
		}
	}

	return s[:cut]
}

// shouldOffloadText reports whether a text payload of the given length
// should be written to stackllm_artifacts instead of inline on the
// block row. The threshold is a library constant; exposing it through
// SQLiteConfig is a trivial follow-up.
func shouldOffloadText(n int) bool {
	return n > defaultArtifactThreshold
}
