package tui

import (
	"regexp"
	"strings"

	"github.com/stack-bound/stackllm/conversation"
)

// pendingImage holds the raw bytes and detected MIME type of an image
// that has been pasted into the textarea but not yet sent. The Model
// keeps a map of these keyed by monotonic index (the number in the
// `[Image #N]` placeholder shown inline in the textarea).
type pendingImage struct {
	mime string
	data []byte
}

// placeholderRE matches the `[Image #N]` markers that mark a pending
// image in the textarea value. The capture group is the decimal index.
var placeholderRE = regexp.MustCompile(`\[Image #(\d+)\]`)

// parseInputBlocks turns the raw textarea value and the current
// pending-image map into an ordered slice of conversation.Block values
// suitable for a user message.
//
// Semantics:
//   - For every `[Image #N]` placeholder whose N is in pending, the
//     preceding (trimmed) text is emitted as a BlockText (skipped if
//     empty), then a BlockImage is emitted with the pending bytes.
//   - Adjacent placeholders emit consecutive BlockImage blocks with no
//     empty BlockText separator.
//   - Duplicate indices (`[Image #1] vs [Image #1]`) emit two BlockImage
//     blocks that reference the same bytes — documented behaviour.
//   - Orphaned placeholders (N not in pending) are kept verbatim as
//     literal text so the user's message is not silently mutilated.
//   - An input with no matches at all returns a single BlockText, which
//     matches the shape the TUI produced before image paste existed.
func parseInputBlocks(input string, pending map[int]pendingImage) []conversation.Block {
	matches := placeholderRE.FindAllStringSubmatchIndex(input, -1)
	if len(matches) == 0 {
		return []conversation.Block{{Type: conversation.BlockText, Text: input}}
	}

	var blocks []conversation.Block
	cursor := 0
	textBuf := strings.Builder{}

	flushText := func() {
		if textBuf.Len() == 0 {
			return
		}
		s := strings.TrimSpace(textBuf.String())
		textBuf.Reset()
		if s == "" {
			return
		}
		blocks = append(blocks, conversation.Block{Type: conversation.BlockText, Text: s})
	}

	for _, m := range matches {
		start, end := m[0], m[1]
		idxStart, idxEnd := m[2], m[3]

		// Append intervening text (between the previous cursor and this match)
		// into the buffer.
		textBuf.WriteString(input[cursor:start])

		idx := 0
		for i := idxStart; i < idxEnd; i++ {
			idx = idx*10 + int(input[i]-'0')
		}

		img, ok := pending[idx]
		if !ok {
			// Orphan — keep the placeholder as literal text.
			textBuf.WriteString(input[start:end])
			cursor = end
			continue
		}

		// Real placeholder — flush any accumulated text, then emit the image.
		flushText()
		blocks = append(blocks, conversation.Block{
			Type:      conversation.BlockImage,
			MimeType:  img.mime,
			ImageData: img.data,
		})
		cursor = end
	}

	// Trailing text after the last match.
	textBuf.WriteString(input[cursor:])
	flushText()

	if len(blocks) == 0 {
		// Only whitespace + orphan placeholders that stripped to nothing.
		// Return an empty text block so the caller still has something
		// to send — parity with the no-match branch above.
		return []conversation.Block{{Type: conversation.BlockText, Text: ""}}
	}

	return blocks
}
