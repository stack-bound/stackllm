package tui

import (
	"bytes"
	"testing"

	"github.com/stack-bound/stackllm/conversation"
)

func TestParseInputBlocks(t *testing.T) {
	t.Parallel()

	pngA := []byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A, 'A'}
	pngB := []byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A, 'B'}

	type want struct {
		kind conversation.BlockType
		text string
		data []byte
	}

	tests := []struct {
		name    string
		input   string
		pending map[int]pendingImage
		want    []want
	}{
		{
			name:    "no placeholders, plain text",
			input:   "hello world",
			pending: map[int]pendingImage{},
			want: []want{
				{kind: conversation.BlockText, text: "hello world"},
			},
		},
		{
			name:    "single image with leading text",
			input:   "look at [Image #1]",
			pending: map[int]pendingImage{1: {mime: "image/png", data: pngA}},
			want: []want{
				{kind: conversation.BlockText, text: "look at"},
				{kind: conversation.BlockImage, data: pngA},
			},
		},
		{
			name:    "single image interleaved with text on both sides",
			input:   "before [Image #1] after",
			pending: map[int]pendingImage{1: {mime: "image/png", data: pngA}},
			want: []want{
				{kind: conversation.BlockText, text: "before"},
				{kind: conversation.BlockImage, data: pngA},
				{kind: conversation.BlockText, text: "after"},
			},
		},
		{
			name:  "adjacent placeholders emit consecutive image blocks",
			input: "[Image #1][Image #2]",
			pending: map[int]pendingImage{
				1: {mime: "image/png", data: pngA},
				2: {mime: "image/png", data: pngB},
			},
			want: []want{
				{kind: conversation.BlockImage, data: pngA},
				{kind: conversation.BlockImage, data: pngB},
			},
		},
		{
			name:    "duplicate indices emit duplicate image blocks",
			input:   "compare [Image #1] to [Image #1]",
			pending: map[int]pendingImage{1: {mime: "image/png", data: pngA}},
			want: []want{
				{kind: conversation.BlockText, text: "compare"},
				{kind: conversation.BlockImage, data: pngA},
				{kind: conversation.BlockText, text: "to"},
				{kind: conversation.BlockImage, data: pngA},
			},
		},
		{
			name:    "orphan placeholder survives as literal text",
			input:   "stray [Image #9] ref",
			pending: map[int]pendingImage{},
			want: []want{
				{kind: conversation.BlockText, text: "stray [Image #9] ref"},
			},
		},
		{
			name:    "real and orphan mixed",
			input:   "a [Image #1] b [Image #9] c",
			pending: map[int]pendingImage{1: {mime: "image/png", data: pngA}},
			want: []want{
				{kind: conversation.BlockText, text: "a"},
				{kind: conversation.BlockImage, data: pngA},
				{kind: conversation.BlockText, text: "b [Image #9] c"},
			},
		},
		{
			name:    "leading image with no text before",
			input:   "[Image #1] tail",
			pending: map[int]pendingImage{1: {mime: "image/png", data: pngA}},
			want: []want{
				{kind: conversation.BlockImage, data: pngA},
				{kind: conversation.BlockText, text: "tail"},
			},
		},
		{
			name:    "trailing image with no text after",
			input:   "head [Image #1]",
			pending: map[int]pendingImage{1: {mime: "image/png", data: pngA}},
			want: []want{
				{kind: conversation.BlockText, text: "head"},
				{kind: conversation.BlockImage, data: pngA},
			},
		},
		{
			name:  "image only, no text at all",
			input: "[Image #1]",
			pending: map[int]pendingImage{
				1: {mime: "image/png", data: pngA},
			},
			want: []want{
				{kind: conversation.BlockImage, data: pngA},
			},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := parseInputBlocks(tc.input, tc.pending)
			if len(got) != len(tc.want) {
				t.Fatalf("block count: got %d, want %d\n got=%#v", len(got), len(tc.want), got)
			}
			for i, w := range tc.want {
				if got[i].Type != w.kind {
					t.Errorf("block[%d] type: got %q, want %q", i, got[i].Type, w.kind)
				}
				switch w.kind {
				case conversation.BlockText:
					if got[i].Text != w.text {
						t.Errorf("block[%d] text: got %q, want %q", i, got[i].Text, w.text)
					}
				case conversation.BlockImage:
					if !bytes.Equal(got[i].ImageData, w.data) {
						t.Errorf("block[%d] image data mismatch:\n got %v\n want %v", i, got[i].ImageData, w.data)
					}
					if got[i].MimeType == "" {
						t.Errorf("block[%d] mime type should not be empty", i)
					}
				}
			}
		})
	}
}
