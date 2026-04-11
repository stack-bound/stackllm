package tui

import (
	"testing"
)

func TestSniffImageMIME(t *testing.T) {
	t.Parallel()

	pngHead := []byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00}
	jpegHead := []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00}
	gif87 := []byte("GIF87athis is a gif")
	gif89 := []byte("GIF89athis is also a gif")
	webp := append([]byte("RIFF\x00\x00\x00\x00WEBP"), 0x00, 0x00)
	bmp := []byte{'B', 'M', 0x00, 0x00}
	empty := []byte{}
	garbage := []byte("not an image")

	tests := []struct {
		name string
		in   []byte
		want string
	}{
		{"png", pngHead, "image/png"},
		{"jpeg", jpegHead, "image/jpeg"},
		{"gif87", gif87, "image/gif"},
		{"gif89", gif89, "image/gif"},
		{"webp", webp, "image/webp"},
		{"bmp", bmp, "image/bmp"},
		{"empty", empty, ""},
		{"garbage", garbage, ""},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := sniffImageMIME(tc.in)
			if got != tc.want {
				t.Errorf("sniffImageMIME(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}
