package tui

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

func TestFinishClipboardBytes(t *testing.T) {
	t.Parallel()

	oversized := make([]byte, maxClipboardImageBytes+1)
	copy(oversized, cannedPNG)

	tests := []struct {
		name     string
		in       []byte
		wantMime string
		wantErr  error // nil means success; errNoImage checked by identity
		wantCap  bool  // true when the size-cap error is expected
	}{
		{"valid png", cannedPNG, "image/png", nil, false},
		{"valid jpeg", []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x01}, "image/jpeg", nil, false},
		{"garbage is no image", []byte("plain text on the clipboard"), "", errNoImage, false},
		{"empty is no image", nil, "", errNoImage, false},
		{"oversized rejected", oversized, "", nil, true},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			data, mime, err := finishClipboardBytes(tc.in)
			if tc.wantCap {
				if err == nil || errors.Is(err, errNoImage) {
					t.Fatalf("expected size-cap error, got %v", err)
				}
				return
			}
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("expected %v, got %v", tc.wantErr, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if mime != tc.wantMime {
				t.Errorf("mime = %q, want %q", mime, tc.wantMime)
			}
			if !bytes.Equal(data, tc.in) {
				t.Errorf("data mutated: got %v, want %v", data, tc.in)
			}
		})
	}
}

// writeStubTool writes an executable shell script named `name` into dir
// whose body is the given script (sans shebang).
func writeStubTool(t *testing.T, dir, name, script string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	content := "#!/bin/sh\n" + script + "\n"
	if err := os.WriteFile(path, []byte(content), 0o755); err != nil {
		t.Fatalf("write stub %s: %v", name, err)
	}
	return path
}

func TestRunClipboardCmd(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell-script stubs require a POSIX shell")
	}
	t.Parallel()

	dir := t.TempDir()
	catPath, err := exec.LookPath("cat")
	if err != nil {
		t.Skipf("cat not found: %v", err)
	}
	payload := filepath.Join(dir, "payload.bin")
	if err := os.WriteFile(payload, cannedPNG, 0o644); err != nil {
		t.Fatalf("write payload: %v", err)
	}

	// Write every stub before any parallel subtest forks: a WriteFile
	// racing a sibling's fork+exec leaks the open write fd into the
	// child and the exec fails with ETXTBSY.
	emitStub := writeStubTool(t, dir, "emit-png", fmt.Sprintf("exec %q %q", catPath, payload))
	failStub := writeStubTool(t, dir, "fail-tool", "exit 3")

	t.Run("stdout captured", func(t *testing.T) {
		t.Parallel()
		data, err := runClipboardCmd(t.Context(), emitStub)
		if err != nil {
			t.Fatalf("runClipboardCmd: %v", err)
		}
		if !bytes.Equal(data, cannedPNG) {
			t.Errorf("stdout bytes = %v, want %v", data, cannedPNG)
		}
	})

	t.Run("nonzero exit is an error", func(t *testing.T) {
		t.Parallel()
		if _, err := runClipboardCmd(t.Context(), failStub); err == nil {
			t.Error("expected error for exit status 3, got nil")
		}
	})

	t.Run("missing binary is an error", func(t *testing.T) {
		t.Parallel()
		if _, err := runClipboardCmd(t.Context(), filepath.Join(dir, "does-not-exist")); err == nil {
			t.Error("expected error for missing binary, got nil")
		}
	})
}

// TestReadClipboardLinux exercises the Linux shell-out chain by
// installing stub wl-paste / xclip executables on a private PATH.
// t.Setenv is incompatible with t.Parallel, so this test runs serially.
func TestReadClipboardLinux(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("readClipboardLinux is Linux-only")
	}

	catPath, err := exec.LookPath("cat")
	if err != nil {
		t.Skipf("cat not found: %v", err)
	}

	writePayload := func(t *testing.T, dir string, data []byte) string {
		t.Helper()
		p := filepath.Join(dir, "payload.bin")
		if err := os.WriteFile(p, data, 0o644); err != nil {
			t.Fatalf("write payload: %v", err)
		}
		return p
	}

	t.Run("wl-paste preferred", func(t *testing.T) {
		dir := t.TempDir()
		payload := writePayload(t, dir, cannedPNG)
		writeStubTool(t, dir, "wl-paste", fmt.Sprintf("exec %q %q", catPath, payload))
		t.Setenv("PATH", dir)

		data, mime, err := readClipboardLinux(t.Context())
		if err != nil {
			t.Fatalf("readClipboardLinux: %v", err)
		}
		if mime != "image/png" {
			t.Errorf("mime = %q, want image/png", mime)
		}
		if !bytes.Equal(data, cannedPNG) {
			t.Errorf("data = %v, want %v", data, cannedPNG)
		}
	})

	t.Run("xclip fallback when wl-paste absent", func(t *testing.T) {
		dir := t.TempDir()
		payload := writePayload(t, dir, cannedPNG)
		writeStubTool(t, dir, "xclip", fmt.Sprintf("exec %q %q", catPath, payload))
		t.Setenv("PATH", dir)

		data, mime, err := readClipboardLinux(t.Context())
		if err != nil {
			t.Fatalf("readClipboardLinux: %v", err)
		}
		if mime != "image/png" {
			t.Errorf("mime = %q, want image/png", mime)
		}
		if !bytes.Equal(data, cannedPNG) {
			t.Errorf("data = %v, want %v", data, cannedPNG)
		}
	})

	t.Run("wl-paste non-image falls through to xclip", func(t *testing.T) {
		dir := t.TempDir()
		payload := writePayload(t, dir, cannedPNG)
		// wl-paste emits text (not an image); xclip has the real PNG.
		writeStubTool(t, dir, "wl-paste", "printf 'just some text'")
		writeStubTool(t, dir, "xclip", fmt.Sprintf("exec %q %q", catPath, payload))
		t.Setenv("PATH", dir)

		data, mime, err := readClipboardLinux(t.Context())
		if err != nil {
			t.Fatalf("readClipboardLinux: %v", err)
		}
		if mime != "image/png" {
			t.Errorf("mime = %q, want image/png", mime)
		}
		if !bytes.Equal(data, cannedPNG) {
			t.Errorf("data = %v, want %v", data, cannedPNG)
		}
	})

	t.Run("failing tools yield errNoImage", func(t *testing.T) {
		dir := t.TempDir()
		writeStubTool(t, dir, "wl-paste", "exit 1")
		writeStubTool(t, dir, "xclip", "exit 1")
		t.Setenv("PATH", dir)

		_, _, err := readClipboardLinux(t.Context())
		if !errors.Is(err, errNoImage) {
			t.Errorf("expected errNoImage, got %v", err)
		}
	})

	t.Run("no tools yields errNoImage", func(t *testing.T) {
		dir := t.TempDir() // empty: no executables at all
		t.Setenv("PATH", dir)

		_, _, err := readClipboardLinux(t.Context())
		if !errors.Is(err, errNoImage) {
			t.Errorf("expected errNoImage, got %v", err)
		}
	})
}

// TestDefaultClipboardReader_Linux verifies the GOOS dispatch routes to
// the Linux reader (observable because the stubbed wl-paste PNG comes
// back). Darwin/Windows branches are unreachable on this platform and
// are deliberately not faked.
func TestDefaultClipboardReader_Linux(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("dispatch test requires linux")
	}

	dir := t.TempDir()
	catPath, err := exec.LookPath("cat")
	if err != nil {
		t.Skipf("cat not found: %v", err)
	}
	payload := filepath.Join(dir, "payload.bin")
	if err := os.WriteFile(payload, cannedPNG, 0o644); err != nil {
		t.Fatalf("write payload: %v", err)
	}
	writeStubTool(t, dir, "wl-paste", fmt.Sprintf("exec %q %q", catPath, payload))
	t.Setenv("PATH", dir)

	data, mime, err := defaultClipboardReader(t.Context())
	if err != nil {
		t.Fatalf("defaultClipboardReader: %v", err)
	}
	if mime != "image/png" {
		t.Errorf("mime = %q, want image/png", mime)
	}
	if !bytes.Equal(data, cannedPNG) {
		t.Errorf("data = %v, want %v", data, cannedPNG)
	}
}

// TestReadClipboardImageCmd_NilReaderUsesDefault verifies that a Model
// whose clipboardReader was explicitly nilled falls back to the
// platform default (which, with an empty PATH, reports errNoImage).
func TestReadClipboardImageCmd_NilReaderUsesDefault(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("default-reader fallback test requires linux")
	}
	t.Setenv("PATH", t.TempDir())

	m := newTestModel(t)
	m.clipboardReader = nil
	msg := m.readClipboardImageCmd()()
	cimg, ok := msg.(clipboardImageMsg)
	if !ok {
		t.Fatalf("expected clipboardImageMsg, got %T", msg)
	}
	if !errors.Is(cimg.err, errNoImage) {
		t.Errorf("expected errNoImage from default reader, got %v", cimg.err)
	}
}
