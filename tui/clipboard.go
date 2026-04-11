package tui

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// maxClipboardImageBytes caps how much data we will read from a single
// clipboard paste. Anything larger than this becomes an error and the
// TUI falls back to the default text paste behaviour.
const maxClipboardImageBytes = 20 * 1024 * 1024

// clipboardReadTimeout bounds how long a clipboard shell-out may take.
// PowerShell cold-start has been observed up to ~800 ms on Windows; we
// give every platform 2 s to keep the UX consistent.
const clipboardReadTimeout = 2 * time.Second

// errNoImage is returned by a ClipboardReader when the clipboard does
// not currently hold an image. Callers use this as the signal to fall
// back to the default text paste behaviour.
var errNoImage = errors.New("clipboard: no image")

// ClipboardReader reads the current system clipboard and returns the
// raw image bytes plus their detected MIME type. Implementations must
// respect the passed context for cancellation and must return errNoImage
// (not a generic error) when the clipboard contains no image, so the
// Model can distinguish "no image, fall through to text paste" from
// "real failure".
type ClipboardReader func(ctx context.Context) (data []byte, mime string, err error)

// WithClipboardReader injects a ClipboardReader, overriding the default
// platform shell-out. Tests use this to feed canned bytes; real
// embedders rarely need it.
func WithClipboardReader(fn ClipboardReader) Option {
	return func(m *Model) { m.clipboardReader = fn }
}

// clipboardImageMsg is delivered after an async clipboard read. Exactly
// one of (data + mime) or err is populated.
type clipboardImageMsg struct {
	data []byte
	mime string
	err  error
}

// readClipboardImageCmd launches the configured ClipboardReader inside
// a tea.Cmd so the Update loop is never blocked waiting on
// powershell.exe / wl-paste / xclip / pngpaste. The command applies a
// 2s timeout and returns a clipboardImageMsg regardless of outcome.
func (m *Model) readClipboardImageCmd() tea.Cmd {
	reader := m.clipboardReader
	if reader == nil {
		reader = defaultClipboardReader
	}
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), clipboardReadTimeout)
		defer cancel()
		data, mime, err := reader(ctx)
		return clipboardImageMsg{data: data, mime: mime, err: err}
	}
}

// sniffImageMIME inspects the leading magic bytes of a payload and
// returns the corresponding MIME type, or the empty string if the
// format is unknown. We do this ourselves rather than trust the
// platform tool because xclip/wl-paste/pngpaste can all be asked for
// PNG but may still hand back other formats in edge cases.
func sniffImageMIME(data []byte) string {
	switch {
	case len(data) >= 8 && bytes.Equal(data[:8], []byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A}):
		return "image/png"
	case len(data) >= 3 && bytes.Equal(data[:3], []byte{0xFF, 0xD8, 0xFF}):
		return "image/jpeg"
	case len(data) >= 6 && (bytes.Equal(data[:6], []byte("GIF87a")) || bytes.Equal(data[:6], []byte("GIF89a"))):
		return "image/gif"
	case len(data) >= 12 && bytes.Equal(data[:4], []byte("RIFF")) && bytes.Equal(data[8:12], []byte("WEBP")):
		return "image/webp"
	case len(data) >= 2 && bytes.Equal(data[:2], []byte("BM")):
		return "image/bmp"
	}
	return ""
}

// defaultClipboardReader is the platform-default ClipboardReader. It
// shells out to `wl-paste`, `xclip`, `pngpaste`, `osascript`, or
// `powershell.exe` depending on runtime.GOOS, enforces a size cap via
// io.LimitReader, sniffs the MIME type from magic bytes, and returns
// errNoImage whenever the clipboard is empty, holds non-image data, or
// the required tool is missing.
func defaultClipboardReader(ctx context.Context) ([]byte, string, error) {
	switch runtime.GOOS {
	case "linux":
		return readClipboardLinux(ctx)
	case "darwin":
		return readClipboardDarwin(ctx)
	case "windows":
		return readClipboardWindows(ctx)
	default:
		return nil, "", errNoImage
	}
}

func readClipboardLinux(ctx context.Context) ([]byte, string, error) {
	if _, err := exec.LookPath("wl-paste"); err == nil {
		data, err := runClipboardCmd(ctx, "wl-paste", "--no-newline", "--type", "image/png")
		if err == nil && len(data) > 0 {
			if out, mime, finishErr := finishClipboardBytes(data); finishErr == nil {
				return out, mime, nil
			}
		}
	}
	if _, err := exec.LookPath("xclip"); err == nil {
		data, err := runClipboardCmd(ctx, "xclip", "-selection", "clipboard", "-target", "image/png", "-out")
		if err == nil && len(data) > 0 {
			if out, mime, finishErr := finishClipboardBytes(data); finishErr == nil {
				return out, mime, nil
			}
		}
	}
	return nil, "", errNoImage
}

func readClipboardDarwin(ctx context.Context) ([]byte, string, error) {
	if _, err := exec.LookPath("pngpaste"); err == nil {
		data, err := runClipboardCmd(ctx, "pngpaste", "-")
		if err == nil && len(data) > 0 {
			if out, mime, finishErr := finishClipboardBytes(data); finishErr == nil {
				return out, mime, nil
			}
		}
	}
	if _, err := exec.LookPath("osascript"); err == nil {
		tmp, err := os.CreateTemp("", "stackllm-clip-*.png")
		if err != nil {
			return nil, "", errNoImage
		}
		path := tmp.Name()
		tmp.Close()
		defer os.Remove(path)

		script := fmt.Sprintf(`try
	set pngData to the clipboard as «class PNGf»
	set fh to open for access POSIX file %q with write permission
	set eof of fh to 0
	write pngData to fh
	close access fh
on error
	return "NO_IMAGE"
end try`, path)
		out, err := runClipboardCmd(ctx, "osascript", "-e", script)
		if err != nil {
			return nil, "", errNoImage
		}
		if bytes.Contains(out, []byte("NO_IMAGE")) {
			return nil, "", errNoImage
		}
		data, err := os.ReadFile(path)
		if err != nil || len(data) == 0 {
			return nil, "", errNoImage
		}
		return finishClipboardBytes(data)
	}
	return nil, "", errNoImage
}

func readClipboardWindows(ctx context.Context) ([]byte, string, error) {
	if _, err := exec.LookPath("powershell.exe"); err != nil {
		return nil, "", errNoImage
	}
	script := `$ErrorActionPreference='Stop';` +
		`Add-Type -AssemblyName System.Windows.Forms;` +
		`Add-Type -AssemblyName System.Drawing;` +
		`$img=[System.Windows.Forms.Clipboard]::GetImage();` +
		`if ($img -eq $null) { exit 2 }` +
		`$ms=New-Object System.IO.MemoryStream;` +
		`$img.Save($ms,[System.Drawing.Imaging.ImageFormat]::Png);` +
		`$stdout=[Console]::OpenStandardOutput();` +
		`$bytes=$ms.ToArray();` +
		`$stdout.Write($bytes,0,$bytes.Length);` +
		`$stdout.Flush();`
	data, err := runClipboardCmd(ctx, "powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script)
	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) && exitErr.ExitCode() == 2 {
			return nil, "", errNoImage
		}
		return nil, "", errNoImage
	}
	if len(data) == 0 {
		return nil, "", errNoImage
	}
	return finishClipboardBytes(data)
}

// runClipboardCmd runs a shell-out and returns its stdout, capped at
// maxClipboardImageBytes+1 so we can detect runaway producers. Stderr
// is discarded — these are read-only probes and the caller only cares
// about success/failure and the byte payload. After the read limit is
// hit, any remaining stdout is drained into io.Discard so cmd.Wait()
// doesn't block on a full pipe; the context's 2-second deadline will
// kill the child if it refuses to exit.
func runClipboardCmd(ctx context.Context, name string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	cmd.Stderr = io.Discard
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	limited := io.LimitReader(stdout, maxClipboardImageBytes+1)
	data, readErr := io.ReadAll(limited)
	// Drain anything past the limit so the child's stdout buffer
	// can flush and Wait() doesn't deadlock on a full pipe.
	_, _ = io.Copy(io.Discard, stdout)
	waitErr := cmd.Wait()
	if readErr != nil {
		return nil, readErr
	}
	if waitErr != nil {
		return nil, waitErr
	}
	return data, nil
}

// finishClipboardBytes enforces the size cap and sniffs the MIME. It
// is the single exit point for every platform branch so the size and
// MIME guarantees hold uniformly.
func finishClipboardBytes(data []byte) ([]byte, string, error) {
	if len(data) > maxClipboardImageBytes {
		return nil, "", errors.New("clipboard: image exceeds size cap")
	}
	mime := sniffImageMIME(data)
	if mime == "" {
		return nil, "", errNoImage
	}
	return data, mime, nil
}
