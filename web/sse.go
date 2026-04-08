package web

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// sseWriter wraps an http.ResponseWriter for Server-Sent Events.
type sseWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
}

func newSSEWriter(w http.ResponseWriter) (*sseWriter, error) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, fmt.Errorf("web: response writer does not support flushing")
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	return &sseWriter{w: w, flusher: flusher}, nil
}

// writeEvent sends a named SSE event with JSON data.
func (s *sseWriter) writeEvent(event string, data any) error {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("web: marshal SSE data: %w", err)
	}
	fmt.Fprintf(s.w, "event: %s\ndata: %s\n\n", event, jsonData)
	s.flusher.Flush()
	return nil
}
