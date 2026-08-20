package provider

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stack-bound/stackllm/auth"
	"github.com/stack-bound/stackllm/conversation"
)

// failingTokenSource always fails to produce a token.
type failingTokenSource struct{}

func (failingTokenSource) Token(context.Context) (*auth.Token, error) {
	return nil, fmt.Errorf("token source down")
}

func TestAuthRoundTripper_TokenSourceError(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("request must not reach the backend when the token source fails")
	}))
	t.Cleanup(srv.Close)

	p := New(Config{
		BaseURL:     srv.URL,
		TokenSource: failingTokenSource{},
		Model:       "m",
		MaxRetries:  1,
		BaseBackoff: time.Millisecond,
	})
	events, err := p.Complete(context.Background(), Request{Messages: []conversation.Message{userText("hi")}, Stream: true})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var gotErr error
	for ev := range events {
		if ev.Type == EventTypeError {
			gotErr = ev.Err
		}
	}
	if gotErr == nil || !strings.Contains(gotErr.Error(), "token source down") {
		t.Errorf("error = %v, want token source failure", gotErr)
	}
}

func TestDoStreamingPOST_ContextCancelledDuringBackoff(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "overloaded", http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)

	p := New(Config{
		BaseURL:     srv.URL,
		TokenSource: auth.NewStatic("k"),
		Model:       "m",
		MaxRetries:  3,
		BaseBackoff: 10 * time.Second, // long enough that cancellation wins
	})

	ctx, cancel := context.WithCancel(context.Background())
	events, err := p.Complete(ctx, Request{Messages: []conversation.Message{userText("hi")}, Stream: true})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	// Cancel while the retry loop is sleeping in its backoff.
	time.AfterFunc(50*time.Millisecond, cancel)

	var gotErr error
	deadline := time.After(5 * time.Second)
	for {
		select {
		case ev, ok := <-events:
			if !ok {
				if gotErr == nil || gotErr != context.Canceled {
					t.Errorf("error = %v, want context.Canceled from backoff wait", gotErr)
				}
				return
			}
			if ev.Type == EventTypeError {
				gotErr = ev.Err
			}
		case <-deadline:
			t.Fatal("stream did not terminate after cancellation")
		}
	}
}
