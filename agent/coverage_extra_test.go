package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stack-bound/stackllm/auth"
	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/provider"
)

// modeledMockProvider is a mockProvider that also exposes a configured
// model name, matching the optional Model() interface Agent.Model probes.
type modeledMockProvider struct {
	mockProvider
	model string
}

func (m *modeledMockProvider) Model() string { return m.model }

func TestAgent_Model_FallsBackToProviderModel(t *testing.T) {
	t.Parallel()

	// No WithModel: the agent must fall back to the provider's own model.
	a := New(&modeledMockProvider{model: "provider-default"})
	if got := a.Model(); got != "provider-default" {
		t.Errorf("Model() = %q, want provider-default", got)
	}

	// WithModel wins over the provider's model.
	a = New(&modeledMockProvider{model: "provider-default"}, WithModel("override"))
	if got := a.Model(); got != "override" {
		t.Errorf("Model() = %q, want override", got)
	}

	// Provider without a Model() method and no WithModel: empty string.
	a = New(&mockProvider{})
	if got := a.Model(); got != "" {
		t.Errorf("Model() = %q, want empty", got)
	}
}

// TestWithTemperatureAndMaxTokens_ReachWire verifies the options are
// carried through the provider request all the way to the JSON body the
// backend receives — behaviour, not structure.
func TestWithTemperatureAndMaxTokens_ReachWire(t *testing.T) {
	t.Parallel()

	var mu sync.Mutex
	var gotBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		var body map[string]any
		if err := json.Unmarshal(data, &body); err != nil {
			t.Errorf("request body is not JSON: %v", err)
		}
		mu.Lock()
		gotBody = body
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()

	p := provider.New(provider.Config{
		BaseURL:     srv.URL,
		TokenSource: auth.NewStatic("test-key"),
		Model:       "test-model",
		MaxRetries:  1,
	})

	a := New(p, WithTemperature(0.55), WithMaxTokens(321))
	_, result, err := a.Step(context.Background(), []conversation.Message{userMessage("hi")})
	if err != nil {
		t.Fatalf("Step: %v", err)
	}
	if !result.Done {
		t.Fatal("expected Done result")
	}

	mu.Lock()
	defer mu.Unlock()
	if gotBody == nil {
		t.Fatal("backend never received a request body")
	}
	temp, ok := gotBody["temperature"].(float64)
	if !ok || temp != 0.55 {
		t.Errorf("body temperature = %v, want 0.55", gotBody["temperature"])
	}
	maxTok, ok := gotBody["max_tokens"].(float64)
	if !ok || maxTok != 321 {
		t.Errorf("body max_tokens = %v, want 321", gotBody["max_tokens"])
	}
}

// TestWithoutTemperatureAndMaxTokens_OmittedFromWire pins the inverse:
// when the options are not set, the fields must be absent so backends
// that reject explicit values (or apply their own defaults) behave.
func TestWithoutTemperatureAndMaxTokens_OmittedFromWire(t *testing.T) {
	t.Parallel()

	var mu sync.Mutex
	var gotBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(data, &body)
		mu.Lock()
		gotBody = body
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()

	p := provider.New(provider.Config{
		BaseURL:     srv.URL,
		TokenSource: auth.NewStatic("test-key"),
		Model:       "test-model",
		MaxRetries:  1,
	})

	a := New(p)
	if _, _, err := a.Step(context.Background(), []conversation.Message{userMessage("hi")}); err != nil {
		t.Fatalf("Step: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if _, present := gotBody["temperature"]; present {
		t.Error("temperature should be omitted when WithTemperature is not used")
	}
	if _, present := gotBody["max_tokens"]; present {
		t.Error("max_tokens should be omitted when WithMaxTokens is not used")
	}
}
