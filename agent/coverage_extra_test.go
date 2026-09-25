package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
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
	maxTok, ok := gotBody["max_completion_tokens"].(float64)
	if !ok || maxTok != 321 {
		t.Errorf("body max_completion_tokens = %v, want 321", gotBody["max_completion_tokens"])
	}
	if _, present := gotBody["max_tokens"]; present {
		t.Errorf("body max_tokens = %v, want absent (max_completion_tokens is the primary parameter)", gotBody["max_tokens"])
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
	if _, present := gotBody["max_completion_tokens"]; present {
		t.Error("max_completion_tokens should be omitted when WithMaxTokens is not used")
	}
}

// TestWithReasoningEffort_ReachesWire pins the option all the way to the
// JSON body on both wire formats: /responses nests the level under
// "reasoning", chat completions sends it flat as "reasoning_effort", and
// leaving the option unset keeps the field off the wire entirely so the
// model applies its own default.
func TestWithReasoningEffort_ReachesWire(t *testing.T) {
	t.Parallel()

	const responsesSSE = "event: response.output_item.added\n" +
		"data: {\"output_index\":0,\"item\":{\"type\":\"message\"}}\n\n" +
		"event: response.output_text.delta\n" +
		"data: {\"output_index\":0,\"delta\":\"ok\"}\n\n" +
		"event: response.output_item.done\n" +
		"data: {\"output_index\":0,\"item\":{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"ok\"}]}}\n\n" +
		"event: response.completed\n" +
		"data: {\"response\":{}}\n\n"
	const chatSSE = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n"

	cases := []struct {
		name     string
		endpoint string
		sse      string
		opts     []Option
		// check inspects the body the backend received.
		check func(t *testing.T, body map[string]any)
	}{
		{
			name:     "responses nests the effort",
			endpoint: provider.EndpointResponses,
			sse:      responsesSSE,
			opts:     []Option{WithReasoningEffort(provider.ReasoningEffortNone)},
			check: func(t *testing.T, body map[string]any) {
				reasoning, ok := body["reasoning"].(map[string]any)
				if !ok {
					t.Fatalf("body reasoning = %+v, want a map", body["reasoning"])
				}
				if reasoning["effort"] != provider.ReasoningEffortNone {
					t.Errorf("body reasoning.effort = %v, want %q", reasoning["effort"], provider.ReasoningEffortNone)
				}
			},
		},
		{
			name:     "chat completions sends it flat",
			endpoint: provider.EndpointChatCompletions,
			sse:      chatSSE,
			opts:     []Option{WithReasoningEffort(provider.ReasoningEffortLow)},
			check: func(t *testing.T, body map[string]any) {
				if body["reasoning_effort"] != provider.ReasoningEffortLow {
					t.Errorf("body reasoning_effort = %v, want %q", body["reasoning_effort"], provider.ReasoningEffortLow)
				}
			},
		},
		{
			name:     "omitted without the option",
			endpoint: provider.EndpointResponses,
			sse:      responsesSSE,
			check: func(t *testing.T, body map[string]any) {
				if _, present := body["reasoning"]; present {
					t.Errorf("body reasoning = %v, want absent", body["reasoning"])
				}
			},
		},
	}

	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
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
				fmt.Fprint(w, test.sse)
			}))
			defer srv.Close()

			p := provider.New(provider.Config{
				BaseURL:     srv.URL,
				TokenSource: auth.NewStatic("test-key"),
				Model:       "test-model",
				Endpoint:    test.endpoint,
				MaxRetries:  1,
			})

			a := New(p, test.opts...)
			if _, _, err := a.Step(context.Background(), []conversation.Message{userMessage("hi")}); err != nil {
				t.Fatalf("Step: %v", err)
			}

			mu.Lock()
			defer mu.Unlock()
			if gotBody == nil {
				t.Fatal("backend never received a request body")
			}
			test.check(t, gotBody)
		})
	}
}

// TestSetReasoningEffort_ChangesWireBetweenSteps pins the runtime
// mutator the TUI's /effort command relies on: each Step sends whatever
// SetReasoningEffort last set, and setting it back to empty drops the
// field so the model's own default applies again.
func TestSetReasoningEffort_ChangesWireBetweenSteps(t *testing.T) {
	t.Parallel()

	var mu sync.Mutex
	var bodies []map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		var body map[string]any
		if err := json.Unmarshal(data, &body); err != nil {
			t.Errorf("request body is not JSON: %v", err)
		}
		mu.Lock()
		bodies = append(bodies, body)
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
	a := New(p, WithReasoningEffort(provider.ReasoningEffortLow))

	steps := []struct {
		set  bool
		to   string
		want string // "" means the field must be absent
	}{
		{want: provider.ReasoningEffortLow},
		{set: true, to: provider.ReasoningEffortHigh, want: provider.ReasoningEffortHigh},
		{set: true, to: "", want: ""},
	}
	for i, step := range steps {
		if step.set {
			a.SetReasoningEffort(step.to)
		}
		if got := a.ReasoningEffort(); got != step.want {
			t.Errorf("step %d: ReasoningEffort() = %q, want %q", i, got, step.want)
		}
		if _, _, err := a.Step(context.Background(), []conversation.Message{userMessage("hi")}); err != nil {
			t.Fatalf("step %d: Step: %v", i, err)
		}
	}

	mu.Lock()
	defer mu.Unlock()
	if len(bodies) != len(steps) {
		t.Fatalf("backend saw %d requests, want %d", len(bodies), len(steps))
	}
	for i, step := range steps {
		got, present := bodies[i]["reasoning_effort"]
		switch {
		case step.want == "" && present:
			t.Errorf("request %d: reasoning_effort = %v, want absent", i, got)
		case step.want != "" && got != step.want:
			t.Errorf("request %d: reasoning_effort = %v, want %q", i, got, step.want)
		}
	}
}

// TestExtraBody_ReachesWireAndSwitchesPerModel pins the agent-level
// extra body the way an OpenRouter embedder uses it: WithExtraBody
// routing reaches the wire and overrides the provider's configured
// default per key; SetModel + SetExtraBody switch routing along with
// the model between steps; nil clears it back to the provider default.
// It also pins the copy semantics, so neither the caller's map nor the
// map ExtraBody() hands back can change what the agent sends.
func TestExtraBody_ReachesWireAndSwitchesPerModel(t *testing.T) {
	t.Parallel()

	var mu sync.Mutex
	var bodies []map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		var body map[string]any
		if err := json.Unmarshal(data, &body); err != nil {
			t.Errorf("request body is not JSON: %v", err)
		}
		mu.Lock()
		bodies = append(bodies, body)
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()

	p := provider.New(provider.Config{
		BaseURL:     srv.URL,
		TokenSource: auth.NewStatic("test-key"),
		Model:       "meta-llama/llama-3.3-70b-instruct",
		MaxRetries:  1,
		ExtraBody: map[string]any{
			"provider": map[string]any{"sort": "price"},
			"user":     "cfg-user",
		},
	})

	groq := map[string]any{"order": []any{"groq"}, "allow_fallbacks": false}
	callerMap := map[string]any{"provider": groq}
	a := New(p, WithExtraBody(callerMap))
	// Mutating the caller's map after New must not change the wire.
	callerMap["provider"] = "mutated"
	// Nor may mutating the copy ExtraBody() returns.
	a.ExtraBody()["provider"] = "mutated"

	azure := map[string]any{"only": []any{"azure"}}
	steps := []struct {
		name         string
		apply        func()
		wantModel    string
		wantProvider any
	}{
		{
			name:         "WithExtraBody overrides config key",
			wantModel:    "meta-llama/llama-3.3-70b-instruct",
			wantProvider: groq,
		},
		{
			name: "switch model and routing together",
			apply: func() {
				a.SetModel("openai/gpt-4o")
				a.SetExtraBody(map[string]any{"provider": azure})
			},
			wantModel:    "openai/gpt-4o",
			wantProvider: azure,
		},
		{
			name:         "nil clears back to the provider default",
			apply:        func() { a.SetExtraBody(nil) },
			wantModel:    "openai/gpt-4o",
			wantProvider: map[string]any{"sort": "price"},
		},
	}
	for _, step := range steps {
		if step.apply != nil {
			step.apply()
		}
		if _, _, err := a.Step(context.Background(), []conversation.Message{userMessage("hi")}); err != nil {
			t.Fatalf("%s: Step: %v", step.name, err)
		}
	}

	mu.Lock()
	defer mu.Unlock()
	if len(bodies) != len(steps) {
		t.Fatalf("backend saw %d requests, want %d", len(bodies), len(steps))
	}
	for i, step := range steps {
		body := bodies[i]
		if body["model"] != step.wantModel {
			t.Errorf("%s: model = %v, want %s", step.name, body["model"], step.wantModel)
		}
		if !reflect.DeepEqual(body["provider"], step.wantProvider) {
			t.Errorf("%s: provider = %#v, want %#v", step.name, body["provider"], step.wantProvider)
		}
		// Config keys the agent did not override always reach the wire.
		if body["user"] != "cfg-user" {
			t.Errorf("%s: user = %v, want cfg-user from Config.ExtraBody", step.name, body["user"])
		}
	}
}
