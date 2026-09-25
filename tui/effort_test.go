package tui

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/stack-bound/stackllm/agent"
	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/provider"
	"github.com/stack-bound/stackllm/session"
)

// fakeEffortStore is an in-memory EffortStore that records every save.
type fakeEffortStore struct {
	mu      sync.Mutex
	effort  string
	loadErr error
	saveErr error
	saved   []string
}

func (f *fakeEffortStore) ReasoningEffort(_ context.Context) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.effort, f.loadErr
}

func (f *fakeEffortStore) SetReasoningEffort(_ context.Context, effort string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.saved = append(f.saved, effort)
	if f.saveErr != nil {
		return f.saveErr
	}
	f.effort = effort
	return nil
}

// recordingProvider answers every call with a short text reply, or with
// failErr as a stream error when set, and records the reasoning effort
// each request carried.
type recordingProvider struct {
	mu      sync.Mutex
	efforts []string
	failErr error
}

func (p *recordingProvider) Complete(_ context.Context, req provider.Request) (<-chan provider.Event, error) {
	p.mu.Lock()
	p.efforts = append(p.efforts, req.ReasoningEffort)
	failErr := p.failErr
	p.mu.Unlock()
	events := concatEvents(textEvents("ok"), []provider.Event{{Type: provider.EventTypeDone}})
	if failErr != nil {
		events = []provider.Event{{Type: provider.EventTypeError, Err: failErr}}
	}
	ch := make(chan provider.Event, len(events))
	for _, ev := range events {
		ch <- ev
	}
	close(ch)
	return ch, nil
}

func (p *recordingProvider) Models(_ context.Context) ([]provider.ModelMeta, error) {
	return nil, nil
}

func (p *recordingProvider) sentEfforts() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.efforts...)
}

// effortTestModel builds a sized TUI around an agent on the Ollama
// provider (model "test") so the status line has a model name to show.
func effortTestModel(t *testing.T, store EffortStore, agentOpts ...agent.Option) *Model {
	t.Helper()
	p := provider.New(provider.OllamaConfig("http://localhost", "test"))
	a := agent.New(p, agentOpts...)
	var opts []Option
	if store != nil {
		opts = append(opts, WithEffortStore(store))
	}
	m := New(a, session.NewInMemoryStore(), opts...)
	updated, _ := m.Update(tea.WindowSizeMsg{Width: 100, Height: 40})
	return updated.(*Model)
}

// statusLine returns the "● ready" row of the idle view, so assertions
// about the status suffix are not fooled by text in the scrollback.
func statusLine(t *testing.T, m *Model) string {
	t.Helper()
	for _, line := range strings.Split(m.View(), "\n") {
		if strings.Contains(line, "● ready") {
			return line
		}
	}
	t.Fatalf("no status line in view:\n%s", m.View())
	return ""
}

func pressKey(m *Model, k tea.KeyType) *Model {
	updated, _ := m.Update(tea.KeyMsg{Type: k})
	return updated.(*Model)
}

// openEffortViaMenu drives /effort exactly as a user would: type it into
// the textarea and hit Enter on the command menu.
func openEffortViaMenu(t *testing.T, m *Model) *Model {
	t.Helper()
	m = typeString(t, m, "/effort")
	if m.state != stateCommandMenu {
		t.Fatalf("after typing /effort state = %v, want command menu", m.state)
	}
	m = pressKey(m, tea.KeyEnter)
	if m.state != stateEffortPicker {
		t.Fatalf("after Enter state = %v, want effort picker", m.state)
	}
	return m
}

// moveEffortCursorTo presses Up/Down until the cursor is on level.
func moveEffortCursorTo(t *testing.T, m *Model, level string) *Model {
	t.Helper()
	target := -1
	for i, c := range m.effortChoices {
		if c.level == level {
			target = i
		}
	}
	if target < 0 {
		t.Fatalf("level %q not in picker", level)
	}
	for m.effortCursor < target {
		m = pressKey(m, tea.KeyDown)
	}
	for m.effortCursor > target {
		m = pressKey(m, tea.KeyUp)
	}
	return m
}

func TestNew_AppliesStoredEffort(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		store     *fakeEffortStore
		agentOpts []agent.Option
		want      string
		wantOut   string
	}{
		{
			name:  "stored effort is applied",
			store: &fakeEffortStore{effort: provider.ReasoningEffortHigh},
			want:  provider.ReasoningEffortHigh,
		},
		{
			name:      "stored effort beats the agent option",
			store:     &fakeEffortStore{effort: provider.ReasoningEffortNone},
			agentOpts: []agent.Option{agent.WithReasoningEffort(provider.ReasoningEffortLow)},
			want:      provider.ReasoningEffortNone,
		},
		{
			name:      "nothing stored keeps the agent option",
			store:     &fakeEffortStore{},
			agentOpts: []agent.Option{agent.WithReasoningEffort(provider.ReasoningEffortLow)},
			want:      provider.ReasoningEffortLow,
		},
		{
			name:      "load error warns and keeps the agent option",
			store:     &fakeEffortStore{loadErr: errors.New("disk on fire")},
			agentOpts: []agent.Option{agent.WithReasoningEffort(provider.ReasoningEffortLow)},
			want:      provider.ReasoningEffortLow,
			wantOut:   "failed to load reasoning effort: disk on fire",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			m := effortTestModel(t, tc.store, tc.agentOpts...)
			if got := m.agent.ReasoningEffort(); got != tc.want {
				t.Errorf("agent effort = %q, want %q", got, tc.want)
			}
			if tc.wantOut != "" && !strings.Contains(m.output.String(), tc.wantOut) {
				t.Errorf("output missing %q:\n%s", tc.wantOut, m.output.String())
			}
		})
	}
}

// TestEffortPicker_SelectAppliesPersistsAndReachesProvider is the
// end-to-end path: pick a level, check it is saved, shown in the status
// line, and actually sent on the next agent run.
func TestEffortPicker_SelectAppliesPersistsAndReachesProvider(t *testing.T) {
	t.Parallel()

	store := &fakeEffortStore{}
	rec := &recordingProvider{}
	m := New(agent.New(rec), newFullFakeStore(), WithEffortStore(store))
	m.currentModel = "test"
	updated, _ := m.Update(tea.WindowSizeMsg{Width: 100, Height: 40})
	m = updated.(*Model)

	m = openEffortViaMenu(t, m)
	if got := m.effortChoices[m.effortCursor].level; got != "" {
		t.Errorf("cursor opened on %q, want the current (default) row", got)
	}
	if out := m.View(); !strings.Contains(out, "select reasoning effort") || !strings.Contains(out, "xhigh") {
		t.Errorf("picker view missing status or levels:\n%s", out)
	}

	m = moveEffortCursorTo(t, m, provider.ReasoningEffortHigh)
	m = pressKey(m, tea.KeyEnter)

	if m.state != stateIdle {
		t.Errorf("state after select = %v, want idle", m.state)
	}
	if got := m.agent.ReasoningEffort(); got != provider.ReasoningEffortHigh {
		t.Errorf("agent effort = %q, want high", got)
	}
	if len(store.saved) != 1 || store.saved[0] != provider.ReasoningEffortHigh {
		t.Errorf("saved = %v, want [high]", store.saved)
	}
	if !strings.Contains(m.output.String(), "Reasoning effort: high") {
		t.Errorf("output missing confirmation:\n%s", m.output.String())
	}
	if line := statusLine(t, m); !strings.Contains(line, "test · effort high") {
		t.Errorf("status line missing effort: %q", line)
	}
	if m.textarea.Value() != "" {
		t.Errorf("textarea = %q, want empty (picker keys must not leak)", m.textarea.Value())
	}

	m.session.AppendMessage(conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hi"}},
	})
	if _, ok := m.runAgent()().(agentDoneMsg); !ok {
		t.Fatal("runAgent did not finish")
	}

	// Back to default: the next request must carry no effort at all.
	m = openEffortViaMenu(t, m)
	if got := m.effortChoices[m.effortCursor].level; got != provider.ReasoningEffortHigh {
		t.Errorf("cursor reopened on %q, want high", got)
	}
	for _, line := range strings.Split(m.renderEffortPicker(), "\n") {
		isHigh := strings.Contains(line, "thorough, slower")
		if marked := strings.Contains(line, "(current)"); marked != isHigh {
			t.Errorf("picker line %q: marked current = %v, want %v", line, marked, isHigh)
		}
	}
	m = moveEffortCursorTo(t, m, "")
	m = pressKey(m, tea.KeyEnter)
	if got := m.agent.ReasoningEffort(); got != "" {
		t.Errorf("agent effort after default = %q, want empty", got)
	}
	if line := statusLine(t, m); strings.Contains(line, "effort") {
		t.Errorf("status line should drop effort once cleared: %q", line)
	}
	if _, ok := m.runAgent()().(agentDoneMsg); !ok {
		t.Fatal("second runAgent did not finish")
	}

	want := []string{provider.ReasoningEffortHigh, ""}
	got := rec.sentEfforts()
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Errorf("provider saw efforts %q, want %q", got, want)
	}
	if len(store.saved) != 2 || store.saved[1] != "" {
		t.Errorf("saved = %q, want [high, \"\"]", store.saved)
	}
}

func TestEffortPicker_EscLeavesEffortUnchanged(t *testing.T) {
	t.Parallel()

	store := &fakeEffortStore{effort: provider.ReasoningEffortLow}
	m := effortTestModel(t, store)
	m = openEffortViaMenu(t, m)
	m = moveEffortCursorTo(t, m, provider.ReasoningEffortMax)
	m = pressKey(m, tea.KeyEsc)

	if m.state != stateIdle {
		t.Errorf("state after Esc = %v, want idle", m.state)
	}
	if got := m.agent.ReasoningEffort(); got != provider.ReasoningEffortLow {
		t.Errorf("agent effort after Esc = %q, want low", got)
	}
	if len(store.saved) != 0 {
		t.Errorf("Esc saved %v, want nothing", store.saved)
	}
}

func TestEffortPicker_CursorStaysInBounds(t *testing.T) {
	t.Parallel()

	m := effortTestModel(t, nil)
	m = openEffortViaMenu(t, m)
	m = pressKey(m, tea.KeyUp)
	if m.effortCursor != 0 {
		t.Errorf("cursor after Up at top = %d, want 0", m.effortCursor)
	}
	for range len(m.effortChoices) + 3 {
		m = pressKey(m, tea.KeyDown)
	}
	if want := len(m.effortChoices) - 1; m.effortCursor != want {
		t.Errorf("cursor after overshooting Down = %d, want %d", m.effortCursor, want)
	}
	m = pressKey(m, tea.KeyEnter)
	if got := m.agent.ReasoningEffort(); got != provider.ReasoningEffortMax {
		t.Errorf("effort after picking last row = %q, want max", got)
	}
}

// TestEffortPicker_WorksWithoutStore covers embedders that pass no
// EffortStore: the choice still applies for this run.
func TestEffortPicker_WorksWithoutStore(t *testing.T) {
	t.Parallel()

	m := effortTestModel(t, nil)
	m = openEffortViaMenu(t, m)
	m = moveEffortCursorTo(t, m, provider.ReasoningEffortNone)
	m = pressKey(m, tea.KeyEnter)
	if got := m.agent.ReasoningEffort(); got != provider.ReasoningEffortNone {
		t.Errorf("agent effort = %q, want none", got)
	}
	if strings.Contains(m.output.String(), "Warning") {
		t.Errorf("unexpected warning without a store:\n%s", m.output.String())
	}
}

func TestEffortPicker_SaveErrorStillApplies(t *testing.T) {
	t.Parallel()

	store := &fakeEffortStore{saveErr: errors.New("read-only fs")}
	m := effortTestModel(t, store)
	m = openEffortViaMenu(t, m)
	m = moveEffortCursorTo(t, m, provider.ReasoningEffortMedium)
	m = pressKey(m, tea.KeyEnter)

	if got := m.agent.ReasoningEffort(); got != provider.ReasoningEffortMedium {
		t.Errorf("agent effort = %q, want medium", got)
	}
	if !strings.Contains(m.output.String(), "failed to save reasoning effort: read-only fs") {
		t.Errorf("output missing save warning:\n%s", m.output.String())
	}
}

// TestRunAgent_EffortHintOn400 checks an unsupported level is explained:
// a 400 with an effort set gets a pointer to /effort, and other errors
// (or a 400 with no effort) do not.
func TestRunAgent_EffortHintOn400(t *testing.T) {
	t.Parallel()

	const bad400 = `provider: status 400: {"error":{"message":"Unsupported value: 'reasoning_effort' does not support 'xhigh'"}}`
	tests := []struct {
		name     string
		effort   string
		err      error
		wantHint bool
	}{
		{"400 with effort", provider.ReasoningEffortXHigh, errors.New(bad400), true},
		{"400 without effort", "", errors.New(bad400), false},
		{"other error with effort", provider.ReasoningEffortXHigh, errors.New("provider: status 401: nope"), false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			rec := &recordingProvider{failErr: tc.err}
			m := runModelWithProvider(t, rec, newFullFakeStore(), agent.WithReasoningEffort(tc.effort))
			if _, ok := m.runAgent()().(agentDoneMsg); !ok {
				t.Fatal("runAgent did not finish")
			}
			out := m.output.String()
			if !strings.Contains(out, "status") {
				t.Fatalf("error not shown:\n%s", out)
			}
			hasHint := strings.Contains(out, `Reasoning effort is "xhigh"`) && strings.Contains(out, "/effort")
			if hasHint != tc.wantHint {
				t.Errorf("hint shown = %v, want %v:\n%s", hasHint, tc.wantHint, out)
			}
		})
	}
}
