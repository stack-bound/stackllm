package tui

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/stack-bound/stackllm/agent"
	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/profile"
	"github.com/stack-bound/stackllm/provider"
	"github.com/stack-bound/stackllm/session"
)

// fakeModelLister is a scriptable ModelLister for /models tests.
type fakeModelLister struct {
	mu         sync.Mutex
	all        []profile.ModelInfo
	allErr     error
	recents    []profile.ModelInfo
	recentsErr error
	loaded     *provider.OpenAIProvider
	loadErr    error
	tracked    []profile.ModelInfo
	trackErr   error
}

func (f *fakeModelLister) ListAllModels(_ context.Context) ([]profile.ModelInfo, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.all, f.allErr
}

func (f *fakeModelLister) LoadProviderForModel(_ context.Context, _ profile.ModelInfo) (*provider.OpenAIProvider, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.loaded, f.loadErr
}

func (f *fakeModelLister) RecentModels(_ context.Context) ([]profile.ModelInfo, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.recents, f.recentsErr
}

func (f *fakeModelLister) TrackRecentModel(_ context.Context, info profile.ModelInfo) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.tracked = append(f.tracked, info)
	return f.trackErr
}

// typeString feeds each rune of s through Update as a key press.
func typeString(t *testing.T, m *Model, s string) *Model {
	t.Helper()
	for _, r := range s {
		updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
		m = updated.(*Model)
	}
	return m
}

// sizedTestModel builds a model backed by the given store with a real
// window size applied, so viewport/textarea math behaves like at runtime.
func sizedTestModel(t *testing.T, store session.SessionStore, opts ...Option) *Model {
	t.Helper()
	p := provider.New(provider.OllamaConfig("http://localhost", "test"))
	a := agent.New(p)
	m := New(a, store, opts...)
	updated, _ := m.Update(tea.WindowSizeMsg{Width: 100, Height: 40})
	return updated.(*Model)
}

func TestWithModelLister(t *testing.T) {
	t.Parallel()
	lister := &fakeModelLister{}
	m := sizedTestModel(t, session.NewInMemoryStore(), WithModelLister(lister))
	if m.modelLister != ModelLister(lister) {
		t.Error("WithModelLister did not install the lister")
	}
}

func TestInit_SchedulesBlinkAndTick(t *testing.T) {
	t.Parallel()
	m := newTestModel(t)
	cmd := m.Init()
	if cmd == nil {
		t.Fatal("Init returned nil cmd")
	}
	msg := cmd()
	batch, ok := msg.(tea.BatchMsg)
	if !ok {
		t.Fatalf("expected tea.BatchMsg, got %T", msg)
	}
	if len(batch) != 2 {
		t.Errorf("expected 2 batched cmds (blink + tick), got %d", len(batch))
	}
}

func TestView_StatusLinePerState(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		state modelState
		err   error
		want  string
	}{
		{"idle", stateIdle, nil, "ready"},
		{"running", stateRunning, nil, "thinking..."},
		{"tool call", stateToolCall, nil, "running tool..."},
		{"model loading", stateModelLoading, nil, "loading models..."},
		{"session loading", stateSessionLoading, nil, "loading sessions..."},
		{"command menu", stateCommandMenu, nil, "command"},
		{"model picker", stateModelPicker, nil, "select a model"},
		{"session picker", stateSessionPicker, nil, "select a session"},
		{"fork picker", stateForkPicker, nil, "select fork point"},
		{"error", stateError, errors.New("kaboom"), "error: kaboom"},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			m := sizedTestModel(t, session.NewInMemoryStore())
			m.state = tc.state
			m.err = tc.err
			out := m.View()
			if !strings.Contains(out, tc.want) {
				t.Errorf("View() for %s missing %q:\n%s", tc.name, tc.want, out)
			}
		})
	}
}

func TestView_ShowsModelSuffix(t *testing.T) {
	t.Parallel()
	m := sizedTestModel(t, session.NewInMemoryStore())
	// The agent was built from an Ollama provider with model "test";
	// New mirrors that into currentModel and View must surface it.
	if m.currentModel != "test" {
		t.Fatalf("precondition: currentModel = %q, want %q", m.currentModel, "test")
	}
	if out := m.View(); !strings.Contains(out, "test") {
		t.Errorf("View() should include model name in status suffix:\n%s", out)
	}
}

func TestView_ModalStates(t *testing.T) {
	t.Parallel()

	m := sizedTestModel(t, newFullFakeStore())
	m.openRenameModal()
	out := m.View()
	if !strings.Contains(out, "Rename session") {
		t.Errorf("View() in text modal missing title:\n%s", out)
	}

	m2 := sizedTestModel(t, newFullFakeStore())
	m2.openConfirmModal("Delete session", "Really?", stateIdle, nil)
	out2 := m2.View()
	if !strings.Contains(out2, "Delete session") || !strings.Contains(out2, "Really?") {
		t.Errorf("View() in confirm modal missing title/prompt:\n%s", out2)
	}

	// With no window size yet, modal renders empty rather than panicking.
	m3 := newTestModel(t)
	m3.openRenameModal()
	if out := m3.View(); out != "" {
		t.Errorf("modal View() with zero size should be empty, got %q", out)
	}
	m3.closeModal()
	m3.openConfirmModal("t", "p", stateIdle, nil)
	if out := m3.View(); out != "" {
		t.Errorf("confirm View() with zero size should be empty, got %q", out)
	}
}

func TestView_CommandMenuFlow(t *testing.T) {
	t.Parallel()
	m := sizedTestModel(t, session.NewInMemoryStore())

	// Typing "/" opens the command menu listing every command.
	m = typeString(t, m, "/")
	if m.state != stateCommandMenu {
		t.Fatalf("expected stateCommandMenu after typing /, got %v", m.state)
	}
	out := m.View()
	for _, c := range commands {
		if !strings.Contains(out, c.Name) {
			t.Errorf("command menu missing %q:\n%s", c.Name, out)
		}
	}
	// Cursor marker sits on the first entry.
	if !strings.Contains(out, "> "+commands[0].Name) {
		t.Errorf("expected cursor on %q:\n%s", commands[0].Name, out)
	}

	// Narrowing the query filters the menu.
	m = typeString(t, m, "mo")
	if len(m.cmdFiltered) != 1 || m.cmdFiltered[0].ID != CommandModels {
		t.Fatalf("expected only /models after typing /mo, got %+v", m.cmdFiltered)
	}
	out = m.View()
	if !strings.Contains(out, "/models") {
		t.Errorf("filtered menu missing /models:\n%s", out)
	}
	if strings.Contains(out, "/rename") {
		t.Errorf("filtered menu should not list /rename:\n%s", out)
	}

	// A query with no matches renders the placeholder.
	m = typeString(t, m, "zzz")
	if got := m.renderCommandMenu(); !strings.Contains(got, "no matching commands") {
		t.Errorf("expected no-match placeholder, got %q", got)
	}

	// Esc resets the textarea and returns to idle.
	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = updated.(*Model)
	if m.state != stateIdle {
		t.Errorf("expected stateIdle after Esc, got %v", m.state)
	}
	if m.textarea.Value() != "" {
		t.Errorf("expected textarea reset after Esc, got %q", m.textarea.Value())
	}
}

func TestUpdate_BackspaceOutOfMenuReturnsToIdle(t *testing.T) {
	t.Parallel()
	m := sizedTestModel(t, session.NewInMemoryStore())
	m = typeString(t, m, "/")
	if m.state != stateCommandMenu {
		t.Fatalf("expected stateCommandMenu, got %v", m.state)
	}
	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyBackspace})
	m = updated.(*Model)
	if m.state != stateIdle {
		t.Errorf("expected stateIdle after erasing the slash, got %v", m.state)
	}
	if m.cmdFiltered != nil {
		t.Errorf("expected cmdFiltered cleared, got %+v", m.cmdFiltered)
	}
}

func TestUpdate_CommandMenuNavigationAndEnter(t *testing.T) {
	t.Parallel()
	m := sizedTestModel(t, session.NewInMemoryStore())
	m = typeString(t, m, "/")

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyDown})
	m = updated.(*Model)
	if m.cmdCursor != 1 {
		t.Errorf("cursor after Down = %d, want 1", m.cmdCursor)
	}
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyUp})
	m = updated.(*Model)
	if m.cmdCursor != 0 {
		t.Errorf("cursor after Up = %d, want 0", m.cmdCursor)
	}
	// Up at the top is a no-op.
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyUp})
	m = updated.(*Model)
	if m.cmdCursor != 0 {
		t.Errorf("cursor should clamp at 0, got %d", m.cmdCursor)
	}
	// Down clamps at the last entry.
	for i := 0; i < len(commands)+3; i++ {
		updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyDown})
		m = updated.(*Model)
	}
	if m.cmdCursor != len(commands)-1 {
		t.Errorf("cursor should clamp at %d, got %d", len(commands)-1, m.cmdCursor)
	}
}

func TestUpdate_EnterOnHelpCommandRendersHelp(t *testing.T) {
	t.Parallel()
	m := sizedTestModel(t, session.NewInMemoryStore())
	m = typeString(t, m, "/help")
	if m.state != stateCommandMenu {
		t.Fatalf("expected stateCommandMenu, got %v", m.state)
	}
	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(*Model)
	if m.state != stateIdle {
		t.Errorf("expected stateIdle after /help, got %v", m.state)
	}
	if m.textarea.Value() != "" {
		t.Errorf("textarea should be reset after executing a command, got %q", m.textarea.Value())
	}
	out := m.output.String()
	if !strings.Contains(out, "Commands:") || !strings.Contains(out, "/help") {
		t.Errorf("expected help text in output, got:\n%s", out)
	}
}

func TestExecuteCommand_Dispatch(t *testing.T) {
	t.Parallel()

	t.Run("models without lister errors inline", func(t *testing.T) {
		t.Parallel()
		m := sizedTestModel(t, newFullFakeStore())
		cmd := m.executeCommand(Command{ID: CommandModels})
		if cmd != nil {
			t.Error("expected nil cmd when lister missing")
		}
		if m.state != stateIdle {
			t.Errorf("expected stateIdle, got %v", m.state)
		}
		if !strings.Contains(m.output.String(), "tui.WithModelLister") {
			t.Errorf("expected configuration hint in output, got:\n%s", m.output.String())
		}
	})

	t.Run("models with lister starts load", func(t *testing.T) {
		t.Parallel()
		m := sizedTestModel(t, newFullFakeStore(), WithModelLister(&fakeModelLister{}))
		cmd := m.executeCommand(Command{ID: CommandModels})
		if cmd == nil {
			t.Fatal("expected load cmd")
		}
		if m.state != stateModelLoading {
			t.Errorf("expected stateModelLoading, got %v", m.state)
		}
	})

	t.Run("new swaps in a fresh session", func(t *testing.T) {
		t.Parallel()
		m := sizedTestModel(t, newFullFakeStore())
		oldID := m.session.ID
		cmd := m.executeCommand(Command{ID: CommandNew})
		if cmd != nil {
			t.Error("expected nil cmd for /new")
		}
		if m.session.ID == oldID {
			t.Error("expected a fresh session ID after /new")
		}
		if !strings.Contains(m.output.String(), "New session started.") {
			t.Errorf("expected /new feedback, got:\n%s", m.output.String())
		}
	})

	t.Run("sessions opens picker load", func(t *testing.T) {
		t.Parallel()
		m := sizedTestModel(t, newFullFakeStore())
		cmd := m.executeCommand(Command{ID: CommandSessions})
		if cmd == nil {
			t.Fatal("expected list cmd")
		}
		if m.state != stateSessionLoading {
			t.Errorf("expected stateSessionLoading, got %v", m.state)
		}
	})

	t.Run("rename opens text modal", func(t *testing.T) {
		t.Parallel()
		m := sizedTestModel(t, newFullFakeStore())
		m.executeCommand(Command{ID: CommandRename})
		if m.state != stateTextModal {
			t.Errorf("expected stateTextModal, got %v", m.state)
		}
		if m.modalTitle != "Rename session" {
			t.Errorf("modal title = %q", m.modalTitle)
		}
	})

	t.Run("fork opens fork picker", func(t *testing.T) {
		t.Parallel()
		m := sizedTestModel(t, newFullFakeStore())
		m.session.Messages = []conversation.Message{
			{ID: "m1", Role: conversation.RoleUser, Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "q"}}},
		}
		m.executeCommand(Command{ID: CommandFork})
		if m.state != stateForkPicker {
			t.Errorf("expected stateForkPicker, got %v", m.state)
		}
	})

	t.Run("delete opens confirm modal", func(t *testing.T) {
		t.Parallel()
		m := sizedTestModel(t, newFullFakeStore())
		m.executeCommand(Command{ID: CommandDelete})
		if m.state != stateConfirmModal {
			t.Errorf("expected stateConfirmModal, got %v", m.state)
		}
		if !strings.Contains(m.confirmTitle, "Delete") {
			t.Errorf("confirm title = %q", m.confirmTitle)
		}
	})

	t.Run("export opens text modal", func(t *testing.T) {
		t.Parallel()
		m := sizedTestModel(t, newFullFakeStore())
		m.executeCommand(Command{ID: CommandExport})
		if m.state != stateTextModal {
			t.Errorf("expected stateTextModal, got %v", m.state)
		}
		if m.modalTitle != "Export session" {
			t.Errorf("modal title = %q", m.modalTitle)
		}
	})

	t.Run("unknown id falls through to idle", func(t *testing.T) {
		t.Parallel()
		m := sizedTestModel(t, newFullFakeStore())
		m.state = stateCommandMenu
		cmd := m.executeCommand(Command{ID: "nonexistent"})
		if cmd != nil {
			t.Error("expected nil cmd for unknown command")
		}
		if m.state != stateIdle {
			t.Errorf("expected stateIdle, got %v", m.state)
		}
	})
}

func TestLoadModels_MergesRecentsWithCatalogue(t *testing.T) {
	t.Parallel()

	a := profile.ModelInfo{Provider: "openai", Model: "gpt-4o"}
	b := profile.ModelInfo{Provider: "copilot", Model: "claude-sonnet-4.6"}
	c := profile.ModelInfo{Provider: "ollama", Model: "llama3"}
	gone := profile.ModelInfo{Provider: "openai", Model: "logged-out-model"}

	lister := &fakeModelLister{
		all: []profile.ModelInfo{a, b, c},
		// c is a valid recent; gone is no longer in the catalogue and
		// must be dropped; the duplicate c must be deduped.
		recents: []profile.ModelInfo{c, gone, c},
	}
	m := sizedTestModel(t, session.NewInMemoryStore(), WithModelLister(lister))

	msg := m.loadModels()()
	loaded, ok := msg.(modelsLoadedMsg)
	if !ok {
		t.Fatalf("expected modelsLoadedMsg, got %T", msg)
	}
	if loaded.err != nil {
		t.Fatalf("unexpected err: %v", loaded.err)
	}
	if loaded.recentCount != 1 {
		t.Errorf("recentCount = %d, want 1", loaded.recentCount)
	}
	wantOrder := []string{"ollama/llama3", "openai/gpt-4o", "copilot/claude-sonnet-4.6"}
	if len(loaded.models) != len(wantOrder) {
		t.Fatalf("models = %+v, want %d entries", loaded.models, len(wantOrder))
	}
	for i, want := range wantOrder {
		if loaded.models[i].String() != want {
			t.Errorf("models[%d] = %q, want %q", i, loaded.models[i].String(), want)
		}
	}
}

func TestLoadModels_Errors(t *testing.T) {
	t.Parallel()

	t.Run("list error propagates", func(t *testing.T) {
		t.Parallel()
		lister := &fakeModelLister{allErr: errors.New("network down")}
		m := sizedTestModel(t, session.NewInMemoryStore(), WithModelLister(lister))
		msg := m.loadModels()()
		loaded := msg.(modelsLoadedMsg)
		if loaded.err == nil || !strings.Contains(loaded.err.Error(), "network down") {
			t.Errorf("expected list error, got %v", loaded.err)
		}
	})

	t.Run("recents error is non-fatal", func(t *testing.T) {
		t.Parallel()
		a := profile.ModelInfo{Provider: "openai", Model: "gpt-4o"}
		lister := &fakeModelLister{all: []profile.ModelInfo{a}, recentsErr: errors.New("corrupt state")}
		m := sizedTestModel(t, session.NewInMemoryStore(), WithModelLister(lister))
		msg := m.loadModels()()
		loaded := msg.(modelsLoadedMsg)
		if loaded.err != nil {
			t.Fatalf("recents error must not fail the load: %v", loaded.err)
		}
		if len(loaded.models) != 1 || loaded.recentCount != 0 {
			t.Errorf("models = %+v recentCount = %d, want just the catalogue", loaded.models, loaded.recentCount)
		}
	})
}

func TestUpdate_ModelsLoadedMsg(t *testing.T) {
	t.Parallel()

	t.Run("success opens picker", func(t *testing.T) {
		t.Parallel()
		m := sizedTestModel(t, session.NewInMemoryStore())
		m.state = stateModelLoading
		infos := []profile.ModelInfo{{Provider: "openai", Model: "gpt-4o"}}
		updated, _ := m.Update(modelsLoadedMsg{models: infos, recentCount: 0})
		m = updated.(*Model)
		if m.state != stateModelPicker {
			t.Errorf("expected stateModelPicker, got %v", m.state)
		}
		if len(m.models) != 1 || m.modelCursor != 0 {
			t.Errorf("models = %+v cursor = %d", m.models, m.modelCursor)
		}
	})

	t.Run("empty list returns to idle with hint", func(t *testing.T) {
		t.Parallel()
		m := sizedTestModel(t, session.NewInMemoryStore())
		m.state = stateModelLoading
		updated, _ := m.Update(modelsLoadedMsg{})
		m = updated.(*Model)
		if m.state != stateIdle {
			t.Errorf("expected stateIdle, got %v", m.state)
		}
		if !strings.Contains(m.output.String(), "No models available") {
			t.Errorf("expected empty-list hint, got:\n%s", m.output.String())
		}
	})

	t.Run("error surfaces inline", func(t *testing.T) {
		t.Parallel()
		m := sizedTestModel(t, session.NewInMemoryStore())
		m.state = stateModelLoading
		updated, _ := m.Update(modelsLoadedMsg{err: errors.New("auth expired")})
		m = updated.(*Model)
		if m.state != stateError {
			t.Errorf("expected stateError, got %v", m.state)
		}
		if !strings.Contains(m.output.String(), "Error loading models: auth expired") {
			t.Errorf("expected error line, got:\n%s", m.output.String())
		}
	})
}

func TestSwitchModel_LoadsProvider(t *testing.T) {
	t.Parallel()

	newProv := provider.New(provider.OllamaConfig("http://localhost", "newmodel"))
	lister := &fakeModelLister{loaded: newProv}
	m := sizedTestModel(t, session.NewInMemoryStore(), WithModelLister(lister))

	info := profile.ModelInfo{Provider: "ollama", Model: "newmodel"}
	msg := m.switchModel(info)()
	switched, ok := msg.(modelSwitchedMsg)
	if !ok {
		t.Fatalf("expected modelSwitchedMsg, got %T", msg)
	}
	if switched.err != nil {
		t.Fatalf("unexpected err: %v", switched.err)
	}
	if switched.provider != provider.Provider(newProv) {
		t.Error("expected the lister-built provider to be carried in the msg")
	}
	if switched.info != info {
		t.Errorf("info = %+v, want %+v", switched.info, info)
	}
}

func TestSwitchModel_LoadError(t *testing.T) {
	t.Parallel()
	lister := &fakeModelLister{loadErr: errors.New("no credentials")}
	m := sizedTestModel(t, session.NewInMemoryStore(), WithModelLister(lister))
	msg := m.switchModel(profile.ModelInfo{Provider: "openai", Model: "gpt-4o"})()
	switched := msg.(modelSwitchedMsg)
	if switched.err == nil || !strings.Contains(switched.err.Error(), "no credentials") {
		t.Errorf("expected load error, got %v", switched.err)
	}
}

func TestUpdate_ModelSwitchedMsg(t *testing.T) {
	t.Parallel()

	t.Run("success swaps agent and records recent", func(t *testing.T) {
		t.Parallel()
		lister := &fakeModelLister{}
		m := sizedTestModel(t, session.NewInMemoryStore(), WithModelLister(lister))
		m.state = stateModelLoading

		newProv := provider.New(provider.OllamaConfig("http://localhost", "newmodel"))
		info := profile.ModelInfo{Provider: "ollama", Model: "newmodel", ContextWindow: 12345}
		updated, _ := m.Update(modelSwitchedMsg{provider: newProv, info: info})
		m = updated.(*Model)

		if m.state != stateIdle {
			t.Errorf("expected stateIdle, got %v", m.state)
		}
		if got := m.agent.Model(); got != "newmodel" {
			t.Errorf("agent model = %q, want newmodel", got)
		}
		if m.currentModel != "newmodel" {
			t.Errorf("currentModel = %q, want newmodel", m.currentModel)
		}
		if m.contextWindow != 12345 {
			t.Errorf("contextWindow = %d, want 12345", m.contextWindow)
		}
		if len(lister.tracked) != 1 || lister.tracked[0] != info {
			t.Errorf("expected TrackRecentModel(%+v), got %+v", info, lister.tracked)
		}
		if !strings.Contains(m.output.String(), "Switched to ollama/newmodel") {
			t.Errorf("expected switch feedback, got:\n%s", m.output.String())
		}
	})

	t.Run("zero context window falls back to table", func(t *testing.T) {
		t.Parallel()
		lister := &fakeModelLister{}
		m := sizedTestModel(t, session.NewInMemoryStore(), WithModelLister(lister))
		newProv := provider.New(provider.OllamaConfig("http://localhost", "gpt-4o"))
		info := profile.ModelInfo{Provider: "openai", Model: "gpt-4o"} // ContextWindow zero
		updated, _ := m.Update(modelSwitchedMsg{provider: newProv, info: info})
		m = updated.(*Model)
		if want := provider.ContextWindow("gpt-4o"); m.contextWindow != want {
			t.Errorf("contextWindow = %d, want fallback table value %d", m.contextWindow, want)
		}
	})

	t.Run("track failure warns but still switches", func(t *testing.T) {
		t.Parallel()
		lister := &fakeModelLister{trackErr: errors.New("readonly config")}
		m := sizedTestModel(t, session.NewInMemoryStore(), WithModelLister(lister))
		newProv := provider.New(provider.OllamaConfig("http://localhost", "newmodel"))
		updated, _ := m.Update(modelSwitchedMsg{provider: newProv, info: profile.ModelInfo{Provider: "ollama", Model: "newmodel"}})
		m = updated.(*Model)
		if m.state != stateIdle {
			t.Errorf("expected stateIdle despite track failure, got %v", m.state)
		}
		out := m.output.String()
		if !strings.Contains(out, "failed to record recent model") {
			t.Errorf("expected warning, got:\n%s", out)
		}
		if !strings.Contains(out, "Switched to ollama/newmodel") {
			t.Errorf("expected switch to complete, got:\n%s", out)
		}
	})

	t.Run("error surfaces inline", func(t *testing.T) {
		t.Parallel()
		m := sizedTestModel(t, session.NewInMemoryStore())
		before := m.agent.Model()
		updated, _ := m.Update(modelSwitchedMsg{err: errors.New("bad key"), info: profile.ModelInfo{Provider: "openai", Model: "gpt-4o"}})
		m = updated.(*Model)
		if m.state != stateError {
			t.Errorf("expected stateError, got %v", m.state)
		}
		if !strings.Contains(m.output.String(), "Error switching model: bad key") {
			t.Errorf("expected error line, got:\n%s", m.output.String())
		}
		if got := m.agent.Model(); got != before {
			t.Errorf("agent model must not change on failed switch: got %q, want %q", got, before)
		}
	})
}

func TestUpdate_ModelPickerNavigationAndEnter(t *testing.T) {
	t.Parallel()

	newProv := provider.New(provider.OllamaConfig("http://localhost", "b-model"))
	lister := &fakeModelLister{loaded: newProv}
	m := sizedTestModel(t, session.NewInMemoryStore(), WithModelLister(lister))
	m.state = stateModelPicker
	m.models = []profile.ModelInfo{
		{Provider: "ollama", Model: "a-model"},
		{Provider: "ollama", Model: "b-model"},
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyDown})
	m = updated.(*Model)
	if m.modelCursor != 1 {
		t.Errorf("cursor after Down = %d, want 1", m.modelCursor)
	}
	// Down clamps at the end.
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyDown})
	m = updated.(*Model)
	if m.modelCursor != 1 {
		t.Errorf("cursor should clamp at 1, got %d", m.modelCursor)
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(*Model)
	if m.state != stateModelLoading {
		t.Fatalf("expected stateModelLoading after Enter, got %v", m.state)
	}
	if cmd == nil {
		t.Fatal("expected switch cmd from Enter")
	}
	// Drain the async switch and confirm it targets the model under
	// the cursor.
	msgs := collectMsgs(cmd)
	var switched *modelSwitchedMsg
	for _, msg := range msgs {
		if sw, ok := msg.(modelSwitchedMsg); ok {
			switched = &sw
		}
	}
	if switched == nil {
		t.Fatalf("expected a modelSwitchedMsg among %d cmd results", len(msgs))
	}
	if switched.info.Model != "b-model" {
		t.Errorf("switched to %q, want b-model", switched.info.Model)
	}
}

// collectMsgs executes a tea.Cmd, recursively flattening tea.BatchMsg
// so tests can inspect every message the runtime would deliver.
func collectMsgs(cmd tea.Cmd) []tea.Msg {
	if cmd == nil {
		return nil
	}
	msg := cmd()
	if batch, ok := msg.(tea.BatchMsg); ok {
		var out []tea.Msg
		for _, c := range batch {
			out = append(out, collectMsgs(c)...)
		}
		return out
	}
	if msg == nil {
		return nil
	}
	return []tea.Msg{msg}
}

func TestUpdate_EscLeavesPickerStates(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		setup func(m *Model)
		check func(t *testing.T, m *Model)
	}{
		{
			"model picker",
			func(m *Model) { m.state = stateModelPicker },
			func(t *testing.T, m *Model) {
				if m.state != stateIdle {
					t.Errorf("state = %v, want idle", m.state)
				}
			},
		},
		{
			"model loading",
			func(m *Model) { m.state = stateModelLoading },
			func(t *testing.T, m *Model) {
				if m.state != stateIdle {
					t.Errorf("state = %v, want idle", m.state)
				}
			},
		},
		{
			"session picker clears rows",
			func(m *Model) {
				m.state = stateSessionPicker
				m.sessions = []*session.Session{m.session}
				m.sessionCursor = 0
			},
			func(t *testing.T, m *Model) {
				if m.state != stateIdle {
					t.Errorf("state = %v, want idle", m.state)
				}
				if m.sessions != nil {
					t.Errorf("sessions should be cleared, got %d", len(m.sessions))
				}
			},
		},
		{
			"session loading",
			func(m *Model) { m.state = stateSessionLoading },
			func(t *testing.T, m *Model) {
				if m.state != stateIdle {
					t.Errorf("state = %v, want idle", m.state)
				}
			},
		},
		{
			"fork picker resets cursor",
			func(m *Model) {
				m.state = stateForkPicker
				m.forkCursor = 3
			},
			func(t *testing.T, m *Model) {
				if m.state != stateIdle {
					t.Errorf("state = %v, want idle", m.state)
				}
				if m.forkCursor != 0 {
					t.Errorf("forkCursor = %d, want 0", m.forkCursor)
				}
			},
		},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			m := sizedTestModel(t, newFullFakeStore())
			tc.setup(m)
			updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
			tc.check(t, updated.(*Model))
		})
	}
}

func TestUpdate_CtrlCQuitsAndCancels(t *testing.T) {
	t.Parallel()
	m := sizedTestModel(t, session.NewInMemoryStore())
	cancelled := false
	m.cancel = func() { cancelled = true }

	_, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	if !cancelled {
		t.Error("expected Ctrl+C to invoke the run cancel func")
	}
	if cmd == nil {
		t.Fatal("expected quit cmd")
	}
	if _, ok := cmd().(tea.QuitMsg); !ok {
		t.Errorf("expected tea.QuitMsg, got %T", cmd())
	}
}

func TestUpdate_CtrlJInsertsNewline(t *testing.T) {
	t.Parallel()
	m := sizedTestModel(t, session.NewInMemoryStore())
	m = typeString(t, m, "line1")
	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlJ})
	m = updated.(*Model)
	m = typeString(t, m, "line2")
	if got := m.textarea.Value(); got != "line1\nline2" {
		t.Errorf("textarea = %q, want %q", got, "line1\nline2")
	}
}

func TestUpdate_EnterWhileBusyIsIgnored(t *testing.T) {
	t.Parallel()
	for _, st := range []modelState{stateRunning, stateToolCall, stateModelLoading, stateSessionLoading} {
		m := sizedTestModel(t, session.NewInMemoryStore())
		m.textarea.SetValue("queued text")
		m.state = st
		updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
		m = updated.(*Model)
		if len(m.session.Messages) != 0 {
			t.Errorf("state %v: Enter while busy must not append messages, got %d", st, len(m.session.Messages))
		}
		if m.state != st {
			t.Errorf("state %v: Enter while busy must not change state, got %v", st, m.state)
		}
	}
}

func TestUpdate_EnterOnEmptyInputIsNoop(t *testing.T) {
	t.Parallel()
	m := sizedTestModel(t, session.NewInMemoryStore())
	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(*Model)
	if len(m.session.Messages) != 0 {
		t.Errorf("expected no messages for empty input, got %d", len(m.session.Messages))
	}
	if m.state != stateIdle {
		t.Errorf("expected stateIdle, got %v", m.state)
	}
}

func TestUpdate_SpinnerTickAdvancesWhileBusy(t *testing.T) {
	t.Parallel()
	m := sizedTestModel(t, session.NewInMemoryStore())
	m.state = stateRunning
	_, cmd := m.Update(m.spinner.Tick())
	if cmd == nil {
		t.Error("expected the spinner to schedule its next tick while running")
	}
}

func TestHandleAgentEvent(t *testing.T) {
	t.Parallel()

	t.Run("text delta streams into output", func(t *testing.T) {
		t.Parallel()
		m := sizedTestModel(t, session.NewInMemoryStore())
		updated, _ := m.Update(agentEventMsg{event: agent.Event{
			Type: agent.EventBlockDelta, BlockType: conversation.BlockText, Content: "partial answer",
		}})
		m = updated.(*Model)
		if m.state != stateRunning {
			t.Errorf("expected stateRunning, got %v", m.state)
		}
		if !strings.Contains(m.output.String(), "partial answer") {
			t.Errorf("expected delta in output, got:\n%s", m.output.String())
		}
	})

	t.Run("non-text delta changes state only", func(t *testing.T) {
		t.Parallel()
		m := sizedTestModel(t, session.NewInMemoryStore())
		updated, _ := m.Update(agentEventMsg{event: agent.Event{
			Type: agent.EventBlockDelta, BlockType: conversation.BlockThinking, Content: "hmm",
		}})
		m = updated.(*Model)
		if m.state != stateRunning {
			t.Errorf("expected stateRunning, got %v", m.state)
		}
		if strings.Contains(m.output.String(), "hmm") {
			t.Errorf("thinking delta should not stream via handleAgentEvent, got:\n%s", m.output.String())
		}
	})

	t.Run("tool call", func(t *testing.T) {
		t.Parallel()
		m := sizedTestModel(t, session.NewInMemoryStore())
		updated, _ := m.Update(agentEventMsg{event: agent.Event{
			Type: agent.EventToolCall, ToolCall: &conversation.ToolCall{ID: "c1", Name: "read_file"},
		}})
		m = updated.(*Model)
		if m.state != stateToolCall {
			t.Errorf("expected stateToolCall, got %v", m.state)
		}
		if !strings.Contains(m.output.String(), "⚡ read_file") {
			t.Errorf("expected tool call line, got:\n%s", m.output.String())
		}
	})

	t.Run("tool result", func(t *testing.T) {
		t.Parallel()
		m := sizedTestModel(t, session.NewInMemoryStore())
		updated, _ := m.Update(agentEventMsg{event: agent.Event{
			Type: agent.EventToolResult, ToolCall: &conversation.ToolCall{ID: "c1", Name: "read_file"}, ToolResult: "file contents",
		}})
		m = updated.(*Model)
		if m.state != stateRunning {
			t.Errorf("expected stateRunning, got %v", m.state)
		}
		if !strings.Contains(m.output.String(), "→ file contents") {
			t.Errorf("expected tool result line, got:\n%s", m.output.String())
		}
	})

	t.Run("error", func(t *testing.T) {
		t.Parallel()
		m := sizedTestModel(t, session.NewInMemoryStore())
		updated, _ := m.Update(agentEventMsg{event: agent.Event{
			Type: agent.EventError, Err: errors.New("rate limited"),
		}})
		m = updated.(*Model)
		if m.state != stateError {
			t.Errorf("expected stateError, got %v", m.state)
		}
		if !strings.Contains(m.output.String(), "Error: rate limited") {
			t.Errorf("expected error line, got:\n%s", m.output.String())
		}
	})
}

func TestUpdate_AgentDoneMsgPersistsSession(t *testing.T) {
	t.Parallel()
	store := newFullFakeStore()
	m := sizedTestModel(t, store)
	m.state = stateRunning
	m.session.AppendMessage(conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hi"}},
	})

	updated, _ := m.Update(agentDoneMsg{})
	m = updated.(*Model)
	if m.state != stateIdle {
		t.Errorf("expected stateIdle after agentDoneMsg, got %v", m.state)
	}
	saved, err := store.Load(context.Background(), m.session.ID)
	if err != nil {
		t.Fatalf("expected session persisted on agentDoneMsg: %v", err)
	}
	if len(saved.Messages) != 1 || saved.Messages[0].TextContent() != "hi" {
		t.Errorf("persisted session content wrong: %+v", saved.Messages)
	}
}

func TestRenderMenu_EmptyOutsideMenuStates(t *testing.T) {
	t.Parallel()
	m := sizedTestModel(t, session.NewInMemoryStore())
	for _, st := range []modelState{stateIdle, stateRunning, stateError, stateTextModal, stateConfirmModal} {
		m.state = st
		if got := m.renderMenu(); got != "" {
			t.Errorf("renderMenu in state %v = %q, want empty", st, got)
		}
	}
}

func TestRenderModelPicker_ScrollWindow(t *testing.T) {
	t.Parallel()
	m := sizedTestModel(t, session.NewInMemoryStore())
	m.state = stateModelPicker
	for i := 0; i < 15; i++ {
		m.models = append(m.models, profile.ModelInfo{Provider: "ollama", Model: fmt.Sprintf("model-%02d", i)})
	}
	m.modelCursor = 14

	out := m.renderModelPicker()
	if !strings.Contains(out, "(15/15)") {
		t.Errorf("expected scroll position indicator, got:\n%s", out)
	}
	if !strings.Contains(out, "> ollama/model-14") {
		t.Errorf("expected cursor on last model, got:\n%s", out)
	}
	if strings.Contains(out, "ollama/model-00") {
		t.Errorf("first model should be scrolled out of view, got:\n%s", out)
	}
}
