# Plan: Reusable Profile Management Layer

## Context

The TUI example requires an OpenAI API key, but you want to use Copilot (or other providers) without reimplementing login code every time. The library already has auth flows for all providers, but each example hardcodes its own provider setup. We need a shared layer that handles login, provider selection, model listing, and default persistence — usable across projects with zero boilerplate.

## New Packages

### 1. `config/` — User preferences persistence

**File: `config/config.go`**

Mirrors `auth.FileStore` pattern. Persists to `~/.config/stackllm/config.json` (same XDG logic as `auth.FileStore.path()`). Atomic writes.

```go
type ProviderSettings struct {
    BaseURL    string `json:"base_url,omitempty"`    // Ollama
    Endpoint   string `json:"endpoint,omitempty"`    // Azure
    Deployment string `json:"deployment,omitempty"`  // Azure
    APIVersion string `json:"api_version,omitempty"` // Azure
}

type Config struct {
    DefaultProvider string                      `json:"default_provider,omitempty"`
    DefaultModel    string                      `json:"default_model,omitempty"`
    Providers       map[string]ProviderSettings  `json:"providers,omitempty"`
}

type Store struct {
    AppName string
    Path    string // override for testing
}

func (s *Store) Load() (*Config, error)   // returns zero Config if file missing (not an error)
func (s *Store) Save(cfg *Config) error
```

Non-secret settings only (Ollama URL, default provider/model). Secrets stay in `auth.json`.

**File: `config/config_test.go`** — round-trip save/load, missing file returns zero config, atomic write leaves no `.tmp`

### 2. `profile/` — Composes auth + config + provider

**File: `profile/profile.go`**

Single entry point that wires everything together. No global state.

```go
const (
    ProviderOpenAI  = "openai"
    ProviderCopilot = "copilot"
    ProviderGemini  = "gemini"
    ProviderOllama  = "ollama"
)

type Manager struct { /* authStore, configStore, callbacks, httpClient */ }

type Callbacks struct {
    OnDeviceCode func(userCode, verifyURL string)
    OnPolling    func()
    OnSuccess    func()
    OnPromptKey  func(providerName string) (string, error)  // prompt for API key
    OnPromptURL  func(providerName, defaultURL string) (string, error) // prompt for base URL
}

type Option func(*Manager)
func WithHTTPClient(c *http.Client) Option
func WithCallbacks(cb Callbacks) Option
func WithAuthStore(s auth.TokenStore) Option   // testing
func WithConfigStore(s *config.Store) Option   // testing

func New(opts ...Option) *Manager
```

Key methods:

| Method | What it does |
|--------|-------------|
| `AvailableProviders() []string` | Returns `[openai, copilot, gemini, ollama]` |
| `Login(ctx, provider) error` | Runs auth flow: OnPromptKey for openai/gemini, device flow for copilot, OnPromptURL for ollama |
| `Logout(ctx, provider) error` | Clears stored tokens/keys |
| `Status(ctx) ([]ProviderStatus, error)` | Shows which providers are authenticated + which is default |
| `ListModels(ctx, provider) ([]string, error)` | Creates temp provider, calls `Models()` |
| `ListAllModels(ctx) ([]ModelInfo, error)` | Queries all authenticated providers, returns combined list |
| `SetDefault(providerSlashModel) error` | Parses `"copilot/gpt-5.4"` format, persists to config.json |
| `LoadDefault(ctx) (*provider.OpenAIProvider, error)` | Returns ready-to-use provider from persisted config |
| `LoadProvider(ctx, provider, model) (*provider.OpenAIProvider, error)` | Returns provider for specific name+model |

**`ModelInfo` type** — used by `ListAllModels` and the CLI for display + selection:

```go
type ModelInfo struct {
    Provider string // e.g. "copilot"
    Model    string // e.g. "gpt-5.4"
}

func (m ModelInfo) String() string { return m.Provider + "/" + m.Model }
```

`ListAllModels` calls `Status(ctx)` to find authenticated providers, then calls `ListModels` for each in parallel (via goroutines), collects results into a `[]ModelInfo` sorted by provider then model name. Errors from individual providers are logged/skipped (e.g. Ollama offline), not fatal.

Login details per provider:
- **openai**: `OnPromptKey("openai")` → save key to auth store under `"openai_api_key"`
- **gemini**: `OnPromptKey("gemini")` → save key under `"gemini_api_key"`
- **copilot**: Creates `auth.CopilotTokenSource` with callbacks, calls `Login(ctx)` (existing device flow)
- **ollama**: `OnPromptURL("ollama", "http://localhost:11434")` → save URL to config store

`LoadProvider` wiring:
- openai → `auth.NewStatic(key)` → `provider.OpenAIConfig(model, ts)`
- copilot → `auth.NewCachingSource(auth.NewCopilotSource(...))` → `provider.CopilotConfig(model, ts)`
- gemini → `auth.NewStatic(key)` → `provider.GeminiConfig(model, ts)`
- ollama → load base_url from config → `provider.OllamaConfig(baseURL, model)`

**File: `profile/profile_test.go`** — Uses `auth.MemoryStore` + `config.Store{Path: t.TempDir()}` + `httptest.NewServer` for model listing. Tests: login saves credentials, logout clears them, status reflects state, LoadDefault constructs correct provider, error when no default set.

### 3. `examples/login/main.go` — Interactive CLI tool

Fully menu-driven. No subcommand args needed (though they can be supported as shortcuts). Uses `fmt.Scanln` / `bufio.Scanner` for input — no external deps.

**Main menu:**
```
stackllm — Provider Management

1) Login to a provider
2) Logout from a provider
3) Show status
4) Browse models
5) Set default model
6) Exit

Choose: _
```

**Login flow** — shows numbered list of providers, user picks:
```
Available providers:
  1) openai
  2) copilot
  3) gemini
  4) ollama

Choose provider: 2

Open https://github.com/login/device and enter code: ABCD-1234
....
Authenticated!
```

**Browse models** — aggregates from ALL authenticated providers via `ListAllModels`:
```
Fetching models from authenticated providers...

  copilot:
    1) copilot/gpt-5.4
    2) copilot/gpt-5.4-mini
    3) copilot/claude-3.5-sonnet
  ollama:
    4) ollama/llama3
    5) ollama/codellama

(23 models total)
```

**Set default** — shows the same combined model list, user picks by number:
```
Select default model:

  1) copilot/gpt-5.4
  2) copilot/gpt-5.4-mini
  3) ollama/llama3
  ...

Choose [1]: _

Default set: copilot/gpt-5.4
```

Provider is inferred from the selection — no need to type it separately. The selected `ModelInfo.String()` (`"copilot/gpt-5.4"`) is passed directly to `SetDefault()` which splits on `/` and saves `default_provider` + `default_model` to config.

**Subcommand shortcuts** also supported for scripting:
```
go run ./examples/login login copilot
go run ./examples/login status
go run ./examples/login models
go run ./examples/login default copilot/gpt-5.4
```

### 4. Update `examples/tui/main.go`

Replace hardcoded `OPENAI_API_KEY` with `profile.Manager.LoadDefault(ctx)`:

```go
mgr := profile.New(profile.WithCallbacks(profile.Callbacks{
    OnDeviceCode: func(userCode, verifyURL string) {
        fmt.Println(tui.DeviceCodePrompt(userCode, verifyURL))
    },
    OnPromptKey: func(name string) (string, error) { /* fmt.Scanln */ },
}))

p, err := mgr.LoadDefault(context.Background())
if err != nil {
    fmt.Fprintln(os.Stderr, "No default provider. Run: go run ./examples/login")
    os.Exit(1)
}
// ... rest unchanged, use p
```

### 5. Update all examples to use `gpt-5.4`

All existing examples hardcode `gpt-4o` which is outdated. Update to `gpt-5.4`:
- `examples/simple/main.go` — `provider.OpenAIConfig("gpt-4o", ...)` → `"gpt-5.4"`
- `examples/copilot/main.go` — `provider.CopilotConfig("gpt-4o", ...)` → `"gpt-5.4"`

## Implementation Order

1. **`config/config.go` + `config/config_test.go`** — no deps on new code
2. **`profile/profile.go` + `profile/profile_test.go`** — depends on config/, auth/, provider/
3. **`examples/login/main.go`** — exercises profile/
4. **Update `examples/tui/main.go`** — uses profile.LoadDefault()
5. **Update `examples/simple/main.go` + `examples/copilot/main.go`** — gpt-4o → gpt-5.4

## Design Decisions

- **Secrets in auth.json, prefs in config.json** — API keys are secrets, Ollama URL and default model are not
- **Callbacks instead of direct stdin** — keeps profile/ testable and usable in non-interactive contexts (TUI, web)
- **Azure deferred** — most config fields of any provider, skip for MVP. Types have Azure fields for forward compat.
- **No OpenAI OAuth in login CLI** — API key is the common case. OAuth flows available via auth/ directly.
- **`profile.New()` creates FileStore internally** — single place for path decisions, options override for tests

## Files to Create/Modify

| File | Action |
|------|--------|
| `config/config.go` | Create |
| `config/config_test.go` | Create |
| `profile/profile.go` | Create |
| `profile/profile_test.go` | Create |
| `examples/login/main.go` | Create |
| `examples/tui/main.go` | Modify — use profile.LoadDefault() |
| `examples/simple/main.go` | Modify — gpt-4o → gpt-5.4 |
| `examples/copilot/main.go` | Modify — gpt-4o → gpt-5.4 |

## Verification

1. `go build ./...` — all packages compile
2. `go test ./...` — all tests pass
3. `go vet ./...` — no issues
4. Manual: `go run ./examples/login` → choose "Show status" → shows no providers authenticated
5. Manual: `go run ./examples/login` → choose "Login" → pick copilot → complete device flow
6. Manual: `go run ./examples/login` → choose "Show status" → copilot shows authenticated
7. Manual: `go run ./examples/login` → choose "Browse models" → lists models from copilot
8. Manual: `go run ./examples/login` → choose "Set default" → pick a model from list
9. Manual: `go run ./examples/tui` → launches TUI using the default provider, no env var needed
