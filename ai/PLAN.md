# Go Agent Library — Project Plan

> A minimal, embeddable Go agent orchestration library. Single binary. Full context control.
> Inspired by the Pi agent framework's philosophy: you own the message slice, the framework
> provides the loop and the plumbing.

---

## Background and design decisions

This library replaces a dependency on OpenCode for agent projects. The goals are:

- **Single Go binary** — no Docker sidecar, no external processes
- **Full context control** — the caller owns `[]Message` and decides what goes in it
- **Modular** — a small core with optional add-on packages
- **Multi-provider** — OpenAI, Azure OpenAI, Ollama, GitHub Copilot, Gemini (all via the OpenAI wire format)
- **Auth handled internally** — device flow and web flow OAuth for Copilot and OpenAI subscriptions
- **TUI and web interfaces** — as optional adapter packages

### Key architectural insight

All five target providers speak the OpenAI chat completions wire format:

| Provider | Base URL | Auth |
|---|---|---|
| OpenAI | `api.openai.com` | Static API key |
| Azure OpenAI | `{resource}.openai.azure.com` | API key or Entra token |
| Ollama | `localhost:11434` | None |
| GitHub Copilot | `api.githubcopilot.com` | Two-phase OAuth (see auth section) |
| Gemini | `generativelanguage.googleapis.com/v1beta/openai/` | Google API key |

This means the provider layer is a single implementation parameterised by config — not five separate clients.

---

## Repository structure

```
goagent/                        ← module root, e.g. github.com/yourname/goagent
│
├── provider/                   ← LLM provider (single OpenAI-compat implementation)
│   ├── provider.go             ← Provider interface + Request/Response/Event types
│   ├── openai.go               ← Concrete implementation (all five providers)
│   └── roundtripper.go         ← http.RoundTripper that injects auth + retries
│
├── auth/                       ← Token acquisition and storage
│   ├── token.go                ← Token struct, TokenSource interface, CachingSource
│   ├── store.go                ← TokenStore interface, FileStore, MemoryStore
│   ├── static.go               ← StaticTokenSource (API keys)
│   ├── copilot.go              ← CopilotTokenSource (two-phase GitHub OAuth)
│   ├── openai_device.go        ← OpenAIDeviceSource (headless device code)
│   ├── openai_web.go           ← OpenAIWebFlowSource (PKCE + local callback server)
│   └── keychain/               ← Optional OS keychain backend (build-tagged)
│       └── keychain.go
│
├── conversation/               ← Message types and context management helpers
│   ├── conversation.go         ← Message, Role, ToolCall, ToolResult types
│   ├── builder.go              ← Fluent builder for constructing message slices
│   └── compact.go              ← Token counting and compaction strategies
│
├── tools/                      ← Tool definition, registration, and dispatch
│   ├── tool.go                 ← Tool interface and Definition type
│   ├── registry.go             ← Registry: register, lookup, dispatch
│   ├── schema.go               ← JSON Schema generation from Go structs via reflection
│   └── mcp.go                  ← MCP server adapter (wraps MCP tools as Tool)
│
├── agent/                      ← ReAct agent loop
│   ├── agent.go                ← Agent struct, Run(), Step()
│   ├── options.go              ← Functional options for Agent
│   ├── hooks.go                ← Hooks type (BeforeCall, OnToken, OnToolCall, etc.)
│   └── events.go               ← Event types emitted by the agent loop
│
├── session/                    ← Session and state management
│   ├── session.go              ← Session interface + InMemorySession
│   └── store.go                ← SessionStore interface (pluggable persistence)
│
├── tui/                        ← Bubbletea TUI adapter (optional add-on)
│   ├── model.go                ← Bubbletea Model wrapping an agent.Agent
│   ├── render.go               ← Token streaming, tool call display, spinners
│   └── auth.go                 ← TUI implementations of auth flow callbacks
│
├── web/                        ← HTTP/SSE adapter (optional add-on)
│   ├── handler.go              ← http.Handler: POST /chat, GET /sessions/{id}
│   ├── sse.go                  ← SSE event marshalling
│   └── auth.go                 ← Web implementations of auth flow callbacks
│
├── examples/
│   ├── simple/                 ← Minimal single-agent example
│   ├── copilot/                ← Copilot auth + agent example
│   └── tui/                    ← TUI agent example
│
├── go.mod
├── go.sum
└── README.md
```

---

## Build order (phases)

Implement in this order. Each phase is independently testable before moving on.

### Phase 1 — Foundation types (`conversation/`)

Start here because every other package imports these types.

**`conversation/conversation.go`**

```go
type Role string

const (
    RoleSystem    Role = "system"
    RoleUser      Role = "user"
    RoleAssistant Role = "assistant"
    RoleTool      Role = "tool"
)

type Message struct {
    Role       Role
    Content    string
    ToolCallID string      // set when Role == RoleTool
    ToolCalls  []ToolCall  // set when Role == RoleAssistant with tool calls
}

type ToolCall struct {
    ID        string
    Name      string
    Arguments string // raw JSON
}
```

**`conversation/compact.go`**

Compaction strategies — the caller chooses which to apply before each Step():

```go
// KeepLast returns the last n non-system messages, always preserving system messages.
func KeepLast(msgs []Message, n int) []Message

// TokenBudget drops oldest non-system messages until estimated token count is below limit.
// Uses a simple character/4 heuristic if no tokeniser is provided.
func TokenBudget(msgs []Message, maxTokens int, count func([]Message) int) []Message
```

Do not build a tokeniser — accept a `count func([]Message) int` parameter so callers can
plug in `tiktoken-go` or a model's token counting API.

**Tests:** table-driven tests covering edge cases — empty slice, only system messages, already under budget.

---

### Phase 2 — Auth package (`auth/`)

Build the full auth package before the provider, because the provider depends on `TokenSource`.

**`auth/token.go`**

```go
type Token struct {
    AccessToken string
    ExpiresAt   time.Time // zero value means never expires
}

func (t *Token) Valid() bool {
    if t == nil || t.AccessToken == "" { return false }
    if t.ExpiresAt.IsZero() { return true }
    return time.Now().Before(t.ExpiresAt.Add(-30 * time.Second)) // 30s buffer
}

type TokenSource interface {
    Token(ctx context.Context) (*Token, error)
}

// CachingSource wraps any TokenSource and caches the token until it expires.
// All callers should wrap non-static sources in this.
type CachingSource struct {
    mu      sync.Mutex
    inner   TokenSource
    current *Token
}

func NewCachingSource(inner TokenSource) *CachingSource

func (c *CachingSource) Token(ctx context.Context) (*Token, error)
```

**`auth/store.go`**

```go
type TokenStore interface {
    Load(ctx context.Context, key string) (string, error)
    Save(ctx context.Context, key string, value string) error
    Delete(ctx context.Context, key string) error
}

// FileStore persists tokens as a JSON map at:
//   $XDG_CONFIG_HOME/<appname>/auth.json
//   or ~/.config/<appname>/auth.json on systems without XDG_CONFIG_HOME
type FileStore struct {
    AppName string
    Path    string // override; if empty, uses XDG default
}
```

File is read on every Load (no in-memory cache) to handle multi-process usage.
Write is atomic: write to `.tmp`, then `os.Rename`.

`MemoryStore` is a plain `map[string]string` behind a mutex. Used in tests and
when persistence is not needed.

**`auth/static.go`**

```go
// StaticTokenSource returns the same token every time. Never expires.
// Use for API keys (OpenAI, Gemini, Azure, Ollama).
func NewStatic(token string) TokenSource
```

**`auth/copilot.go`** — the most complex; implement carefully.

```go
// CopilotConfig configures the two-phase Copilot auth flow.
type CopilotConfig struct {
    // Called during Phase 1 to display the one-time code to the user.
    // The implementation must not block — return immediately after
    // displaying. The flow polls in the background.
    OnDeviceCode func(userCode, verifyURL string)

    // Optional: called each time the poll loop checks for authorisation.
    OnPolling func()

    // Optional: called when Phase 1 succeeds.
    OnSuccess func()

    // Token store for persisting the long-lived Phase 1 GitHub token.
    Store TokenStore

    // GitHub host. Defaults to "github.com". Override for GHE.
    Host string
}

// CopilotTokenSource implements the two-phase GitHub → Copilot auth flow:
//
//   Phase 1: GitHub OAuth device flow (RFC 8628)
//     POST https://github.com/login/device/code
//       client_id=Iv1.b507a08c87ecfe98
//       scope=read:user copilot
//     → display user_code + verification_uri
//     Poll POST https://github.com/login/oauth/access_token
//     → ghu_... token (long-lived, persisted via Store)
//
//   Phase 2: Copilot token exchange
//     GET https://api.github.com/copilot_internal/v2/token
//       Authorization: Bearer {ghu_token}
//       Editor-Version: vscode/1.85.0
//       Editor-Plugin-Version: copilot-chat/0.12.0
//       Copilot-Integration-Id: vscode-chat
//       User-Agent: GithubCopilot/1.0
//     → {"token": "...", "expires_at": "..."}  (~30 min TTL)
//     Cached in memory; refreshed automatically before expiry.
//
// If Phase 2 returns 401/403, the Phase 1 token is deleted from the store
// and an error is returned instructing the caller to re-authenticate.
type CopilotTokenSource struct { ... }

func NewCopilotSource(cfg CopilotConfig) *CopilotTokenSource

func (s *CopilotTokenSource) Token(ctx context.Context) (*Token, error)
func (s *CopilotTokenSource) Login(ctx context.Context) error  // force Phase 1 re-auth
func (s *CopilotTokenSource) Logout(ctx context.Context) error // delete stored token
```

Known client ID (do not make this configurable — it is a published constant used
by every third-party Copilot integration): `Iv1.b507a08c87ecfe98`

**`auth/openai_device.go`**

```go
// OpenAIDeviceConfig configures the OpenAI device code flow.
// Use this for headless/SSH environments where a browser cannot be opened.
type OpenAIDeviceConfig struct {
    ClientID string
    Store    TokenStore

    // Called with the user code and verification URL to display to the user.
    OnCode    func(userCode, verifyURL string)
    OnPolling func()
    OnSuccess func()
}

type OpenAIDeviceSource struct { ... }

func NewOpenAIDeviceSource(cfg OpenAIDeviceConfig) *OpenAIDeviceSource
func (s *OpenAIDeviceSource) Token(ctx context.Context) (*Token, error)
func (s *OpenAIDeviceSource) Login(ctx context.Context) error
func (s *OpenAIDeviceSource) Logout(ctx context.Context) error
```

Internally uses `golang.org/x/oauth2` for the device flow protocol and token refresh.

**`auth/openai_web.go`**

```go
// OpenAIWebFlowConfig configures the PKCE authorization code flow.
// Starts a local HTTP server to receive the redirect callback.
type OpenAIWebFlowConfig struct {
    ClientID string
    Port     int    // local callback port, default 1455
    Store    TokenStore

    // Called with the full authorization URL. Implementation should open
    // this in a browser, or display it for the user to open manually.
    OnOpenURL func(authURL string)
    OnSuccess func()
}

type OpenAIWebFlowSource struct { ... }

func NewOpenAIWebFlowSource(cfg OpenAIWebFlowConfig) *OpenAIWebFlowSource
func (s *OpenAIWebFlowSource) Token(ctx context.Context) (*Token, error)
func (s *OpenAIWebFlowSource) Login(ctx context.Context) error
func (s *OpenAIWebFlowSource) Logout(ctx context.Context) error
```

The local callback server listens on `http://localhost:{Port}/callback`, waits for
the OAuth redirect, extracts the code, exchanges it for tokens, then shuts down.
Use a `context.WithTimeout` of 5 minutes for the whole flow.

**Tests:**
- `StaticTokenSource` — trivial
- `CachingSource` — verify it caches, verify it refreshes when expired, verify thread safety
- `FileStore` — write/load/delete, atomic write (simulate crash between write and rename), concurrent access
- `CopilotTokenSource` — mock the GitHub device flow and token exchange HTTP endpoints; verify Phase 2 refresh on expiry; verify Phase 1 re-auth on 401

---

### Phase 3 — Tools (`tools/`)

**`tools/tool.go`**

```go
// Definition is what gets sent to the LLM in the tools array.
type Definition struct {
    Name        string
    Description string
    Parameters  map[string]any // JSON Schema object
}

// Tool is anything that can be called by the agent loop.
type Tool interface {
    Definition() Definition
    Call(ctx context.Context, arguments string) (string, error)
}
```

**`tools/schema.go`** — JSON Schema generation from Go structs

Use `reflect` to generate schemas. Support these struct tags:
- `json:"name"` — field name in schema
- `jsonschema:"description=..."` — field description
- `jsonschema:"required"` — mark field as required
- `jsonschema:"enum=a,b,c"` — enum values

Support these Go types → JSON Schema types:
- `string` → `"type": "string"`
- `int`, `int64`, `float64` → `"type": "number"`
- `bool` → `"type": "boolean"`
- `[]T` → `"type": "array", "items": {schema of T}`
- `struct` → `"type": "object", "properties": {...}`
- Pointer types → same as base type but not required by default

```go
// SchemaOf returns a JSON Schema map for the given value's type.
func SchemaOf(v any) map[string]any
```

**`tools/registry.go`**

```go
type Registry struct { ... }

func NewRegistry() *Registry

// Register adds a function-backed tool. fn must be a func that takes a
// single struct argument and returns (any, error) or (string, error).
// The struct's fields become the tool's JSON Schema parameters.
//
// Example:
//   type ReadFileArgs struct {
//       Path string `json:"path" jsonschema:"description=Absolute path to read,required"`
//   }
//   registry.Register("read_file", "Read a file from disk", func(args ReadFileArgs) (string, error) {
//       data, err := os.ReadFile(args.Path)
//       return string(data), err
//   })
func (r *Registry) Register(name, description string, fn any) error

// Add adds a pre-constructed Tool directly.
func (r *Registry) Add(tool Tool)

// Definitions returns all tool definitions for sending to the LLM.
func (r *Registry) Definitions() []Definition

// Dispatch calls the named tool with the given JSON arguments string.
// Returns the result as a string (JSON-encoded if not already a string).
func (r *Registry) Dispatch(ctx context.Context, name, arguments string) (string, error)
```

`Register` uses `reflect` to:
1. Validate `fn` is a function with the right signature
2. Extract the argument struct type
3. Call `SchemaOf` to generate the JSON Schema
4. Build a `funcTool` that `json.Unmarshal`s arguments into the struct, calls `fn`, and marshals the result

**`tools/mcp.go`**

```go
// MCPTool wraps a single tool from an MCP server as a Tool.
// Requires the MCP server to be running and reachable.
type MCPTool struct {
    ServerURL string
    ToolName  string
    def       Definition // cached from server
}

// NewMCPTool connects to an MCP server and retrieves the named tool's schema.
func NewMCPTool(ctx context.Context, serverURL, toolName string) (*MCPTool, error)

func (t *MCPTool) Definition() Definition
func (t *MCPTool) Call(ctx context.Context, arguments string) (string, error)

// NewMCPRegistry discovers all tools from an MCP server and registers them.
func NewMCPRegistry(ctx context.Context, serverURL string) (*Registry, error)
```

Use `github.com/mark3labs/mcp-go` as the MCP client library.

**Tests:**
- `SchemaOf` — test each supported type, nested structs, pointers, arrays
- `Registry.Register` — verify schema generation, dispatch with valid args, dispatch with invalid args
- `Registry.Dispatch` — unknown tool returns error, tool error propagates

---

### Phase 4 — Provider (`provider/`)

**`provider/provider.go`**

```go
// Request is the input to a provider call.
type Request struct {
    Model       string
    Messages    []conversation.Message
    Tools       []tools.Definition
    MaxTokens   int
    Temperature float64
    Stream      bool  // always true in practice; kept for testing
}

// Event is a single item in the streaming response.
type Event struct {
    Type    EventType
    Content string       // EventTypeToken: the token text
    Call    *ToolCall    // EventTypeToolCall: complete tool call (name + args)
    Err     error        // EventTypeError
}

type EventType int
const (
    EventTypeToken    EventType = iota // streaming text delta
    EventTypeToolCall                  // complete tool call ready for dispatch
    EventTypeDone                      // stream finished, no error
    EventTypeError                     // terminal error
)

type ToolCall = conversation.ToolCall

// Provider makes LLM calls and returns a stream of events.
type Provider interface {
    Complete(ctx context.Context, req Request) (<-chan Event, error)
    // Models returns available model names. Optional — may return nil.
    Models(ctx context.Context) ([]string, error)
}
```

**`provider/openai.go`** — single implementation for all five providers

```go
type Config struct {
    // Required
    BaseURL     string
    TokenSource auth.TokenSource
    Model       string

    // Optional
    APIVersion   string            // Azure requires e.g. "2024-02-01"
    ExtraHeaders map[string]string // static headers added to every request
    HTTPClient   *http.Client      // override for testing
    MaxRetries   int               // default 3; retries on 429 and 5xx
}

// Prebuilt configs for each provider. Callers fill in TokenSource.

func OpenAIConfig(model string, ts auth.TokenSource) Config
func AzureConfig(endpoint, deployment, apiVersion string, ts auth.TokenSource) Config
func OllamaConfig(baseURL, model string) Config  // no auth needed
func CopilotConfig(model string, ts auth.TokenSource) Config
func GeminiConfig(model string, ts auth.TokenSource) Config

type OpenAIProvider struct { cfg Config }

func New(cfg Config) *OpenAIProvider
func (p *OpenAIProvider) Complete(ctx context.Context, req Request) (<-chan Event, error)
func (p *OpenAIProvider) Models(ctx context.Context) ([]string, error)
```

The `Complete` implementation:
1. Calls `cfg.TokenSource.Token(ctx)` to get a fresh bearer token
2. Builds the OpenAI-format request body (map `conversation.Message` → OpenAI messages, map `tools.Definition` → OpenAI tools)
3. Makes a streaming POST with `Accept: text/event-stream`
4. Launches a goroutine that reads SSE chunks and sends `Event`s to the returned channel
5. Closes the channel on done or error

Use `sashabaranov/go-openai` for the request/response types only (to avoid re-implementing the structs). Do the actual HTTP call yourself via `cfg.HTTPClient` so you control streaming, retries, and header injection.

**`provider/roundtripper.go`**

```go
// authRoundTripper injects the Authorization header from a TokenSource
// before each request. Wraps another http.RoundTripper.
type authRoundTripper struct {
    inner  http.RoundTripper
    source auth.TokenSource
}

func (rt *authRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
    tok, err := rt.source.Token(req.Context())
    if err != nil { return nil, err }
    req = req.Clone(req.Context())
    req.Header.Set("Authorization", "Bearer "+tok.AccessToken)
    return rt.inner.RoundTrip(req)
}
```

**Tests:**
- Mock HTTP server returning canned SSE responses
- Verify token is injected correctly
- Verify retry on 429 with backoff
- Verify all five provider configs produce correct base URLs and headers

---

### Phase 5 — Agent loop (`agent/`)

**`agent/hooks.go`**

```go
// Hooks are called at each stage of the agent loop.
// All fields are optional — nil hooks are silently skipped.
type Hooks struct {
    // Called before each LLM request with the current message slice.
    BeforeCall func(ctx context.Context, msgs []conversation.Message)

    // Called for each streaming token as it arrives.
    OnToken func(ctx context.Context, delta string)

    // Called when the LLM emits a complete tool call (before dispatch).
    OnToolCall func(ctx context.Context, call conversation.ToolCall)

    // Called after a tool returns (before appending to conversation).
    OnToolResult func(ctx context.Context, call conversation.ToolCall, result string, err error)

    // Called when the agent loop completes (naturally or via MaxSteps).
    AfterComplete func(ctx context.Context, msgs []conversation.Message)
}
```

**`agent/agent.go`**

```go
type Agent struct {
    provider provider.Provider
    registry *tools.Registry
    hooks    Hooks
    opts     options
}

func New(p provider.Provider, opts ...Option) *Agent

// Step executes one complete LLM round-trip plus tool dispatch.
//
// It takes the current conversation, calls the provider, collects the
// response (assembling streaming tokens into a complete message), dispatches
// any tool calls, and returns the updated conversation plus a StepResult.
//
// The caller is responsible for:
//   - Building and trimming msgs before calling Step
//   - Deciding whether to call Step again based on StepResult
//
// This is the primitive. Use Run for the full ReAct loop.
func (a *Agent) Step(ctx context.Context, msgs []conversation.Message) ([]conversation.Message, StepResult, error)

// Run drives the ReAct loop until one of:
//   - The model returns a final text response with no tool calls
//   - MaxSteps is reached (returns ErrMaxStepsReached)
//   - ctx is cancelled
//
// Events are emitted to the returned channel as the loop progresses.
// The channel is closed when Run returns.
func (a *Agent) Run(ctx context.Context, msgs []conversation.Message) (<-chan Event, error)

// StepResult describes the outcome of a single Step.
type StepResult struct {
    // AssistantMessage is the message to append to the conversation.
    AssistantMessage conversation.Message

    // ToolResults are ready to append after AssistantMessage.
    // Empty if the model produced no tool calls.
    ToolResults []conversation.Message

    // Done is true if the model produced a final response (no tool calls).
    Done bool
}
```

**Canonical Step() implementation:**

```
1. Call hooks.BeforeCall(ctx, msgs)
2. Call provider.Complete(ctx, Request{Messages: msgs, Tools: registry.Definitions(), ...})
3. Read events from channel:
   - EventTypeToken → hooks.OnToken(ctx, delta); accumulate content
   - EventTypeToolCall → hooks.OnToolCall(ctx, call); accumulate calls
   - EventTypeDone → break
   - EventTypeError → return error
4. Build AssistantMessage from accumulated content + tool calls
5. For each tool call:
   a. hooks.OnToolCall(ctx, call)  [if not already called during streaming]
   b. result, err := registry.Dispatch(ctx, call.Name, call.Arguments)
   c. hooks.OnToolResult(ctx, call, result, err)
   d. Build ToolResult message (RoleTool, content=result or error string, ToolCallID=call.ID)
6. Return updated msgs + StepResult
```

**`agent/options.go`**

```go
type Option func(*options)

type options struct {
    maxSteps    int            // default 20
    model       string         // overrides provider default
    temperature float64
    maxTokens   int
    hooks       Hooks
    registry    *tools.Registry
}

func WithMaxSteps(n int) Option
func WithModel(model string) Option
func WithTemperature(t float64) Option
func WithMaxTokens(n int) Option
func WithHooks(h Hooks) Option
func WithTools(r *tools.Registry) Option
```

**Tests:**
- `Step` with mock provider returning text only — verify message appended correctly
- `Step` with mock provider returning one tool call — verify dispatch called, result appended
- `Step` with dispatch returning error — verify error message appended, not propagated
- `Run` driving three steps to completion — verify correct conversation evolution
- `Run` hitting MaxSteps — verify `ErrMaxStepsReached` returned

---

### Phase 6 — Session (`session/`)

```go
// Session holds conversation history and arbitrary KV state for an agent run.
type Session struct {
    ID       string
    Messages []conversation.Message
    State    map[string]any
    Created  time.Time
    Updated  time.Time
}

// SessionStore persists sessions.
type SessionStore interface {
    Save(ctx context.Context, s *Session) error
    Load(ctx context.Context, id string) (*Session, error)
    Delete(ctx context.Context, id string) error
    List(ctx context.Context) ([]*Session, error)
}

// InMemoryStore is the default. Not persistent across restarts.
func NewInMemoryStore() SessionStore
```

Keep this simple in Phase 6. File-backed and Redis-backed stores come later.

---

### Phase 7 — TUI adapter (`tui/`)

Dependencies: `github.com/charmbracelet/bubbletea`, `github.com/charmbracelet/lipgloss`

```go
// Model is a Bubbletea model that drives an agent.Agent interactively.
type Model struct {
    agent    *agent.Agent
    session  *session.Session
    store    session.SessionStore
    input    string
    events   <-chan agent.Event
    viewport viewport.Model  // charmbracelet/bubbles viewport
    spinner  spinner.Model
    state    modelState
}

type modelState int
const (
    stateIdle modelState = iota
    stateRunning
    stateToolCall
    stateError
)

func New(a *agent.Agent, store session.SessionStore) *Model

// AuthHooks returns agent.Hooks configured to render auth prompts in the TUI.
// Pass these to agent.New via WithHooks when using the TUI adapter.
func AuthHooks() agent.Hooks
```

The `AuthHooks()` function returns a `Hooks` struct where `OnToolCall` renders tool
call spinners inline, `OnToken` appends streamed text, and the Copilot/OpenAI
`OnDeviceCode` / `OnOpenURL` callbacks render the auth prompt as a styled box
in the terminal.

---

### Phase 8 — Web/SSE adapter (`web/`)

```go
// Handler exposes an agent over HTTP with SSE streaming.
type Handler struct {
    agent   *agent.Agent
    store   session.SessionStore
    mux     *http.ServeMux
}

func NewHandler(a *agent.Agent, store session.SessionStore) *Handler

// ServeHTTP implements http.Handler.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request)
```

**Routes:**

```
POST   /chat                    Start or continue a session
                                Body: {"session_id": "...", "message": "..."}
                                Responds with SSE stream

GET    /sessions/{id}           Retrieve session history
DELETE /sessions/{id}           Delete session

GET    /auth/copilot/start      Returns JSON with user_code + verify_url
GET    /auth/copilot/status     Polls until authed; returns token status
GET    /auth/openai/start       Initiates web flow; returns redirect URL
GET    /auth/openai/callback    OAuth callback (registered as redirect URI)
```

**SSE event types (streamed from POST /chat):**

```
event: token
data: {"delta": "Hello"}

event: tool_call
data: {"id": "...", "name": "read_file", "arguments": "{\"path\":\"/tmp/foo\"}"}

event: tool_result
data: {"id": "...", "result": "file contents here"}

event: done
data: {"session_id": "abc123"}

event: error
data: {"message": "context deadline exceeded"}
```

---

## Dependencies

Pin these in go.mod. Do not add others without strong justification.

| Package | Purpose |
|---|---|
| `github.com/sashabaranov/go-openai` | OpenAI request/response struct types only |
| `golang.org/x/oauth2` | OAuth 2.0 protocol for device flow and web flow |
| `github.com/charmbracelet/bubbletea` | TUI framework (tui/ package only) |
| `github.com/charmbracelet/bubbles` | TUI components: viewport, spinner, textarea |
| `github.com/charmbracelet/lipgloss` | TUI styling |
| `github.com/mark3labs/mcp-go` | MCP client (tools/mcp.go only) |

Standard library covers everything else: `net/http`, `encoding/json`, `sync`, `context`, `reflect`, `os`, `time`.

Do **not** add: LangChain, Eino, Google ADK, or any other agent framework. The point of this library is to own the primitives.

---

## Code conventions

- **Errors:** always `fmt.Errorf("package: action: %w", err)`. No `errors.New` with format strings.
- **Context:** every exported function that does I/O takes `ctx context.Context` as first argument.
- **Interfaces:** define interfaces in the package that *uses* them, not the package that implements them (Go convention).
- **No global state:** no package-level variables except constants. Everything via dependency injection.
- **Functional options:** all structs with more than two config fields use the `Option` pattern.
- **No init():** never.
- **Tests:** table-driven where there are multiple cases. Use `httptest.NewServer` for provider tests. Use `t.Parallel()` on all tests that don't share state.
- **File names:** one primary type or concept per file. Match file name to primary export.

---

## What Claude Code should do first

When starting a new session with this plan:

1. `go mod init github.com/yourname/goagent` (replace with actual module path)
2. Implement `conversation/` package completely with tests passing
3. Implement `auth/token.go`, `auth/store.go`, `auth/static.go` with tests passing
4. Implement `auth/copilot.go` with mocked HTTP tests passing
5. Continue through phases in order

Do not skip ahead. Each phase depends on the previous. Tests must pass before moving on.

---

## Open questions to resolve before or during Phase 1

1. **Module path** — what is the Go module path? (`github.com/yourname/goagent` is a placeholder)
2. **App name for auth storage** — what name should `FileStore` use for the config directory?
3. **Copilot client ID** — the known VSCode client ID `Iv1.b507a08c87ecfe98` is widely used by third-party tools but is not officially published by GitHub. Confirm this is acceptable or investigate registering your own OAuth app.
4. **OpenAI subscription client ID** — the Codex CLI client ID is embedded in that binary. You will likely need to register your own OAuth app with OpenAI to get a client ID for the web/device flows.
5. **TUI library** — confirmed Bubbletea? Alternatives: `tview`, raw `tcell`.
6. **Token counting** — will you ship a default tokeniser (e.g. `tiktoken-go`) or leave it as a caller-supplied function? The latter keeps the core dependency-free.
