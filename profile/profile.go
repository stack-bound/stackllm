// Package profile composes auth, config, and provider into a single
// entry point for provider management. It handles login, logout, provider
// selection, model listing, and default persistence — usable across
// projects with zero boilerplate.
package profile

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"time"
	"strings"
	"sync"

	"github.com/stack-bound/stackllm/auth"
	"github.com/stack-bound/stackllm/config"
	"github.com/stack-bound/stackllm/provider"
)

// Provider name constants.
const (
	ProviderOpenAI  = "openai"
	ProviderCopilot = "copilot"
	ProviderGemini  = "gemini"
	ProviderOllama  = "ollama"
)

// Auth store keys — must match the keys used by the auth package.
const (
	keyOpenAI        = "openai_api_key"
	keyCopilotGitHub = "copilot_github_token"
	keyGemini        = "gemini_api_key"
)

// allProviders is the canonical ordering.
var allProviders = []string{ProviderOpenAI, ProviderCopilot, ProviderGemini, ProviderOllama}

// Callbacks lets callers inject UI behaviour for interactive flows.
type Callbacks struct {
	// OnDeviceCode is called during Copilot login with the one-time code and URL.
	OnDeviceCode func(userCode, verifyURL string)

	// OnPolling is called each time the device flow polls for authorisation.
	OnPolling func()

	// OnSuccess is called when a device flow completes successfully.
	OnSuccess func()

	// OnPromptKey prompts the user for an API key (e.g. OpenAI, Gemini).
	OnPromptKey func(providerName string) (string, error)

	// OnPromptURL prompts the user for a base URL (e.g. Ollama).
	OnPromptURL func(providerName, defaultURL string) (string, error)
}

// ProviderStatus describes the authentication state of a single provider.
type ProviderStatus struct {
	Name          string
	Authenticated bool
	IsDefault     bool
}

// ModelInfo identifies a model scoped to its provider.
type ModelInfo struct {
	Provider string // e.g. "copilot"
	Model    string // e.g. "gpt-5.4"

	// Endpoint is the API path the model is reachable on. The empty
	// string means the provider's default (/chat/completions).
	// provider.EndpointResponses ("/responses") is used for Copilot
	// models that are not accessible via /chat/completions, e.g.
	// gpt-5.4-mini and gpt-5.x-codex.
	//
	// Populated by ListAllModels from each model's supported_endpoints
	// metadata. LoadProviderForModel routes the request accordingly.
	Endpoint string
}

// String returns "provider/model" format.
func (m ModelInfo) String() string { return m.Provider + "/" + m.Model }

// Manager is the single entry point for provider management. It wires
// together auth, config, and provider packages. No global state.
type Manager struct {
	authStore    auth.TokenStore
	configStore  *config.Store
	callbacks    Callbacks
	httpClient   *http.Client
	pollInterval time.Duration
}

// Option configures a Manager.
type Option func(*Manager)

// WithHTTPClient sets the HTTP client for provider and auth operations.
func WithHTTPClient(c *http.Client) Option {
	return func(m *Manager) { m.httpClient = c }
}

// WithCallbacks sets the interactive callbacks.
func WithCallbacks(cb Callbacks) Option {
	return func(m *Manager) { m.callbacks = cb }
}

// WithAuthStore overrides the token store (useful for testing).
func WithAuthStore(s auth.TokenStore) Option {
	return func(m *Manager) { m.authStore = s }
}

// WithConfigStore overrides the config store (useful for testing).
func WithConfigStore(s *config.Store) Option {
	return func(m *Manager) { m.configStore = s }
}

// WithPollInterval overrides the device-flow poll interval (useful for testing).
func WithPollInterval(d time.Duration) Option {
	return func(m *Manager) { m.pollInterval = d }
}

// New creates a Manager with the given options.
// By default it uses auth.FileStore and config.Store with AppName "stackllm".
func New(opts ...Option) *Manager {
	m := &Manager{
		authStore:   &auth.FileStore{AppName: "stackllm"},
		configStore: &config.Store{AppName: "stackllm"},
	}
	for _, o := range opts {
		o(m)
	}
	return m
}

// AvailableProviders returns the list of supported provider names.
func (m *Manager) AvailableProviders() []string {
	out := make([]string, len(allProviders))
	copy(out, allProviders)
	return out
}

// Login runs the authentication flow for the named provider.
func (m *Manager) Login(ctx context.Context, providerName string) error {
	switch providerName {
	case ProviderOpenAI:
		return m.loginAPIKey(ctx, ProviderOpenAI, keyOpenAI)
	case ProviderGemini:
		return m.loginAPIKey(ctx, ProviderGemini, keyGemini)
	case ProviderCopilot:
		return m.loginCopilot(ctx)
	case ProviderOllama:
		return m.loginOllama(ctx)
	default:
		return fmt.Errorf("profile: unknown provider %q", providerName)
	}
}

func (m *Manager) loginAPIKey(ctx context.Context, name, storeKey string) error {
	if m.callbacks.OnPromptKey == nil {
		return fmt.Errorf("profile: OnPromptKey callback required for %s login", name)
	}
	key, err := m.callbacks.OnPromptKey(name)
	if err != nil {
		return fmt.Errorf("profile: prompt key for %s: %w", name, err)
	}
	if key == "" {
		return fmt.Errorf("profile: empty API key for %s", name)
	}
	if err := m.authStore.Save(ctx, storeKey, key); err != nil {
		return fmt.Errorf("profile: save %s key: %w", name, err)
	}
	return nil
}

func (m *Manager) loginCopilot(ctx context.Context) error {
	src := auth.NewCopilotSource(auth.CopilotConfig{
		Store:        m.authStore,
		OnDeviceCode: m.callbacks.OnDeviceCode,
		OnPolling:    m.callbacks.OnPolling,
		OnSuccess:    m.callbacks.OnSuccess,
		PollInterval: m.pollInterval,
		HTTPClient:   m.httpClient,
	})
	return src.Login(ctx)
}

func (m *Manager) loginOllama(ctx context.Context) error {
	if m.callbacks.OnPromptURL == nil {
		return fmt.Errorf("profile: OnPromptURL callback required for ollama login")
	}
	urlStr, err := m.callbacks.OnPromptURL(ProviderOllama, "http://localhost:11434")
	if err != nil {
		return fmt.Errorf("profile: prompt URL for ollama: %w", err)
	}
	if urlStr == "" {
		urlStr = "http://localhost:11434"
	}

	cfg, err := m.configStore.Load()
	if err != nil {
		return fmt.Errorf("profile: load config for ollama: %w", err)
	}
	if cfg.Providers == nil {
		cfg.Providers = make(map[string]config.ProviderSettings)
	}
	settings := cfg.Providers[ProviderOllama]
	settings.BaseURL = urlStr
	cfg.Providers[ProviderOllama] = settings

	if err := m.configStore.Save(cfg); err != nil {
		return fmt.Errorf("profile: save ollama config: %w", err)
	}
	_ = ctx // context not needed for config write
	return nil
}

// Logout clears stored credentials for the named provider.
func (m *Manager) Logout(ctx context.Context, providerName string) error {
	switch providerName {
	case ProviderOpenAI:
		return m.authStore.Delete(ctx, keyOpenAI)
	case ProviderGemini:
		return m.authStore.Delete(ctx, keyGemini)
	case ProviderCopilot:
		return m.authStore.Delete(ctx, keyCopilotGitHub)
	case ProviderOllama:
		cfg, err := m.configStore.Load()
		if err != nil {
			return fmt.Errorf("profile: load config for ollama logout: %w", err)
		}
		delete(cfg.Providers, ProviderOllama)
		return m.configStore.Save(cfg)
	default:
		return fmt.Errorf("profile: unknown provider %q", providerName)
	}
}

// Status returns the authentication state of all providers.
func (m *Manager) Status(ctx context.Context) ([]ProviderStatus, error) {
	cfg, err := m.configStore.Load()
	if err != nil {
		return nil, fmt.Errorf("profile: load config for status: %w", err)
	}

	statuses := make([]ProviderStatus, len(allProviders))
	for i, name := range allProviders {
		statuses[i] = ProviderStatus{
			Name:          name,
			Authenticated: m.isAuthenticated(ctx, name, cfg),
			IsDefault:     name == cfg.DefaultProvider,
		}
	}
	return statuses, nil
}

func (m *Manager) isAuthenticated(ctx context.Context, name string, cfg *config.Config) bool {
	switch name {
	case ProviderOpenAI:
		_, err := m.authStore.Load(ctx, keyOpenAI)
		return err == nil
	case ProviderGemini:
		_, err := m.authStore.Load(ctx, keyGemini)
		return err == nil
	case ProviderCopilot:
		_, err := m.authStore.Load(ctx, keyCopilotGitHub)
		return err == nil
	case ProviderOllama:
		if cfg.Providers == nil {
			return false
		}
		settings, ok := cfg.Providers[ProviderOllama]
		return ok && settings.BaseURL != ""
	default:
		return false
	}
}

// ListModels returns available chat model IDs for a single provider.
// The provider must be authenticated. Embedding-only models are
// filtered out.
func (m *Manager) ListModels(ctx context.Context, providerName string) ([]string, error) {
	infos, err := m.listModelsForProvider(ctx, providerName)
	if err != nil {
		return nil, err
	}
	out := make([]string, len(infos))
	for i, info := range infos {
		out[i] = info.Model
	}
	return out, nil
}

// listModelsForProvider returns ModelInfo entries for a single provider's
// chat-capable models, with each entry's Endpoint populated from the
// model's metadata. Embedding-only models are filtered out.
func (m *Manager) listModelsForProvider(ctx context.Context, providerName string) ([]ModelInfo, error) {
	p, err := m.buildProvider(ctx, providerName, "", provider.EndpointChatCompletions)
	if err != nil {
		return nil, fmt.Errorf("profile: build provider for models: %w", err)
	}
	metas, err := p.Models(ctx)
	if err != nil {
		return nil, fmt.Errorf("profile: list models for %s: %w", providerName, err)
	}

	out := make([]ModelInfo, 0, len(metas))
	for _, meta := range metas {
		if meta.Type == "embeddings" {
			continue
		}
		// Drop entries the upstream has explicitly disabled in its
		// model picker (Copilot routers, pinned legacy versions,
		// internal load-test models, etc.). Providers that don't
		// expose model_picker_enabled (OpenAI, Gemini, Ollama)
		// leave the field nil and are unaffected.
		if meta.ModelPickerEnabled != nil && !*meta.ModelPickerEnabled {
			continue
		}
		out = append(out, ModelInfo{
			Provider: providerName,
			Model:    meta.ID,
			Endpoint: endpointForMeta(meta),
		})
	}
	return out, nil
}

// endpointForMeta selects the API endpoint a model should be called on,
// based on its SupportedEndpoints metadata.
//
// Rules:
//   - SupportedEndpoints absent or empty → default (chat completions).
//     Most providers (OpenAI, Gemini, Ollama) and many legacy Copilot
//     models do not expose this metadata.
//   - Contains "/chat/completions" → default (chat completions). Models
//     with both endpoints prefer the legacy path for compatibility.
//   - Contains "/responses" but not "/chat/completions" → responses.
//     This is the case for gpt-5.4-mini and gpt-5.x-codex on Copilot.
//   - Otherwise (some other unknown endpoint set) → default and let
//     the upstream API surface a clear error.
func endpointForMeta(meta provider.ModelMeta) string {
	if len(meta.SupportedEndpoints) == 0 {
		return provider.EndpointChatCompletions
	}
	hasResponses := false
	for _, ep := range meta.SupportedEndpoints {
		if ep == "/chat/completions" {
			return provider.EndpointChatCompletions
		}
		if ep == "/responses" {
			hasResponses = true
		}
	}
	if hasResponses {
		return provider.EndpointResponses
	}
	return provider.EndpointChatCompletions
}

// ListAllModels queries all authenticated providers for models and returns
// a combined list sorted by provider then model name. Each ModelInfo has
// its Endpoint populated from the upstream metadata so callers can pass
// it to LoadProviderForModel without re-querying. Errors from individual
// providers are skipped (e.g. Ollama offline), not fatal.
func (m *Manager) ListAllModels(ctx context.Context) ([]ModelInfo, error) {
	statuses, err := m.Status(ctx)
	if err != nil {
		return nil, err
	}

	var wg sync.WaitGroup
	ch := make(chan []ModelInfo, len(statuses))

	for _, s := range statuses {
		if !s.Authenticated {
			continue
		}
		wg.Add(1)
		go func(name string) {
			defer wg.Done()
			infos, err := m.listModelsForProvider(ctx, name)
			if err != nil {
				// Skip providers that fail (e.g. offline Ollama).
				return
			}
			ch <- infos
		}(s.Name)
	}

	wg.Wait()
	close(ch)

	var all []ModelInfo
	for batch := range ch {
		all = append(all, batch...)
	}

	sort.Slice(all, func(i, j int) bool {
		if all[i].Provider != all[j].Provider {
			return all[i].Provider < all[j].Provider
		}
		return all[i].Model < all[j].Model
	})

	return all, nil
}

// RecentModels returns the persisted list of models the user has
// recently selected, most-recent-first. Returns an empty slice when
// no recents have been recorded yet (not an error). The Endpoint
// field is preserved so callers can pass entries straight to
// LoadProviderForModel.
func (m *Manager) RecentModels(ctx context.Context) ([]ModelInfo, error) {
	cfg, err := m.configStore.Load()
	if err != nil {
		return nil, fmt.Errorf("profile: load config for recent models: %w", err)
	}
	out := make([]ModelInfo, 0, len(cfg.RecentModels))
	for _, r := range cfg.RecentModels {
		if r.Provider == "" || r.Model == "" {
			continue
		}
		out = append(out, ModelInfo{
			Provider: r.Provider,
			Model:    r.Model,
			Endpoint: r.Endpoint,
		})
	}
	_ = ctx // config is local
	return out, nil
}

// TrackRecentModel pushes the supplied model to the front of the
// recent-models list and dedupes any prior occurrence by
// provider+model. The list is unbounded — every model the user has
// ever selected is retained, with the most recent at the top.
// Persists the result.
//
// Endpoint is overwritten when the same provider/model is re-added,
// so a model that has been routed via /responses on its most recent
// use will retain that routing for the next session.
func (m *Manager) TrackRecentModel(ctx context.Context, info ModelInfo) error {
	if info.Provider == "" || info.Model == "" {
		return fmt.Errorf("profile: TrackRecentModel: provider and model required")
	}
	cfg, err := m.configStore.Load()
	if err != nil {
		return fmt.Errorf("profile: load config for track recent: %w", err)
	}

	entry := config.RecentModel{
		Provider: info.Provider,
		Model:    info.Model,
		Endpoint: info.Endpoint,
	}

	updated := make([]config.RecentModel, 0, len(cfg.RecentModels)+1)
	updated = append(updated, entry)
	for _, r := range cfg.RecentModels {
		if r.Provider == entry.Provider && r.Model == entry.Model {
			continue
		}
		updated = append(updated, r)
	}
	cfg.RecentModels = updated

	if err := m.configStore.Save(cfg); err != nil {
		return fmt.Errorf("profile: save recent model: %w", err)
	}
	_ = ctx // config is local
	return nil
}

// SetDefault parses "provider/model" format and persists the choice.
//
// The persisted endpoint is cleared (defaults to chat completions on
// load). Use SetDefaultModel to persist a model that requires a
// non-default endpoint such as /responses.
func (m *Manager) SetDefault(providerSlashModel string) error {
	parts := strings.SplitN(providerSlashModel, "/", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return fmt.Errorf("profile: invalid default format %q, expected provider/model", providerSlashModel)
	}
	return m.SetDefaultModel(ModelInfo{Provider: parts[0], Model: parts[1]})
}

// SetDefaultModel persists provider, model, and endpoint from a ModelInfo.
// This is the preferred entry point for code that already has the
// endpoint metadata in hand (e.g. interactive pickers calling
// ListAllModels) — it preserves the endpoint so LoadDefault routes
// requests correctly on the next run.
func (m *Manager) SetDefaultModel(info ModelInfo) error {
	if info.Provider == "" || info.Model == "" {
		return fmt.Errorf("profile: SetDefaultModel: provider and model required")
	}

	valid := false
	for _, p := range allProviders {
		if p == info.Provider {
			valid = true
			break
		}
	}
	if !valid {
		return fmt.Errorf("profile: unknown provider %q", info.Provider)
	}

	cfg, err := m.configStore.Load()
	if err != nil {
		return fmt.Errorf("profile: load config for set default: %w", err)
	}
	cfg.DefaultProvider = info.Provider
	cfg.DefaultModel = info.Model
	cfg.DefaultEndpoint = info.Endpoint
	if err := m.configStore.Save(cfg); err != nil {
		return fmt.Errorf("profile: save default: %w", err)
	}
	return nil
}

// LoadDefault returns a ready-to-use provider from the persisted default config.
// Honours the persisted DefaultEndpoint so a default of e.g.
// copilot/gpt-5.4-mini built via SetDefaultModel correctly routes to
// /responses on the next run.
func (m *Manager) LoadDefault(ctx context.Context) (*provider.OpenAIProvider, error) {
	cfg, err := m.configStore.Load()
	if err != nil {
		return nil, fmt.Errorf("profile: load config: %w", err)
	}
	if cfg.DefaultProvider == "" || cfg.DefaultModel == "" {
		return nil, fmt.Errorf("profile: no default provider/model set (run: go run ./examples/login)")
	}
	return m.LoadProviderForModel(ctx, ModelInfo{
		Provider: cfg.DefaultProvider,
		Model:    cfg.DefaultModel,
		Endpoint: cfg.DefaultEndpoint,
	})
}

// LoadProvider returns a ready-to-use provider for the given name and
// model on the provider's default endpoint (/chat/completions). Use
// LoadProviderForModel when you need to target /responses for a Copilot
// responses-only model.
func (m *Manager) LoadProvider(ctx context.Context, providerName, model string) (*provider.OpenAIProvider, error) {
	return m.buildProvider(ctx, providerName, model, provider.EndpointChatCompletions)
}

// LoadProviderForModel returns a ready-to-use provider configured for
// the supplied ModelInfo, including its Endpoint. This is the entry
// point that interactive pickers should use after ListAllModels —
// it correctly routes Copilot responses-only models to /responses.
func (m *Manager) LoadProviderForModel(ctx context.Context, info ModelInfo) (*provider.OpenAIProvider, error) {
	return m.buildProvider(ctx, info.Provider, info.Model, info.Endpoint)
}

func (m *Manager) buildProvider(ctx context.Context, providerName, model, endpoint string) (*provider.OpenAIProvider, error) {
	switch providerName {
	case ProviderOpenAI:
		key, err := m.authStore.Load(ctx, keyOpenAI)
		if err != nil {
			return nil, fmt.Errorf("profile: openai not authenticated: %w", err)
		}
		ts := auth.NewStatic(key)
		cfg := provider.OpenAIConfig(model, ts)
		cfg.Endpoint = endpoint
		if m.httpClient != nil {
			cfg.HTTPClient = m.httpClient
		}
		return provider.New(cfg), nil

	case ProviderCopilot:
		src := auth.NewCopilotSource(auth.CopilotConfig{
			Store:        m.authStore,
			OnDeviceCode: m.callbacks.OnDeviceCode,
			OnPolling:    m.callbacks.OnPolling,
			OnSuccess:    m.callbacks.OnSuccess,
			PollInterval: m.pollInterval,
			HTTPClient:   m.httpClient,
		})
		ts := auth.NewCachingSource(src)
		cfg := provider.CopilotConfig(model, ts)
		cfg.Endpoint = endpoint
		if m.httpClient != nil {
			cfg.HTTPClient = m.httpClient
		}
		return provider.New(cfg), nil

	case ProviderGemini:
		key, err := m.authStore.Load(ctx, keyGemini)
		if err != nil {
			return nil, fmt.Errorf("profile: gemini not authenticated: %w", err)
		}
		ts := auth.NewStatic(key)
		cfg := provider.GeminiConfig(model, ts)
		cfg.Endpoint = endpoint
		if m.httpClient != nil {
			cfg.HTTPClient = m.httpClient
		}
		return provider.New(cfg), nil

	case ProviderOllama:
		cfgData, err := m.configStore.Load()
		if err != nil {
			return nil, fmt.Errorf("profile: load config for ollama: %w", err)
		}
		baseURL := "http://localhost:11434"
		if cfgData.Providers != nil {
			if settings, ok := cfgData.Providers[ProviderOllama]; ok && settings.BaseURL != "" {
				baseURL = settings.BaseURL
			}
		}
		cfg := provider.OllamaConfig(baseURL, model)
		cfg.Endpoint = endpoint
		if m.httpClient != nil {
			cfg.HTTPClient = m.httpClient
		}
		return provider.New(cfg), nil

	default:
		return nil, fmt.Errorf("profile: unknown provider %q", providerName)
	}
}
