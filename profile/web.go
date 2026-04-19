package profile

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/stack-bound/stackllm/auth"
	"github.com/stack-bound/stackllm/config"
)

// SaveAPIKey stores an API key for the named provider directly,
// skipping the Callbacks.OnPromptKey interactive prompt used by Login.
// This is the web / headless counterpart of Login for API-key
// providers (openai, gemini).
func (m *Manager) SaveAPIKey(ctx context.Context, providerName, key string) error {
	if key == "" {
		return fmt.Errorf("profile: SaveAPIKey: empty key for %s", providerName)
	}
	var storeKey string
	switch providerName {
	case ProviderOpenAI:
		storeKey = keyOpenAI
	case ProviderGemini:
		storeKey = keyGemini
	default:
		return fmt.Errorf("profile: SaveAPIKey: %s is not an API-key provider", providerName)
	}
	if err := m.authStore.Save(ctx, storeKey, key); err != nil {
		return fmt.Errorf("profile: save %s key: %w", providerName, err)
	}
	return nil
}

// SaveOllamaURL persists the Ollama base URL. An empty string is
// normalised to the standard http://localhost:11434.
func (m *Manager) SaveOllamaURL(_ context.Context, baseURL string) error {
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}
	cfg, err := m.configStore.Load()
	if err != nil {
		return fmt.Errorf("profile: load config for ollama: %w", err)
	}
	if cfg.Providers == nil {
		cfg.Providers = make(map[string]config.ProviderSettings)
	}
	settings := cfg.Providers[ProviderOllama]
	settings.BaseURL = baseURL
	cfg.Providers[ProviderOllama] = settings
	if err := m.configStore.Save(cfg); err != nil {
		return fmt.Errorf("profile: save ollama config: %w", err)
	}
	return nil
}

// Default returns the currently persisted default model. The bool is
// false when no default has been set yet.
func (m *Manager) Default(_ context.Context) (ModelInfo, bool, error) {
	cfg, err := m.configStore.Load()
	if err != nil {
		return ModelInfo{}, false, fmt.Errorf("profile: load config for default: %w", err)
	}
	if cfg.DefaultProvider == "" || cfg.DefaultModel == "" {
		return ModelInfo{}, false, nil
	}
	return ModelInfo{
		Provider: cfg.DefaultProvider,
		Model:    cfg.DefaultModel,
		Endpoint: cfg.DefaultEndpoint,
	}, true, nil
}

// DeviceFlowState is a snapshot of an in-progress device auth flow.
type DeviceFlowState string

const (
	// DeviceFlowPending means the device code has been issued but
	// the user has not yet completed authorisation.
	DeviceFlowPending DeviceFlowState = "pending"
	// DeviceFlowAuthenticated means the flow completed successfully
	// and the long-lived token has been persisted to the auth store.
	DeviceFlowAuthenticated DeviceFlowState = "authenticated"
	// DeviceFlowError means the flow failed before completing. The
	// reason is available via DeviceFlow.Err.
	DeviceFlowError DeviceFlowState = "error"
)

// DeviceFlow is a handle to an in-progress device auth flow. It is
// safe for concurrent reads from a web handler while the background
// goroutine progresses the flow.
type DeviceFlow struct {
	mu        sync.Mutex
	userCode  string
	verifyURL string
	state     DeviceFlowState
	err       error
	cancel    context.CancelFunc
	// done is closed when the background goroutine exits, whether
	// via success, upstream error, or cancellation. Wait blocks on
	// this so a caller (e.g. web logout) can safely sequence a
	// cancel → credential-clear pair without racing the goroutine's
	// final store.Save.
	done chan struct{}
}

// UserCode returns the device code issued by the provider for the
// user to enter at VerifyURL.
func (d *DeviceFlow) UserCode() string {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.userCode
}

// VerifyURL returns the URL the user should visit to complete the
// flow.
func (d *DeviceFlow) VerifyURL() string {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.verifyURL
}

// State returns the current state of the flow.
func (d *DeviceFlow) State() DeviceFlowState {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.state
}

// Err returns the error that terminated the flow, or nil.
func (d *DeviceFlow) Err() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.err
}

// Cancel aborts the in-progress flow. Safe to call multiple times.
// Does not wait for the background goroutine to exit — pair with
// Wait when the caller needs to sequence cancellation with an
// operation that would race the final store.Save (e.g. logout).
func (d *DeviceFlow) Cancel() {
	d.mu.Lock()
	cancel := d.cancel
	d.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

// Wait blocks until the background goroutine has exited. Returns
// immediately if the flow has already completed. Returns ctx.Err()
// if ctx is cancelled before the goroutine exits.
func (d *DeviceFlow) Wait(ctx context.Context) error {
	select {
	case <-d.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// runDeviceFlow is the shared "start a device flow, wait for the
// user code, then let polling continue in the background" dance used
// by every device-flow login (Copilot, OpenAI). login is the blocking
// auth call that invokes onCode exactly once before entering its poll
// loop. runDeviceFlow returns after onCode fires or login returns an
// error — whichever comes first — and the background goroutine is
// left running to drive polling to completion. onSuccess (if
// non-nil) fires after login returns cleanly, giving callers a hook
// to do post-login work like mirroring the token into another store
// key.
func runDeviceFlow(
	parent context.Context,
	onCodeInstall func(capture func(userCode, verifyURL string)),
	login func(ctx context.Context) error,
	onSuccess func(ctx context.Context) error,
) (*DeviceFlow, error) {
	flow := &DeviceFlow{state: DeviceFlowPending, done: make(chan struct{})}

	codeCh := make(chan struct{}, 1)
	errCh := make(chan error, 1)

	loginCtx, cancel := context.WithCancel(context.Background())
	flow.cancel = cancel

	onCodeInstall(func(userCode, verifyURL string) {
		flow.mu.Lock()
		flow.userCode = userCode
		flow.verifyURL = verifyURL
		flow.mu.Unlock()
		select {
		case codeCh <- struct{}{}:
		default:
		}
	})

	go func() {
		defer close(flow.done)
		err := login(loginCtx)
		if err == nil && onSuccess != nil {
			err = onSuccess(loginCtx)
		}
		flow.mu.Lock()
		if err != nil {
			flow.state = DeviceFlowError
			flow.err = err
		} else {
			flow.state = DeviceFlowAuthenticated
		}
		flow.mu.Unlock()
		select {
		case errCh <- err:
		default:
		}
	}()

	waitCtx := parent
	if _, hasDeadline := parent.Deadline(); !hasDeadline {
		var waitCancel context.CancelFunc
		waitCtx, waitCancel = context.WithTimeout(parent, 60*time.Second)
		defer waitCancel()
	}

	select {
	case <-codeCh:
		return flow, nil
	case err := <-errCh:
		flow.mu.Lock()
		if flow.err == nil {
			flow.err = err
			flow.state = DeviceFlowError
		}
		flow.mu.Unlock()
		if err != nil {
			return flow, fmt.Errorf("profile: device flow: %w", err)
		}
		return flow, nil
	case <-waitCtx.Done():
		cancel()
		flow.mu.Lock()
		flow.state = DeviceFlowError
		flow.err = waitCtx.Err()
		flow.mu.Unlock()
		return flow, fmt.Errorf("profile: device flow: waiting for device code: %w", waitCtx.Err())
	}
}

// BeginCopilotLogin starts the GitHub device flow for Copilot auth in
// a background goroutine and blocks until either the device code is
// available, the flow errors out, or the provided context deadline
// elapses. The returned DeviceFlow can then be polled for completion.
//
// The device flow carries two distinct phases: (1) request a device
// code and show it to the user, (2) poll the access_token endpoint
// until the user completes authorisation. This method returns after
// phase 1 so the web caller can show the code to the end user; phase
// 2 continues in the background and updates the DeviceFlow state.
func (m *Manager) BeginCopilotLogin(parent context.Context) (*DeviceFlow, error) {
	var src *auth.CopilotTokenSource
	return runDeviceFlow(
		parent,
		func(capture func(userCode, verifyURL string)) {
			src = auth.NewCopilotSource(auth.CopilotConfig{
				Store:        m.authStore,
				PollInterval: m.pollInterval,
				HTTPClient:   m.httpClient,
				OnDeviceCode: capture,
				OnPolling:    m.callbacks.OnPolling,
				OnSuccess:    m.callbacks.OnSuccess,
			})
		},
		func(ctx context.Context) error { return src.Login(ctx) },
		nil,
	)
}

// BeginOpenAIDeviceLogin starts the OpenAI OAuth device-code flow
// using the supplied OAuth client ID. It mirrors BeginCopilotLogin:
// blocks until the device code is issued, then drives polling in the
// background.
//
// On success the resulting access token is also mirrored into the
// profile's API-key slot for OpenAI so LoadProvider can use it
// verbatim against the `Authorization: Bearer ...` header. That
// copy is a one-time snapshot — when the token expires the user
// must run the flow again (the refresh path in auth.OpenAIDeviceSource
// is not used by the manager's default OpenAI configuration).
//
// The clientID is supplied by the embedder because OpenAI does not
// publish a single public OAuth client ID for third parties — each
// application registers its own.
func (m *Manager) BeginOpenAIDeviceLogin(parent context.Context, clientID string) (*DeviceFlow, error) {
	if clientID == "" {
		return nil, fmt.Errorf("profile: BeginOpenAIDeviceLogin: client ID is required")
	}

	var src *auth.OpenAIDeviceSource
	return runDeviceFlow(
		parent,
		func(capture func(userCode, verifyURL string)) {
			src = auth.NewOpenAIDeviceSource(auth.OpenAIDeviceConfig{
				ClientID:     clientID,
				Store:        m.authStore,
				PollInterval: m.pollInterval,
				HTTPClient:   m.httpClient,
				OnCode:       capture,
				OnPolling:    m.callbacks.OnPolling,
				OnSuccess:    m.callbacks.OnSuccess,
			})
		},
		func(ctx context.Context) error { return src.Login(ctx) },
		func(ctx context.Context) error {
			tok, err := src.Token(ctx)
			if err != nil {
				return fmt.Errorf("profile: openai device login: read token: %w", err)
			}
			if err := m.authStore.Save(ctx, keyOpenAI, tok.AccessToken); err != nil {
				return fmt.Errorf("profile: openai device login: mirror key: %w", err)
			}
			return nil
		},
	)
}
