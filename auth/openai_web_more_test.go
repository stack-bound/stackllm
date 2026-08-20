package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"
)

var (
	allocatedPortsMu sync.Mutex
	allocatedPorts   = map[int]bool{}
)

// freePort grabs an ephemeral TCP port from the kernel and releases it
// so a Login callback listener can bind it. A port is never handed out
// twice within this test binary: after freePort closes its probe
// listener the kernel may immediately reuse the port for a concurrent
// freePort call, and two parallel web-flow tests sharing a port
// cross-talk through each other's /callback handlers (a foreign GET
// carries the wrong state, which aborts the victim's Login). A stolen
// port from outside the process remains possible but only fails the
// bind, which awaitAuthURL surfaces fast.
func freePort(t *testing.T) int {
	t.Helper()
	allocatedPortsMu.Lock()
	defer allocatedPortsMu.Unlock()
	for range 100 {
		l, err := net.Listen("tcp", "localhost:0")
		if err != nil {
			t.Fatalf("freePort: %v", err)
		}
		port := l.Addr().(*net.TCPAddr).Port
		l.Close()
		if !allocatedPorts[port] {
			allocatedPorts[port] = true
			return port
		}
	}
	t.Fatal("freePort: no unused port after 100 attempts")
	return 0
}

// awaitAuthURL waits for Login to publish its authorization URL,
// failing fast if Login exits first (e.g. its callback port was taken
// between freePort and the listener bind) instead of deadlocking on
// the URL channel.
func awaitAuthURL(t *testing.T, urlCh <-chan string, loginErr <-chan error) string {
	t.Helper()
	select {
	case u := <-urlCh:
		return u
	case err := <-loginErr:
		t.Fatalf("Login exited before publishing auth URL: %v", err)
	case <-time.After(30 * time.Second):
		t.Fatal("timed out waiting for auth URL")
	}
	return ""
}

func TestOpenAIWebFlowConfig_Port(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		cfg  OpenAIWebFlowConfig
		want int
	}{
		{"default", OpenAIWebFlowConfig{}, defaultCallbackPort},
		{"override", OpenAIWebFlowConfig{Port: 9123}, 9123},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := tt.cfg.port(); got != tt.want {
				t.Errorf("port() = %d, want %d", got, tt.want)
			}
		})
	}
}

// driveCallback issues the OAuth redirect GET that a browser would make
// to the local callback listener, retrying briefly in case the listener
// is not accepting yet.
func driveCallback(t *testing.T, rawURL string) *http.Response {
	t.Helper()
	var resp *http.Response
	var err error
	for i := 0; i < 50; i++ {
		resp, err = http.Get(rawURL)
		if err == nil {
			return resp
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("callback GET %s: %v", rawURL, err)
	return nil
}

func TestOpenAIWebFlowSource_LoginEndToEnd(t *testing.T) {
	t.Parallel()

	tokenServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Errorf("ParseForm: %v", err)
		}
		if r.Form.Get("grant_type") != "authorization_code" {
			t.Errorf("grant_type = %q, want authorization_code", r.Form.Get("grant_type"))
		}
		if r.Form.Get("code") != "browser-code" {
			t.Errorf("code = %q, want browser-code", r.Form.Get("code"))
		}
		if r.Form.Get("code_verifier") == "" {
			t.Error("expected code_verifier")
		}
		json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "login-access",
			"refresh_token": "login-refresh",
			"expires_in":    3600,
		})
	}))
	defer tokenServer.Close()

	port := freePort(t)
	urlCh := make(chan string, 1)
	successCh := make(chan struct{}, 1)
	store := NewMemoryStore()

	src := NewOpenAIWebFlowSource(OpenAIWebFlowConfig{
		ClientID:   "web-client",
		Port:       port,
		Store:      store,
		OnOpenURL:  func(u string) { urlCh <- u },
		OnSuccess:  func() { successCh <- struct{}{} },
		HTTPClient: tokenServer.Client(),
		AuthURL:    "https://test.openai.local/authorize",
		TokenURL:   tokenServer.URL,
	})

	loginErr := make(chan error, 1)
	go func() { loginErr <- src.Login(context.Background()) }()

	authURL := awaitAuthURL(t, urlCh, loginErr)
	parsed, err := url.Parse(authURL)
	if err != nil {
		t.Fatalf("parse authURL: %v", err)
	}
	q := parsed.Query()
	if q.Get("client_id") != "web-client" {
		t.Errorf("client_id = %q", q.Get("client_id"))
	}
	if q.Get("code_challenge_method") != "S256" {
		t.Errorf("code_challenge_method = %q", q.Get("code_challenge_method"))
	}
	wantRedirect := fmt.Sprintf("http://localhost:%d/callback", port)
	if q.Get("redirect_uri") != wantRedirect {
		t.Errorf("redirect_uri = %q, want %q", q.Get("redirect_uri"), wantRedirect)
	}
	// The code_challenge must be the S256 hash of the verifier the
	// source generated — we can't see the verifier directly, but the
	// challenge must be present and non-empty base64url.
	if q.Get("code_challenge") == "" {
		t.Error("missing code_challenge")
	}
	state := q.Get("state")
	if state == "" {
		t.Fatal("missing state")
	}

	resp := driveCallback(t, fmt.Sprintf("%s?state=%s&code=browser-code", wantRedirect, url.QueryEscape(state)))
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("callback status = %d, body %s", resp.StatusCode, body)
	}
	if !strings.Contains(string(body), "Authentication successful") {
		t.Errorf("callback body = %q, want success page", body)
	}

	if err := <-loginErr; err != nil {
		t.Fatalf("Login: %v", err)
	}
	select {
	case <-successCh:
	default:
		t.Error("expected OnSuccess callback")
	}

	// The exchanged token must be persisted with its expiry honoured
	// (expires_in 3600 → ExpiresAt roughly one hour out).
	record, err := loadOpenAITokenRecord(context.Background(), store, openaiWebStoreKey)
	if err != nil {
		t.Fatalf("loadOpenAITokenRecord: %v", err)
	}
	if record.AccessToken != "login-access" {
		t.Errorf("AccessToken = %q", record.AccessToken)
	}
	if record.RefreshToken != "login-refresh" {
		t.Errorf("RefreshToken = %q", record.RefreshToken)
	}
	if record.ExpiresAt.Before(time.Now().Add(3500 * time.Second)) {
		t.Errorf("ExpiresAt = %v — expires_in=3600 not honoured (too early)", record.ExpiresAt)
	}
	if record.ExpiresAt.After(time.Now().Add(3700 * time.Second)) {
		t.Errorf("ExpiresAt = %v — expires_in=3600 not honoured (too late)", record.ExpiresAt)
	}
}

func TestOpenAIWebFlowSource_LoginCallbackErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		query   func(state string) string
		wantErr string
	}{
		{
			name:    "state mismatch",
			query:   func(string) string { return "state=wrong&code=abc" },
			wantErr: "state mismatch",
		},
		{
			name: "provider error",
			query: func(state string) string {
				return "state=" + url.QueryEscape(state) + "&error=access_denied&error_description=nope"
			},
			wantErr: "access_denied",
		},
		{
			name:    "missing code",
			query:   func(state string) string { return "state=" + url.QueryEscape(state) },
			wantErr: "no code",
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			port := freePort(t)
			urlCh := make(chan string, 1)
			src := NewOpenAIWebFlowSource(OpenAIWebFlowConfig{
				ClientID:  "web-client",
				Port:      port,
				Store:     NewMemoryStore(),
				OnOpenURL: func(u string) { urlCh <- u },
				AuthURL:   "https://test.openai.local/authorize",
				TokenURL:  "https://test.openai.local/oauth/token",
			})

			loginErr := make(chan error, 1)
			go func() { loginErr <- src.Login(context.Background()) }()

			authURL := awaitAuthURL(t, urlCh, loginErr)
			parsed, err := url.Parse(authURL)
			if err != nil {
				t.Fatalf("parse authURL: %v", err)
			}
			state := parsed.Query().Get("state")

			resp := driveCallback(t, fmt.Sprintf("http://localhost:%d/callback?%s", port, tt.query(state)))
			resp.Body.Close()
			if resp.StatusCode != http.StatusBadRequest {
				t.Errorf("callback status = %d, want 400", resp.StatusCode)
			}

			err = <-loginErr
			if err == nil {
				t.Fatal("expected Login error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("Login error = %q, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestOpenAIWebFlowSource_LoginListenError(t *testing.T) {
	t.Parallel()

	// Occupy the port so Login's listener cannot bind.
	l, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer l.Close()
	port := l.Addr().(*net.TCPAddr).Port

	src := NewOpenAIWebFlowSource(OpenAIWebFlowConfig{
		ClientID: "web-client",
		Port:     port,
		Store:    NewMemoryStore(),
	})
	if err := src.Login(context.Background()); err == nil {
		t.Fatal("expected listen error when port is occupied")
	}
}

func TestOpenAIWebFlowSource_LoginTimeout(t *testing.T) {
	t.Parallel()

	port := freePort(t)
	src := NewOpenAIWebFlowSource(OpenAIWebFlowConfig{
		ClientID: "web-client",
		Port:     port,
		Store:    NewMemoryStore(),
	})

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	err := src.Login(ctx)
	if err == nil {
		t.Fatal("expected timeout error")
	}
	if !strings.Contains(err.Error(), "timed out") {
		t.Errorf("Login error = %q, want timeout", err)
	}
}

func TestOpenAIWebFlowSource_TokenTriggersLogin(t *testing.T) {
	t.Parallel()

	tokenServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "via-login",
			"refresh_token": "via-login-refresh",
			"expires_in":    3600,
		})
	}))
	defer tokenServer.Close()

	port := freePort(t)
	urlCh := make(chan string, 1)
	store := NewMemoryStore()

	src := NewOpenAIWebFlowSource(OpenAIWebFlowConfig{
		ClientID:   "web-client",
		Port:       port,
		Store:      store,
		OnOpenURL:  func(u string) { urlCh <- u },
		HTTPClient: tokenServer.Client(),
		AuthURL:    "https://test.openai.local/authorize",
		TokenURL:   tokenServer.URL,
	})

	type tokenResult struct {
		tok *Token
		err error
	}
	resCh := make(chan tokenResult, 1)
	go func() {
		tok, err := src.Token(context.Background())
		resCh <- tokenResult{tok, err}
	}()

	var authURL string
	select {
	case authURL = <-urlCh:
	case res := <-resCh:
		t.Fatalf("Token exited before publishing auth URL: %v", res.err)
	case <-time.After(30 * time.Second):
		t.Fatal("timed out waiting for auth URL")
	}
	parsed, err := url.Parse(authURL)
	if err != nil {
		t.Fatalf("parse authURL: %v", err)
	}
	state := parsed.Query().Get("state")

	resp := driveCallback(t, fmt.Sprintf("http://localhost:%d/callback?state=%s&code=x", port, url.QueryEscape(state)))
	resp.Body.Close()

	res := <-resCh
	if res.err != nil {
		t.Fatalf("Token: %v", res.err)
	}
	if res.tok.AccessToken != "via-login" {
		t.Errorf("AccessToken = %q, want via-login", res.tok.AccessToken)
	}
	if !res.tok.Valid() {
		t.Error("token from login should be valid (expiry honoured)")
	}

	// Round-trip: the record Token loaded after Login must match the store.
	record, err := loadOpenAITokenRecord(context.Background(), store, openaiWebStoreKey)
	if err != nil {
		t.Fatalf("loadOpenAITokenRecord: %v", err)
	}
	if record.AccessToken != "via-login" {
		t.Errorf("stored AccessToken = %q", record.AccessToken)
	}
}

func TestOpenAIWebFlowSource_TokenUsesValidStoredRecord(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	ctx := context.Background()
	valid := openAITokenRecord{
		AccessToken: "stored-valid",
		ExpiresAt:   time.Now().Add(time.Hour),
	}
	if err := saveOpenAITokenRecord(ctx, store, openaiWebStoreKey, valid); err != nil {
		t.Fatalf("seed: %v", err)
	}

	// Any network call is a test failure — a valid stored record must
	// be returned without HTTP.
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		t.Errorf("unexpected HTTP call to %s", req.URL)
		return nil, fmt.Errorf("no network expected")
	})}

	src := NewOpenAIWebFlowSource(OpenAIWebFlowConfig{
		ClientID:   "web-client",
		Store:      store,
		HTTPClient: client,
	})

	tok, err := src.Token(ctx)
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if tok.AccessToken != "stored-valid" {
		t.Errorf("AccessToken = %q, want stored-valid", tok.AccessToken)
	}

	// Second call hits the in-memory cache path.
	tok2, err := src.Token(ctx)
	if err != nil {
		t.Fatalf("Token (cached): %v", err)
	}
	if tok2.AccessToken != "stored-valid" {
		t.Errorf("cached AccessToken = %q", tok2.AccessToken)
	}
}
