// Command web demonstrates embedding stackllm behind a browser-only
// UI. It starts an HTTP server exposing:
//
//   - /api/* — the ManagedHandler: auth, model selection, chat
//   - /     — a minimal single-page HTML UI that drives the API
//
// No TUI is involved — the entire provider lifecycle (login, model
// selection, default setting) happens through the browser. Copilot
// device-flow auth is surfaced as a code + verification URL in the
// UI; polling is handled server-side.
//
// Run:
//
//	go run ./examples/web
//
// Then open http://localhost:8080 in a browser.
//
// The example embeds its own minimal HTML so there are no external
// asset dependencies — embedders should replace it with their own UI.
package main

import (
	_ "embed"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/stack-bound/stackllm/auth"
	"github.com/stack-bound/stackllm/config"
	"github.com/stack-bound/stackllm/profile"
	"github.com/stack-bound/stackllm/session"
	"github.com/stack-bound/stackllm/web"
)

//go:embed index.html
var indexHTML []byte

func main() {
	addr := ":8080"
	if v := os.Getenv("STACKLLM_WEB_ADDR"); v != "" {
		addr = v
	}

	// Persist auth + config under a per-app directory so this example
	// does not stomp on the stackllm default profile.
	dataDir := filepath.Join(os.TempDir(), "stackllm-web-example")
	if err := os.MkdirAll(dataDir, 0700); err != nil {
		log.Fatalf("create data dir: %v", err)
	}
	authStore := &auth.FileStore{Path: filepath.Join(dataDir, "auth.json")}
	configStore := &config.Store{Path: filepath.Join(dataDir, "config.json")}

	mgr := profile.New(
		profile.WithAuthStore(authStore),
		profile.WithConfigStore(configStore),
	)

	store := session.NewInMemoryStore()

	// Sign-in-with-ChatGPT (device + PKCE) is always available — the
	// handler uses the Codex CLI's public OAuth client ID so no app
	// registration is required. Users who prefer a plain API key can
	// still paste one into the UI.
	api := web.NewManagedHandler(mgr, store)

	mux := http.NewServeMux()
	mux.Handle("/api/", http.StripPrefix("/api", api))
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(indexHTML)
	})

	fmt.Printf("stackllm web example listening on http://localhost%s\n", addr)
	fmt.Printf("data dir: %s\n", dataDir)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}
