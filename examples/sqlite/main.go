// Command sqlite demonstrates embedding stackllm's session store
// alongside a parent application's own tables in the same SQLite file
// AND wiring a tool that reads from the parent application's tables
// through the shared connection pool.
//
// The shape demonstrated here is the "shared DB" pattern for embedders:
//
//  1. the parent application opens *sql.DB itself (modernc.org/sqlite,
//     no CGO) with whichever pragmas it wants
//  2. the parent app runs its own schema migrations on that DB
//  3. the parent app hands the same *sql.DB to session.NewSQLiteStore,
//     which runs stackllm's migrations side-by-side — all stackllm
//     tables are prefixed `stackllm_` so nothing collides with
//     parent-app tables
//  4. a tool registered with the agent can query the parent-app
//     tables through the same connection pool, so the agent can reach
//     into host application state with zero extra plumbing
//  5. the full agent turn (user prompt → model → tool_use →
//     tool_result → final text) is persisted through the same
//     SQLiteStore, including blocks, artifacts, and FTS5 indexing
//
// This is an integration example, not a test fixture. It runs an
// actual agent turn against whichever provider you have configured as
// your profile default (or via the interactive setup on first run).
// For a no-network-required unit test of the store itself, see
// session/sqlite_test.go.
//
// Usage:
//
//	go run ./examples/sqlite
package main

import (
	"bufio"
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/stack-bound/stackllm/agent"
	"github.com/stack-bound/stackllm/conversation"
	"github.com/stack-bound/stackllm/profile"
	"github.com/stack-bound/stackllm/provider"
	"github.com/stack-bound/stackllm/session"
	"github.com/stack-bound/stackllm/tools"

	_ "modernc.org/sqlite"
)

// ReadMemoryArgs is the parameter struct for the read_memory tool.
// The agent uses the description to decide when and how to call it.
type ReadMemoryArgs struct {
	ID int `json:"id" jsonschema:"description=The integer id of the memory row to read,required"`
}

func main() {
	ctx := context.Background()
	scanner := bufio.NewScanner(os.Stdin)

	// 1. Resolve a path under $XDG_DATA_HOME so this example writes
	//    its demo DB where real stackllm-embedder state would live.
	dir := os.Getenv("XDG_DATA_HOME")
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			fmt.Fprintln(os.Stderr, "resolve home:", err)
			os.Exit(1)
		}
		dir = filepath.Join(home, ".local", "share")
	}
	stateDir := filepath.Join(dir, "stackllm-example")
	if err := os.MkdirAll(stateDir, 0o700); err != nil {
		fmt.Fprintln(os.Stderr, "create state dir:", err)
		os.Exit(1)
	}
	dbPath := filepath.Join(stateDir, "state.db")
	fmt.Println("Database:", dbPath)

	// 2. Open *sql.DB directly. The example sets the "durable but
	//    fast" pragma profile in its own DSN. stackllm's
	//    NewSQLiteStore will additionally flip journal_mode=WAL at
	//    bootstrap and apply foreign_keys / busy_timeout per stackllm
	//    transaction — but parent-app queries against this handle
	//    only see the DSN-level pragmas, which is why the example
	//    sets them here too.
	dsn := "file:" + dbPath + "?_pragma=foreign_keys(1)&_pragma=journal_mode(wal)&_pragma=synchronous(normal)&_pragma=busy_timeout(5000)"
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		fmt.Fprintln(os.Stderr, "sql.Open:", err)
		os.Exit(1)
	}
	defer db.Close()

	// 3. Parent-app migration: a simple `memories` table the host
	//    application owns. Note it does NOT use the stackllm_ prefix —
	//    that namespace is reserved for stackllm-owned tables.
	if _, err := db.ExecContext(ctx, `
		CREATE TABLE IF NOT EXISTS memories (
			id   INTEGER PRIMARY KEY,
			text TEXT NOT NULL
		)`); err != nil {
		fmt.Fprintln(os.Stderr, "create memories:", err)
		os.Exit(1)
	}
	// Seed one row idempotently so re-runs don't duplicate it.
	if _, err := db.ExecContext(ctx,
		`INSERT INTO memories(id, text) VALUES (1, ?) ON CONFLICT(id) DO NOTHING`,
		"Matt prefers terse code reviews and one-shot fixes.",
	); err != nil {
		fmt.Fprintln(os.Stderr, "seed memories:", err)
		os.Exit(1)
	}

	// 4. Hand the same connection to stackllm. NewSQLiteStore runs
	//    stackllm's migrations against this DB, creating every
	//    stackllm_* table needed.
	store, err := session.NewSQLiteStore(db)
	if err != nil {
		fmt.Fprintln(os.Stderr, "NewSQLiteStore:", err)
		os.Exit(1)
	}

	// 5. Load (or interactively configure) a provider. The example
	//    runs against whichever provider you've set as your profile
	//    default — Ollama, OpenAI, Copilot, Azure, Gemini are all
	//    fine.
	mgr := profile.New(profile.WithCallbacks(profile.Callbacks{
		OnDeviceCode: func(userCode, verifyURL string) {
			fmt.Printf("\nOpen %s and enter code: %s\n\n", verifyURL, userCode)
		},
		OnPolling: func() { fmt.Print(".") },
		OnSuccess: func() { fmt.Println("\nAuthenticated!") },
		OnPromptKey: func(providerName string) (string, error) {
			fmt.Printf("Enter API key for %s: ", providerName)
			if !scanner.Scan() {
				return "", fmt.Errorf("no input")
			}
			return strings.TrimSpace(scanner.Text()), nil
		},
		OnPromptURL: func(providerName, defaultURL string) (string, error) {
			fmt.Printf("Enter base URL for %s [%s]: ", providerName, defaultURL)
			if !scanner.Scan() {
				return defaultURL, nil
			}
			v := strings.TrimSpace(scanner.Text())
			if v == "" {
				return defaultURL, nil
			}
			return v, nil
		},
	}))

	p, err := mgr.LoadDefault(ctx)
	if err != nil {
		fmt.Println("No default provider configured. Let's set one up.")
		p, err = interactiveSetup(ctx, mgr, scanner)
		if err != nil {
			fmt.Fprintln(os.Stderr, "provider setup:", err)
			os.Exit(1)
		}
	}

	// 6. Register a tool that reads from the parent-app `memories`
	//    table via the shared *sql.DB. This is the unique value of
	//    the example: it shows a tool reaching into host application
	//    state through the same connection pool that stackllm uses.
	registry := tools.NewRegistry()
	if err := registry.Register(
		"read_memory",
		"Read a stored memory row from the host application's memories table by integer id.",
		func(ctx context.Context, args ReadMemoryArgs) (string, error) {
			var text string
			err := db.QueryRowContext(ctx,
				`SELECT text FROM memories WHERE id = ?`,
				args.ID,
			).Scan(&text)
			if err == sql.ErrNoRows {
				return fmt.Sprintf("no memory found with id %d", args.ID), nil
			}
			if err != nil {
				return "", err
			}
			return text, nil
		},
	); err != nil {
		fmt.Fprintln(os.Stderr, "register read_memory:", err)
		os.Exit(1)
	}

	// 7. Build the agent. Hooks print block deltas as they stream,
	//    using the block-oriented streaming API (not the legacy
	//    OnToken shim).
	a := agent.New(p,
		agent.WithTools(registry),
		agent.WithMaxSteps(5),
		agent.WithHooks(agent.Hooks{
			OnBlockStart: func(ctx context.Context, bt conversation.BlockType) {
				switch bt {
				case conversation.BlockThinking:
					fmt.Print("\n[thinking] ")
				case conversation.BlockText:
					fmt.Print("\n")
				case conversation.BlockToolUse:
					fmt.Print("\n[tool_use] ")
				}
			},
			OnBlockDelta: func(ctx context.Context, bt conversation.BlockType, delta string) {
				fmt.Print(delta)
			},
			OnToolResult: func(ctx context.Context, call conversation.ToolCall, result string, err error) {
				fmt.Printf("\n[tool_result %s] %s\n", call.Name, result)
			},
		}),
	)

	// 8. Build the session and seed a user prompt that should drive
	//    the agent to call read_memory.
	sess := session.New()
	sess.Name = "sqlite-demo"
	sess.ProjectPath = stateDir
	sess.Model = "default"
	initial := conversation.NewBuilder().
		System("You are a helpful assistant. Use the read_memory tool to read stored memories from the host application when asked.").
		User("Read memory id 1 and tell me what it says.").
		Build()
	for _, m := range initial {
		sess.AppendMessage(m)
	}

	// 9. Run the agent and collect the evolved message list on
	//    EventComplete, mirroring the adapter contract in
	//    web/handler.go and tui/model.go.
	events, err := a.Run(ctx, sess.Messages)
	if err != nil {
		fmt.Fprintln(os.Stderr, "agent.Run:", err)
		os.Exit(1)
	}
	var finalMessages []conversation.Message
	for ev := range events {
		switch ev.Type {
		case agent.EventComplete:
			finalMessages = append([]conversation.Message(nil), ev.Messages...)
		case agent.EventError:
			if len(ev.Messages) > 0 {
				finalMessages = append([]conversation.Message(nil), ev.Messages...)
			}
			fmt.Fprintln(os.Stderr, "\nagent error:", ev.Err)
			os.Exit(1)
		}
	}
	if finalMessages != nil {
		sess.Messages = finalMessages
	}

	// 10. Persist the evolved session. Save is append-only, so a
	//     second run of this example creates a new session rather
	//     than overwriting the previous one.
	if err := store.Save(ctx, sess); err != nil {
		fmt.Fprintln(os.Stderr, "store.Save:", err)
		os.Exit(1)
	}

	// 11. Read back both halves of the DB through the same connection
	//     pool to prove coexistence, and reload the session from
	//     scratch to exercise the SQLite Load path end-to-end.
	var memoryText string
	if err := db.QueryRowContext(ctx,
		`SELECT text FROM memories WHERE id = 1`,
	).Scan(&memoryText); err != nil {
		fmt.Fprintln(os.Stderr, "read memories:", err)
		os.Exit(1)
	}

	var (
		sessionCount int
		messageCount int
		blockCount   int
	)
	db.QueryRowContext(ctx, `SELECT COUNT(*) FROM stackllm_sessions`).Scan(&sessionCount)
	db.QueryRowContext(ctx, `SELECT COUNT(*) FROM stackllm_messages WHERE session_id = ?`, sess.ID).Scan(&messageCount)
	db.QueryRowContext(ctx, `
		SELECT COUNT(*) FROM stackllm_blocks b
		JOIN stackllm_messages m ON b.message_id = m.id
		WHERE m.session_id = ?`, sess.ID).Scan(&blockCount)

	reloaded, err := store.Load(ctx, sess.ID)
	if err != nil {
		fmt.Fprintln(os.Stderr, "store.Load:", err)
		os.Exit(1)
	}

	fmt.Println()
	fmt.Println()
	fmt.Println("Parent-app table (memories):")
	fmt.Println("  id=1 text=", memoryText)
	fmt.Println()
	fmt.Println("stackllm state:")
	fmt.Println("  sessions in db:", sessionCount)
	fmt.Println("  session id:    ", sess.ID)
	fmt.Println("  messages:      ", messageCount)
	fmt.Println("  blocks:        ", blockCount)
	fmt.Println()
	fmt.Println("Round-tripped conversation:")
	for i, msg := range reloaded.Messages {
		fmt.Printf("  [%d] %-9s %d blocks\n", i, msg.Role, len(msg.Blocks))
		for j, b := range msg.Blocks {
			preview := b.Text
			if len(preview) > 80 {
				preview = preview[:77] + "..."
			}
			switch b.Type {
			case conversation.BlockToolUse:
				fmt.Printf("        %d. %s -> %s(%s)\n", j, b.Type, b.ToolName, b.ToolArgsJSON)
			case conversation.BlockToolResult:
				fmt.Printf("        %d. %s <- %s\n", j, b.Type, preview)
			default:
				fmt.Printf("        %d. %s: %s\n", j, b.Type, preview)
			}
		}
	}
}

// interactiveSetup walks the user through picking a provider, logging
// in, selecting a model, and persisting the choice as the profile
// default. Same shape as examples/simple's flow, kept inline so the
// example stays self-contained.
func interactiveSetup(ctx context.Context, mgr *profile.Manager, scanner *bufio.Scanner) (*provider.OpenAIProvider, error) {
	providers := mgr.AvailableProviders()
	fmt.Println()
	fmt.Println("Available providers:")
	for i, p := range providers {
		fmt.Printf("  %d) %s\n", i+1, p)
	}
	fmt.Println()
	fmt.Print("Choose provider: ")

	if !scanner.Scan() {
		return nil, fmt.Errorf("no input")
	}
	idx, err := strconv.Atoi(strings.TrimSpace(scanner.Text()))
	if err != nil || idx < 1 || idx > len(providers) {
		return nil, fmt.Errorf("invalid choice")
	}
	providerName := providers[idx-1]

	if err := mgr.Login(ctx, providerName); err != nil {
		return nil, fmt.Errorf("login %s: %w", providerName, err)
	}
	fmt.Printf("Logged in to %s.\n", providerName)

	fmt.Println()
	fmt.Println("Fetching models...")
	models, err := mgr.ListModels(ctx, providerName)
	if err != nil {
		return nil, fmt.Errorf("list models: %w", err)
	}
	if len(models) == 0 {
		return nil, fmt.Errorf("no models available for %s", providerName)
	}

	fmt.Println()
	fmt.Println("Available models:")
	for i, m := range models {
		fmt.Printf("  %d) %s\n", i+1, m)
	}
	fmt.Println()
	fmt.Print("Choose model [1]: ")

	if !scanner.Scan() {
		return nil, fmt.Errorf("no input")
	}
	input := strings.TrimSpace(scanner.Text())
	if input == "" {
		input = "1"
	}
	midx, err := strconv.Atoi(input)
	if err != nil || midx < 1 || midx > len(models) {
		return nil, fmt.Errorf("invalid choice")
	}
	model := models[midx-1]

	defaultStr := providerName + "/" + model
	if err := mgr.SetDefault(defaultStr); err != nil {
		return nil, fmt.Errorf("set default: %w", err)
	}
	fmt.Printf("\nDefault set: %s\n\n", defaultStr)

	return mgr.LoadDefault(ctx)
}
