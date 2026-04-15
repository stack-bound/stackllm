# Change Log
All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).

## [0.3.0] - 2026-04-15
### Changed
- Changed colour of the thinking blocks and made italic.
### Fixed
- Show Gemini thinking blocks as they come under a differnt name to other models

## [0.2.0] - 2026-04-11
### Added
- TUI `/help` command listing all available slash commands with descriptions
- TUI `/sessions` command to browse, switch, and delete persisted sessions from a keyboard-driven picker
- TUI `/rename` command with a centered modal for renaming the current session
- TUI `/fork` command with a message picker for branching the current session at any point in its history
- TUI `/delete` command to remove the current session and start fresh
- TUI `/export` command for saving the current session to a JSONL file
- Session indicator in the TUI status bar showing the current session name and message count
- TUI image paste: Ctrl+V reads the system clipboard via wl-paste / xclip / pngpaste / osascript / powershell.exe, inserts a `[Image #N]` placeholder into the textarea, and emits an interleaved `BlockImage` on send so all five OpenAI-compatible backends receive the bytes as a base64 data URI. Falls back transparently to the default text paste when the clipboard holds no image or the platform tools are unavailable.
- TUI status line now shows the current model and context usage right-aligned (e.g. `gpt-5.4 · 12,345 / 200,000 (6.2%)`). Token counts come from actual provider `usage` fields (opts into `stream_options.include_usage` on chat completions; reads `response.completed.usage` on `/responses`), and the max context length is read from `capabilities.limits.max_prompt_tokens` where the provider exposes it (Copilot) with a hardcoded fallback table covering the GPT / Claude / Gemini / Llama / Mistral / Qwen / DeepSeek families. Per-session `LastUsage` is persisted via SQLite schema v2 so reopening a session restores the figures without a round-trip to the model.
### Changed
- TUI example now uses the SQLite-backed session store so sessions persist across runs

## [0.1.0] - 2026-04-10
### Added
- Block-shaped messages: every Message now holds an ordered slice of typed blocks (text, thinking, redacted_thinking, image, tool_use, tool_result), so interleaved assistant output replays faithfully through every layer — providers, agent loop, TUI, web SSE, and the session store.
- Reasoning capture across providers: chat completions parses delta.reasoning_content alongside delta.content, and the Responses API stream maps every output_item (reasoning / message / function_call) to its own block instead of dropping reasoning on the floor.
- Durable SQLite session store: session.SQLiteStore persists full conversation history to a single pure-Go SQLite file (modernc.org/sqlite, no CGO) with OpenSQLiteStore for standalone embedders and NewSQLiteStore for parent apps that share a *sql.DB — all stackllm tables are prefixed stackllm_ so host-app schemas coexist.
- Session branching: Fork, Rewind, and ListBranches let callers create sibling branches at any message boundary without deleting history, backed by a parent_id message tree and a current_leaf_id pointer.
- Artifact offload with SHA-256 dedupe: large tool results, inline image bytes, and redacted thinking payloads move to a side table with a small inline preview kept on the block row; HydrateArtifact fetches the full payload lazily and identical blobs share a single artifact row across sessions.
- FTS5 full-text search: session.Search runs full-text queries across text, thinking, and tool_result blocks with optional block-type and per-session scoping.
- JSONL export and import: ExportJSONL / ImportJSONL round-trips every block type, inlining artifact bytes so exported sessions are fully self-contained.
- examples/sqlite: runnable shared-DB demo that runs a parent-app migration alongside session.NewSQLiteStore on the same SQLite file.
### Removed
- conversation: Message.Content (string) and Message.ToolCalls ([]ToolCall) — superseded by Message.Blocks. Readers should switch to m.TextContent() or Blocks iteration; builders should switch to the block-oriented Builder methods.

## [0.0.2] - 2026-04-08
### Added
- Configurable poll intervals and retry backoff (PollInterval on auth configs, BaseBackoff on provider Config, WithPollInterval on profile Manager) to allow fast test execution without hardcoded sleeps
- Provider management layer: config/ package for user preferences, profile/ package composing auth+config+provider with Login/Logout/Status/ListModels/SetDefault/LoadDefault, examples/login interactive CLI, and updated examples to use profile-first resolution with interactive onboarding fallback

## [0.0.1] - 2026-04-08
### Added
- Initial Build

# Notes
[Deployment] Notes for deployment
[Added] for new features.
[Changed] for changes in existing functionality.
[Deprecated] for once-stable features removed in upcoming releases.
[Removed] for deprecated features removed in this release.
[Fixed] for any bug fixes.
[Security] to invite users to upgrade in case of vulnerabilities.
[YANKED] Note the emphasis, used for Hotfixes
