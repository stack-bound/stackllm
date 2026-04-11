package session

import (
	"time"

	"github.com/stack-bound/stackllm/conversation"
)

// Session holds conversation history and arbitrary KV state for an agent run.
//
// Name, ProjectPath, and Model are surfaced as dedicated columns by
// SQLiteStore so List can return them without loading the full
// conversation; they are optional and may be left zero by embedders
// that don't need them.
//
// LastUsage is the token usage reported by the most recent provider
// turn. It is nil until the first turn completes and is persisted by
// SQLiteStore so reopening a session restores the correct figures
// without a round-trip to the model.
type Session struct {
	ID          string                   `json:"id"`
	Name        string                   `json:"name,omitempty"`
	ProjectPath string                   `json:"project_path,omitempty"`
	Model       string                   `json:"model,omitempty"`
	Messages    []conversation.Message   `json:"messages"`
	State       map[string]any           `json:"state"`
	LastUsage   *conversation.TokenUsage `json:"last_usage,omitempty"`
	Created     time.Time                `json:"created"`
	Updated     time.Time                `json:"updated"`
}

// New creates a new session with a fresh UUIDv7 ID (via
// conversation.NewID, which is the single ID factory for the library).
func New() *Session {
	return &Session{
		ID:      conversation.NewID(),
		State:   make(map[string]any),
		Created: time.Now(),
		Updated: time.Now(),
	}
}

// AppendMessage adds a message and updates the timestamp.
func (s *Session) AppendMessage(msg conversation.Message) {
	conversation.EnsureMessageIDs(&msg)
	s.Messages = append(s.Messages, msg)
	s.Updated = time.Now()
}

// SetState sets a key-value pair in the session state.
func (s *Session) SetState(key string, value any) {
	if s.State == nil {
		s.State = make(map[string]any)
	}
	s.State[key] = value
	s.Updated = time.Now()
}

// GetState retrieves a value from the session state.
func (s *Session) GetState(key string) (any, bool) {
	v, ok := s.State[key]
	return v, ok
}

