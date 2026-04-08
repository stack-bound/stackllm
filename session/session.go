package session

import (
	"crypto/rand"
	"encoding/hex"
	"time"

	"github.com/stack-bound/stackllm/conversation"
)

// Session holds conversation history and arbitrary KV state for an agent run.
type Session struct {
	ID       string
	Messages []conversation.Message
	State    map[string]any
	Created  time.Time
	Updated  time.Time
}

// New creates a new session with a random ID.
func New() *Session {
	return &Session{
		ID:      generateID(),
		State:   make(map[string]any),
		Created: time.Now(),
		Updated: time.Now(),
	}
}

// AppendMessage adds a message and updates the timestamp.
func (s *Session) AppendMessage(msg conversation.Message) {
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

func generateID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return hex.EncodeToString(b)
}
