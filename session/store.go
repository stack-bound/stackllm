package session

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"github.com/stack-bound/stackllm/conversation"
)

// SessionStore persists sessions.
type SessionStore interface {
	Save(ctx context.Context, s *Session) error
	Load(ctx context.Context, id string) (*Session, error)
	Delete(ctx context.Context, id string) error
	List(ctx context.Context) ([]*Session, error)
}

// DefaultListLimit is the page size SessionPaginator implementations
// apply when ListOptions.Limit is zero.
const DefaultListLimit = 50

// ListOptions controls pagination for SessionPaginator.ListPage.
// The zero value (Limit=0, Offset=0) returns the first page at the
// default page size; pass a negative Limit to disable the cap.
type ListOptions struct {
	Limit  int // 0 → DefaultListLimit; <0 → no limit
	Offset int // rows to skip
}

// ListResult is one page of sessions plus the total matching row
// count (ignoring Limit/Offset) so callers can render "page X of Y".
type ListResult struct {
	Sessions []*Session
	Total    int
}

// SessionPaginator is the optional capability a SessionStore opts
// into when it can return sessions in pages. Both the built-in
// InMemoryStore and SQLiteStore implement it; embedders can
// feature-detect via type assertion.
//
// Sort order matches List: most-recently-updated first.
type SessionPaginator interface {
	ListPage(ctx context.Context, opts ListOptions) (ListResult, error)
}

// InMemoryStore is the default session store. Not persistent across restarts.
type InMemoryStore struct {
	mu       sync.RWMutex
	sessions map[string]*Session
}

// NewInMemoryStore creates a new in-memory session store.
func NewInMemoryStore() *InMemoryStore {
	return &InMemoryStore{sessions: make(map[string]*Session)}
}

// Save stores or updates a session.
func (s *InMemoryStore) Save(_ context.Context, sess *Session) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range sess.Messages {
		conversation.EnsureMessageIDs(&sess.Messages[i])
	}
	s.sessions[sess.ID] = sess
	return nil
}

// Load retrieves a session by ID.
func (s *InMemoryStore) Load(_ context.Context, id string) (*Session, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	sess, ok := s.sessions[id]
	if !ok {
		return nil, fmt.Errorf("session: %q not found", id)
	}
	return sess, nil
}

// Delete removes a session by ID.
func (s *InMemoryStore) Delete(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.sessions, id)
	return nil
}

// List returns all sessions.
func (s *InMemoryStore) List(_ context.Context) ([]*Session, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	result := make([]*Session, 0, len(s.sessions))
	for _, sess := range s.sessions {
		result = append(result, sess)
	}
	return result, nil
}

// ListPage returns a sorted page of sessions plus the total row
// count. Sessions are ordered by Updated desc (newest first); ties
// fall back to ID for determinism.
func (s *InMemoryStore) ListPage(_ context.Context, opts ListOptions) (ListResult, error) {
	s.mu.RLock()
	all := make([]*Session, 0, len(s.sessions))
	for _, sess := range s.sessions {
		all = append(all, sess)
	}
	s.mu.RUnlock()

	sort.SliceStable(all, func(i, j int) bool {
		if all[i].Updated.Equal(all[j].Updated) {
			return all[i].ID < all[j].ID
		}
		return all[i].Updated.After(all[j].Updated)
	})

	total := len(all)
	offset := opts.Offset
	if offset < 0 {
		offset = 0
	}
	if offset >= total {
		return ListResult{Sessions: []*Session{}, Total: total}, nil
	}
	page := all[offset:]

	limit := opts.Limit
	if limit == 0 {
		limit = DefaultListLimit
	}
	if limit > 0 && limit < len(page) {
		page = page[:limit]
	}
	return ListResult{Sessions: page, Total: total}, nil
}
