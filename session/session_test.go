package session

import (
	"context"
	"testing"

	"github.com/stack-bound/stackllm/conversation"
)

func TestNew(t *testing.T) {
	t.Parallel()

	s := New()
	if s.ID == "" {
		t.Error("expected non-empty ID")
	}
	if s.State == nil {
		t.Error("expected non-nil State map")
	}
	if s.Created.IsZero() {
		t.Error("expected non-zero Created time")
	}

	// Two sessions should have different IDs.
	s2 := New()
	if s.ID == s2.ID {
		t.Error("expected unique IDs")
	}
}

func TestSession_AppendMessage(t *testing.T) {
	t.Parallel()

	s := New()
	before := s.Updated

	s.AppendMessage(conversation.Message{Role: conversation.RoleUser, Content: "hello"})

	if len(s.Messages) != 1 {
		t.Fatalf("Messages len = %d, want 1", len(s.Messages))
	}
	if s.Messages[0].Content != "hello" {
		t.Errorf("Content = %q, want %q", s.Messages[0].Content, "hello")
	}
	if s.Updated.Before(before) {
		t.Error("Updated should advance after AppendMessage")
	}
}

func TestSession_State(t *testing.T) {
	t.Parallel()

	s := New()
	s.SetState("key", "value")

	v, ok := s.GetState("key")
	if !ok {
		t.Fatal("expected key to exist")
	}
	if v != "value" {
		t.Errorf("value = %v, want %q", v, "value")
	}

	_, ok = s.GetState("missing")
	if ok {
		t.Error("expected missing key to not exist")
	}
}

func TestInMemoryStore(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	store := NewInMemoryStore()

	// Load non-existent.
	_, err := store.Load(ctx, "missing")
	if err == nil {
		t.Fatal("expected error for missing session")
	}

	// Save and load.
	s := New()
	s.AppendMessage(conversation.Message{Role: conversation.RoleUser, Content: "hi"})

	if err := store.Save(ctx, s); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	loaded, err := store.Load(ctx, s.ID)
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}
	if loaded.ID != s.ID {
		t.Errorf("ID = %q, want %q", loaded.ID, s.ID)
	}
	if len(loaded.Messages) != 1 {
		t.Errorf("Messages len = %d, want 1", len(loaded.Messages))
	}

	// List.
	s2 := New()
	store.Save(ctx, s2)

	list, err := store.List(ctx)
	if err != nil {
		t.Fatalf("List error: %v", err)
	}
	if len(list) != 2 {
		t.Errorf("List len = %d, want 2", len(list))
	}

	// Delete.
	if err := store.Delete(ctx, s.ID); err != nil {
		t.Fatalf("Delete error: %v", err)
	}
	_, err = store.Load(ctx, s.ID)
	if err == nil {
		t.Error("expected error after delete")
	}

	list, _ = store.List(ctx)
	if len(list) != 1 {
		t.Errorf("List after delete len = %d, want 1", len(list))
	}
}

func TestInMemoryStore_Update(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	store := NewInMemoryStore()
	s := New()
	store.Save(ctx, s)

	// Modify and re-save.
	s.AppendMessage(conversation.Message{Role: conversation.RoleUser, Content: "update"})
	store.Save(ctx, s)

	loaded, _ := store.Load(ctx, s.ID)
	if len(loaded.Messages) != 1 {
		t.Errorf("Messages len = %d, want 1", len(loaded.Messages))
	}
}
