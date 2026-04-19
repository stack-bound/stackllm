package session

import (
	"context"
	"fmt"
	"sort"
	"testing"
	"time"

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

	s.AppendMessage(conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hello"}},
	})

	if len(s.Messages) != 1 {
		t.Fatalf("Messages len = %d, want 1", len(s.Messages))
	}
	if got := s.Messages[0].TextContent(); got != "hello" {
		t.Errorf("TextContent = %q, want %q", got, "hello")
	}
	if s.Messages[0].ID == "" || s.Messages[0].Blocks[0].ID == "" {
		t.Fatal("AppendMessage should assign message and block IDs")
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
	s.AppendMessage(conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "hi"}},
	})

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
	s.AppendMessage(conversation.Message{
		Role:   conversation.RoleUser,
		Blocks: []conversation.Block{{Type: conversation.BlockText, Text: "update"}},
	})
	store.Save(ctx, s)

	loaded, _ := store.Load(ctx, s.ID)
	if len(loaded.Messages) != 1 {
		t.Errorf("Messages len = %d, want 1", len(loaded.Messages))
	}
}

// TestInMemoryStore_ListPage covers the SessionPaginator contract:
// ordering, total count, default limit, negative limit, offset past
// end, and parity with List for the unbounded case.
func TestInMemoryStore_ListPage(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	// Compile-time check that InMemoryStore satisfies the optional
	// pagination interface.
	var _ SessionPaginator = (*InMemoryStore)(nil)

	store := NewInMemoryStore()
	now := time.Now()
	for i := 0; i < 10; i++ {
		s := New()
		s.Name = fmt.Sprintf("sess-%02d", i)
		s.Updated = now.Add(time.Duration(i) * time.Second)
		if err := store.Save(ctx, s); err != nil {
			t.Fatalf("Save: %v", err)
		}
	}

	// First page: 3 newest, in updated-desc order.
	got, err := store.ListPage(ctx, ListOptions{Limit: 3})
	if err != nil {
		t.Fatalf("ListPage: %v", err)
	}
	if got.Total != 10 {
		t.Errorf("Total = %d, want 10", got.Total)
	}
	if len(got.Sessions) != 3 {
		t.Fatalf("Sessions len = %d, want 3", len(got.Sessions))
	}
	wantNames := []string{"sess-09", "sess-08", "sess-07"}
	for i, want := range wantNames {
		if got.Sessions[i].Name != want {
			t.Errorf("Sessions[%d].Name = %q, want %q", i, got.Sessions[i].Name, want)
		}
	}

	// Last partial page.
	got, err = store.ListPage(ctx, ListOptions{Limit: 3, Offset: 9})
	if err != nil {
		t.Fatalf("ListPage offset=9: %v", err)
	}
	if got.Total != 10 {
		t.Errorf("Total = %d, want 10", got.Total)
	}
	if len(got.Sessions) != 1 {
		t.Errorf("Sessions len = %d, want 1", len(got.Sessions))
	}
	if got.Sessions[0].Name != "sess-00" {
		t.Errorf("last page name = %q, want sess-00", got.Sessions[0].Name)
	}

	// Offset past end → empty page, total still correct.
	got, err = store.ListPage(ctx, ListOptions{Limit: 3, Offset: 999})
	if err != nil {
		t.Fatalf("ListPage offset=999: %v", err)
	}
	if got.Total != 10 || len(got.Sessions) != 0 {
		t.Errorf("offset past end: Total=%d Sessions=%d", got.Total, len(got.Sessions))
	}

	// Default limit kicks in at Limit=0.
	store2 := NewInMemoryStore()
	for i := 0; i < DefaultListLimit+10; i++ {
		s := New()
		s.Updated = now.Add(time.Duration(i) * time.Second)
		store2.Save(ctx, s)
	}
	got, err = store2.ListPage(ctx, ListOptions{})
	if err != nil {
		t.Fatalf("ListPage default: %v", err)
	}
	if len(got.Sessions) != DefaultListLimit {
		t.Errorf("default page len = %d, want %d", len(got.Sessions), DefaultListLimit)
	}
	if got.Total != DefaultListLimit+10 {
		t.Errorf("default Total = %d, want %d", got.Total, DefaultListLimit+10)
	}

	// Negative limit returns everything.
	got, err = store2.ListPage(ctx, ListOptions{Limit: -1})
	if err != nil {
		t.Fatalf("ListPage Limit=-1: %v", err)
	}
	if len(got.Sessions) != DefaultListLimit+10 {
		t.Errorf("Limit=-1 len = %d, want %d", len(got.Sessions), DefaultListLimit+10)
	}

	// Parity with List for the unbounded case (set comparison).
	listAll, err := store2.List(ctx)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if !sameIDSet(listAll, got.Sessions) {
		t.Error("ListPage(Limit=-1) and List returned different ID sets")
	}
}

// sameIDSet returns true if both slices contain the same session IDs
// regardless of order.
func sameIDSet(a, b []*Session) bool {
	if len(a) != len(b) {
		return false
	}
	idsA := make([]string, len(a))
	idsB := make([]string, len(b))
	for i := range a {
		idsA[i] = a[i].ID
		idsB[i] = b[i].ID
	}
	sort.Strings(idsA)
	sort.Strings(idsB)
	for i := range idsA {
		if idsA[i] != idsB[i] {
			return false
		}
	}
	return true
}

// TestInMemoryStore_BlocksRoundTrip is the Phase 1 completion gate: a
// session carrying an assistant message with every supported block
// type must round-trip through Save/Load with block order and content
// preserved verbatim.
func TestInMemoryStore_BlocksRoundTrip(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	store := NewInMemoryStore()
	s := New()

	// Assistant message with an interleaved 7-block turn covering the
	// full Phase 1 goal:
	//   thinking → text → tool_use → thinking → tool_use → thinking → text
	s.AppendMessage(conversation.Message{
		Role: conversation.RoleAssistant,
		Blocks: []conversation.Block{
			{Type: conversation.BlockThinking, Text: "planning"},
			{Type: conversation.BlockText, Text: "Let me check."},
			{Type: conversation.BlockToolUse, ToolCallID: "c1", ToolName: "read_file", ToolArgsJSON: `{"p":"a"}`},
			{Type: conversation.BlockThinking, Text: "found it"},
			{Type: conversation.BlockToolUse, ToolCallID: "c2", ToolName: "grep", ToolArgsJSON: `{"q":"b"}`},
			{Type: conversation.BlockThinking, Text: "analyzing"},
			{Type: conversation.BlockText, Text: "The bug is X."},
		},
	})

	// Tool-role message carrying two BlockToolResult blocks (one for
	// each tool_use above). Single tool message with multiple results
	// is the Phase 1 shape for multi-tool turns.
	s.AppendMessage(conversation.Message{
		Role: conversation.RoleTool,
		Blocks: []conversation.Block{
			{Type: conversation.BlockToolResult, ToolCallID: "c1", Text: "file contents"},
			{Type: conversation.BlockToolResult, ToolCallID: "c2", Text: "matches: a b c", ToolIsError: false},
		},
	})

	// Additional message types.
	s.AppendMessage(conversation.Message{
		Role: conversation.RoleUser,
		Blocks: []conversation.Block{
			{Type: conversation.BlockText, Text: "what's in this?"},
			{Type: conversation.BlockImage, MimeType: "image/png", ImageData: []byte{0x89, 0x50, 0x4E, 0x47}},
		},
	})
	s.AppendMessage(conversation.Message{
		Role: conversation.RoleAssistant,
		Blocks: []conversation.Block{
			{Type: conversation.BlockRedactedThinking, RedactedData: []byte{0xde, 0xad, 0xbe, 0xef}},
		},
	})

	if err := store.Save(ctx, s); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	loaded, err := store.Load(ctx, s.ID)
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}

	if len(loaded.Messages) != 4 {
		t.Fatalf("loaded messages = %d, want 4", len(loaded.Messages))
	}

	// Assistant turn: verify all 7 blocks round-trip in order.
	assistant := loaded.Messages[0]
	wantTypes := []conversation.BlockType{
		conversation.BlockThinking, conversation.BlockText, conversation.BlockToolUse,
		conversation.BlockThinking, conversation.BlockToolUse,
		conversation.BlockThinking, conversation.BlockText,
	}
	if len(assistant.Blocks) != len(wantTypes) {
		t.Fatalf("assistant blocks = %d, want %d", len(assistant.Blocks), len(wantTypes))
	}
	for i, want := range wantTypes {
		if assistant.Blocks[i].Type != want {
			t.Errorf("Blocks[%d].Type = %q, want %q", i, assistant.Blocks[i].Type, want)
		}
	}
	if got := assistant.Blocks[2].ToolCallID; got != "c1" {
		t.Errorf("tool_use[0] call id = %q, want c1", got)
	}
	if got := assistant.Blocks[2].ToolName; got != "read_file" {
		t.Errorf("tool_use[0] name = %q, want read_file", got)
	}
	if got := assistant.Blocks[2].ToolArgsJSON; got != `{"p":"a"}` {
		t.Errorf("tool_use[0] args = %q", got)
	}
	if got := assistant.Blocks[4].ToolName; got != "grep" {
		t.Errorf("tool_use[1] name = %q, want grep", got)
	}
	if got := assistant.Blocks[0].Text; got != "planning" {
		t.Errorf("thinking[0] text = %q", got)
	}
	if got := assistant.Blocks[6].Text; got != "The bug is X." {
		t.Errorf("text[1] text = %q", got)
	}

	// Tool-role message with two tool_result blocks.
	tool := loaded.Messages[1]
	if tool.Role != conversation.RoleTool {
		t.Errorf("messages[1].Role = %q, want tool", tool.Role)
	}
	results := tool.ToolResults()
	if len(results) != 2 {
		t.Fatalf("tool_result blocks = %d, want 2", len(results))
	}
	if results[0].ToolCallID != "c1" || results[0].Text != "file contents" {
		t.Errorf("result[0] = %+v", results[0])
	}
	if results[1].ToolCallID != "c2" || results[1].Text != "matches: a b c" {
		t.Errorf("result[1] = %+v", results[1])
	}

	// User turn with image.
	user := loaded.Messages[2]
	if len(user.Blocks) != 2 {
		t.Fatalf("user blocks = %d, want 2", len(user.Blocks))
	}
	if user.Blocks[1].Type != conversation.BlockImage {
		t.Errorf("user blocks[1].Type = %q", user.Blocks[1].Type)
	}
	if user.Blocks[1].MimeType != "image/png" {
		t.Errorf("user image mime = %q", user.Blocks[1].MimeType)
	}
	if len(user.Blocks[1].ImageData) != 4 {
		t.Errorf("user image data len = %d", len(user.Blocks[1].ImageData))
	}

	// Redacted thinking.
	redacted := loaded.Messages[3]
	if len(redacted.Blocks) != 1 || redacted.Blocks[0].Type != conversation.BlockRedactedThinking {
		t.Fatalf("redacted thinking block missing")
	}
	if len(redacted.Blocks[0].RedactedData) != 4 {
		t.Errorf("redacted data len = %d", len(redacted.Blocks[0].RedactedData))
	}
}
