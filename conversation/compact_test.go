package conversation

import "testing"

func TestKeepLast(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		msgs     []Message
		n        int
		wantLen  int
		wantDesc string // description of expected result for debugging
	}{
		{
			name:     "empty slice",
			msgs:     nil,
			n:        5,
			wantLen:  0,
			wantDesc: "empty input returns empty output",
		},
		{
			name: "only system messages",
			msgs: []Message{
				{Role: RoleSystem, Content: "sys1"},
				{Role: RoleSystem, Content: "sys2"},
			},
			n:        1,
			wantLen:  2,
			wantDesc: "system messages always preserved",
		},
		{
			name: "already under limit",
			msgs: []Message{
				{Role: RoleSystem, Content: "sys"},
				{Role: RoleUser, Content: "hello"},
				{Role: RoleAssistant, Content: "hi"},
			},
			n:        5,
			wantLen:  3,
			wantDesc: "no trimming needed",
		},
		{
			name: "trim to last 1",
			msgs: []Message{
				{Role: RoleSystem, Content: "sys"},
				{Role: RoleUser, Content: "first"},
				{Role: RoleAssistant, Content: "response1"},
				{Role: RoleUser, Content: "second"},
				{Role: RoleAssistant, Content: "response2"},
			},
			n:        1,
			wantLen:  2, // system + last 1 non-system
			wantDesc: "keep system + last non-system message",
		},
		{
			name: "trim to last 2",
			msgs: []Message{
				{Role: RoleSystem, Content: "sys"},
				{Role: RoleUser, Content: "first"},
				{Role: RoleAssistant, Content: "response1"},
				{Role: RoleUser, Content: "second"},
				{Role: RoleAssistant, Content: "response2"},
			},
			n:        2,
			wantLen:  3, // system + last 2 non-system
			wantDesc: "keep system + last 2 non-system messages",
		},
		{
			name: "n=0 keeps only system",
			msgs: []Message{
				{Role: RoleSystem, Content: "sys"},
				{Role: RoleUser, Content: "hello"},
			},
			n:        0,
			wantLen:  1,
			wantDesc: "n=0 keeps only system messages",
		},
		{
			name: "system messages interspersed",
			msgs: []Message{
				{Role: RoleSystem, Content: "sys1"},
				{Role: RoleUser, Content: "a"},
				{Role: RoleSystem, Content: "sys2"},
				{Role: RoleUser, Content: "b"},
				{Role: RoleAssistant, Content: "c"},
			},
			n:        1,
			wantLen:  3, // sys1 + sys2 + last non-system
			wantDesc: "multiple system messages preserved with last non-system",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := KeepLast(tt.msgs, tt.n)
			if len(got) != tt.wantLen {
				t.Errorf("KeepLast() len = %d, want %d (%s)", len(got), tt.wantLen, tt.wantDesc)
				for i, m := range got {
					t.Logf("  [%d] %s: %s", i, m.Role, m.Content)
				}
			}
		})
	}
}

func TestKeepLast_PreservesContent(t *testing.T) {
	t.Parallel()

	msgs := []Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleUser, Content: "first"},
		{Role: RoleAssistant, Content: "response1"},
		{Role: RoleUser, Content: "second"},
		{Role: RoleAssistant, Content: "response2"},
	}

	got := KeepLast(msgs, 2)

	if got[0].Content != "sys" {
		t.Errorf("got[0].Content = %q, want %q", got[0].Content, "sys")
	}
	if got[1].Content != "second" {
		t.Errorf("got[1].Content = %q, want %q", got[1].Content, "second")
	}
	if got[2].Content != "response2" {
		t.Errorf("got[2].Content = %q, want %q", got[2].Content, "response2")
	}
}

func TestKeepLast_DoesNotMutateInput(t *testing.T) {
	t.Parallel()

	msgs := []Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleUser, Content: "a"},
		{Role: RoleUser, Content: "b"},
	}
	original := make([]Message, len(msgs))
	copy(original, msgs)

	_ = KeepLast(msgs, 1)

	for i := range msgs {
		if msgs[i].Content != original[i].Content {
			t.Errorf("input msgs[%d] was mutated", i)
		}
	}
}

func TestTokenBudget(t *testing.T) {
	t.Parallel()

	// Simple counter: each message costs len(Content).
	charCount := func(msgs []Message) int {
		total := 0
		for _, m := range msgs {
			total += len(m.Content)
		}
		return total
	}

	tests := []struct {
		name      string
		msgs      []Message
		maxTokens int
		wantLen   int
	}{
		{
			name:      "empty slice",
			msgs:      nil,
			maxTokens: 100,
			wantLen:   0,
		},
		{
			name: "already under budget",
			msgs: []Message{
				{Role: RoleSystem, Content: "sys"},
				{Role: RoleUser, Content: "hi"},
			},
			maxTokens: 100,
			wantLen:   2,
		},
		{
			name: "drops oldest non-system first",
			msgs: []Message{
				{Role: RoleSystem, Content: "sys"},       // 3 chars
				{Role: RoleUser, Content: "aaaaaaaaaa"},  // 10 chars
				{Role: RoleAssistant, Content: "bbbbb"},  // 5 chars
				{Role: RoleUser, Content: "cc"},          // 2 chars
			},
			maxTokens: 11, // need to drop "aaaaaaaaaa" (total would be 3+5+2=10)
			wantLen:   3,
		},
		{
			name: "only system messages remain",
			msgs: []Message{
				{Role: RoleSystem, Content: "s"},
				{Role: RoleUser, Content: "xxxxxxxxxx"},
				{Role: RoleUser, Content: "yyyyyyyyyy"},
			},
			maxTokens: 1,
			wantLen:   1,
		},
		{
			name: "system messages never dropped",
			msgs: []Message{
				{Role: RoleSystem, Content: "a very long system message that exceeds budget"},
			},
			maxTokens: 5,
			wantLen:   1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := TokenBudget(tt.msgs, tt.maxTokens, charCount)
			if len(got) != tt.wantLen {
				t.Errorf("TokenBudget() len = %d, want %d", len(got), tt.wantLen)
				for i, m := range got {
					t.Logf("  [%d] %s: %q", i, m.Role, m.Content)
				}
			}
		})
	}
}

func TestTokenBudget_NilCount(t *testing.T) {
	t.Parallel()

	// With nil count, uses default chars/4 heuristic.
	msgs := []Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleUser, Content: "hello world this is a test message"},
	}

	// Should not panic with nil count function.
	got := TokenBudget(msgs, 1000, nil)
	if len(got) != 2 {
		t.Errorf("expected 2 messages, got %d", len(got))
	}
}

func TestTokenBudget_DoesNotMutateInput(t *testing.T) {
	t.Parallel()

	msgs := []Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleUser, Content: "aaaaaaaaaa"},
		{Role: RoleUser, Content: "bb"},
	}
	original := make([]Message, len(msgs))
	copy(original, msgs)

	charCount := func(msgs []Message) int {
		total := 0
		for _, m := range msgs {
			total += len(m.Content)
		}
		return total
	}

	_ = TokenBudget(msgs, 5, charCount)

	for i := range msgs {
		if msgs[i].Content != original[i].Content {
			t.Errorf("input msgs[%d] was mutated", i)
		}
	}
}
