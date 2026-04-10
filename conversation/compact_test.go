package conversation

import "testing"

// mkMsg is a tiny helper that returns a Message with a single BlockText.
func mkMsg(role Role, text string) Message {
	return Message{Role: role, Blocks: []Block{{Type: BlockText, Text: text}}}
}

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
				mkMsg(RoleSystem, "sys1"),
				mkMsg(RoleSystem, "sys2"),
			},
			n:        1,
			wantLen:  2,
			wantDesc: "system messages always preserved",
		},
		{
			name: "already under limit",
			msgs: []Message{
				mkMsg(RoleSystem, "sys"),
				mkMsg(RoleUser, "hello"),
				mkMsg(RoleAssistant, "hi"),
			},
			n:        5,
			wantLen:  3,
			wantDesc: "no trimming needed",
		},
		{
			name: "trim to last 1",
			msgs: []Message{
				mkMsg(RoleSystem, "sys"),
				mkMsg(RoleUser, "first"),
				mkMsg(RoleAssistant, "response1"),
				mkMsg(RoleUser, "second"),
				mkMsg(RoleAssistant, "response2"),
			},
			n:        1,
			wantLen:  2, // system + last 1 non-system
			wantDesc: "keep system + last non-system message",
		},
		{
			name: "trim to last 2",
			msgs: []Message{
				mkMsg(RoleSystem, "sys"),
				mkMsg(RoleUser, "first"),
				mkMsg(RoleAssistant, "response1"),
				mkMsg(RoleUser, "second"),
				mkMsg(RoleAssistant, "response2"),
			},
			n:        2,
			wantLen:  3, // system + last 2 non-system
			wantDesc: "keep system + last 2 non-system messages",
		},
		{
			name: "n=0 keeps only system",
			msgs: []Message{
				mkMsg(RoleSystem, "sys"),
				mkMsg(RoleUser, "hello"),
			},
			n:        0,
			wantLen:  1,
			wantDesc: "n=0 keeps only system messages",
		},
		{
			name: "system messages interspersed",
			msgs: []Message{
				mkMsg(RoleSystem, "sys1"),
				mkMsg(RoleUser, "a"),
				mkMsg(RoleSystem, "sys2"),
				mkMsg(RoleUser, "b"),
				mkMsg(RoleAssistant, "c"),
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
					t.Logf("  [%d] %s: %s", i, m.Role, m.TextContent())
				}
			}
		})
	}
}

func TestKeepLast_PreservesContent(t *testing.T) {
	t.Parallel()

	msgs := []Message{
		mkMsg(RoleSystem, "sys"),
		mkMsg(RoleUser, "first"),
		mkMsg(RoleAssistant, "response1"),
		mkMsg(RoleUser, "second"),
		mkMsg(RoleAssistant, "response2"),
	}

	got := KeepLast(msgs, 2)

	if got[0].TextContent() != "sys" {
		t.Errorf("got[0] text = %q, want %q", got[0].TextContent(), "sys")
	}
	if got[1].TextContent() != "second" {
		t.Errorf("got[1] text = %q, want %q", got[1].TextContent(), "second")
	}
	if got[2].TextContent() != "response2" {
		t.Errorf("got[2] text = %q, want %q", got[2].TextContent(), "response2")
	}
}

func TestKeepLast_DoesNotMutateInput(t *testing.T) {
	t.Parallel()

	msgs := []Message{
		mkMsg(RoleSystem, "sys"),
		mkMsg(RoleUser, "a"),
		mkMsg(RoleUser, "b"),
	}
	original := make([]Message, len(msgs))
	copy(original, msgs)

	_ = KeepLast(msgs, 1)

	for i := range msgs {
		if msgs[i].TextContent() != original[i].TextContent() {
			t.Errorf("input msgs[%d] was mutated", i)
		}
	}
}

func TestTokenBudget(t *testing.T) {
	t.Parallel()

	// Simple counter: each message costs the total chars across all its
	// text-bearing blocks.
	charCount := func(msgs []Message) int {
		total := 0
		for _, m := range msgs {
			for _, b := range m.Blocks {
				total += len(b.Text)
			}
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
				mkMsg(RoleSystem, "sys"),
				mkMsg(RoleUser, "hi"),
			},
			maxTokens: 100,
			wantLen:   2,
		},
		{
			name: "drops oldest non-system first",
			msgs: []Message{
				mkMsg(RoleSystem, "sys"),           // 3 chars
				mkMsg(RoleUser, "aaaaaaaaaa"),      // 10 chars
				mkMsg(RoleAssistant, "bbbbb"),      // 5 chars
				mkMsg(RoleUser, "cc"),              // 2 chars
			},
			maxTokens: 11, // need to drop "aaaaaaaaaa" (total would be 3+5+2=10)
			wantLen:   3,
		},
		{
			name: "only system messages remain",
			msgs: []Message{
				mkMsg(RoleSystem, "s"),
				mkMsg(RoleUser, "xxxxxxxxxx"),
				mkMsg(RoleUser, "yyyyyyyyyy"),
			},
			maxTokens: 1,
			wantLen:   1,
		},
		{
			name: "system messages never dropped",
			msgs: []Message{
				mkMsg(RoleSystem, "a very long system message that exceeds budget"),
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
					t.Logf("  [%d] %s: %q", i, m.Role, m.TextContent())
				}
			}
		})
	}
}

func TestTokenBudget_NilCount(t *testing.T) {
	t.Parallel()

	// With nil count, uses default chars/4 heuristic.
	msgs := []Message{
		mkMsg(RoleSystem, "sys"),
		mkMsg(RoleUser, "hello world this is a test message"),
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
		mkMsg(RoleSystem, "sys"),
		mkMsg(RoleUser, "aaaaaaaaaa"),
		mkMsg(RoleUser, "bb"),
	}
	original := make([]Message, len(msgs))
	copy(original, msgs)

	charCount := func(msgs []Message) int {
		total := 0
		for _, m := range msgs {
			for _, b := range m.Blocks {
				total += len(b.Text)
			}
		}
		return total
	}

	_ = TokenBudget(msgs, 5, charCount)

	for i := range msgs {
		if msgs[i].TextContent() != original[i].TextContent() {
			t.Errorf("input msgs[%d] was mutated", i)
		}
	}
}
