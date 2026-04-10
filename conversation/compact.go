package conversation

// KeepLast returns the last n non-system messages, always preserving all
// system messages in their original positions. If there are n or fewer
// non-system messages, the slice is returned unchanged.
func KeepLast(msgs []Message, n int) []Message {
	if n <= 0 {
		// Keep only system messages.
		var out []Message
		for _, m := range msgs {
			if m.IsSystem() {
				out = append(out, m)
			}
		}
		return out
	}

	// Count non-system messages.
	var nonSystem int
	for _, m := range msgs {
		if !m.IsSystem() {
			nonSystem++
		}
	}

	if nonSystem <= n {
		out := make([]Message, len(msgs))
		copy(out, msgs)
		return out
	}

	// Number of non-system messages to skip from the front.
	skip := nonSystem - n

	var out []Message
	skipped := 0
	for _, m := range msgs {
		if m.IsSystem() {
			out = append(out, m)
			continue
		}
		if skipped < skip {
			skipped++
			continue
		}
		out = append(out, m)
	}
	return out
}

// defaultTokenEstimate uses a simple chars/4 heuristic across every
// block's text-bearing fields. Images and raw artifact bytes are
// ignored — callers that need to account for image tokens should
// supply their own count function.
func defaultTokenEstimate(msgs []Message) int {
	total := 0
	for _, m := range msgs {
		for _, b := range m.Blocks {
			total += len(b.Text) / 4
			if b.Type == BlockToolUse {
				total += len(b.ToolName)/4 + len(b.ToolArgsJSON)/4
			}
		}
	}
	return total
}

// TokenBudget drops the oldest non-system messages until the estimated token
// count is at or below maxTokens. System messages are always preserved.
//
// If count is nil, a simple len(content)/4 heuristic is used.
// The caller can supply a precise tokeniser via the count parameter.
func TokenBudget(msgs []Message, maxTokens int, count func([]Message) int) []Message {
	if count == nil {
		count = defaultTokenEstimate
	}

	out := make([]Message, len(msgs))
	copy(out, msgs)

	for count(out) > maxTokens {
		// Find the first non-system message to drop.
		idx := -1
		for i, m := range out {
			if !m.IsSystem() {
				idx = i
				break
			}
		}
		if idx == -1 {
			// Only system messages remain; nothing more to drop.
			break
		}
		out = append(out[:idx], out[idx+1:]...)
	}

	return out
}
