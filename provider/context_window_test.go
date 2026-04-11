package provider

import "testing"

// TestContextWindow_KnownFamilies walks the exported table and asserts
// every entry returns itself under a direct prefix lookup. It also
// checks that a versioned suffix resolves to the same value so the
// matcher is actually prefix-based, and that case-insensitive input
// still resolves.
func TestContextWindow_KnownFamilies(t *testing.T) {
	t.Parallel()

	table := KnownContextWindows()
	if len(table) == 0 {
		t.Fatal("KnownContextWindows() is empty")
	}

	for _, entry := range table {
		entry := entry
		t.Run(entry.Prefix, func(t *testing.T) {
			t.Parallel()
			if got := ContextWindow(entry.Prefix); got != entry.Tokens {
				t.Errorf("ContextWindow(%q) = %d, want %d", entry.Prefix, got, entry.Tokens)
			}
			// Versioned suffix should collapse to the same family.
			if got := ContextWindow(entry.Prefix + "-2024-08-06"); got != entry.Tokens {
				t.Errorf("ContextWindow(%q suffix) = %d, want %d", entry.Prefix, got, entry.Tokens)
			}
			// Case insensitive.
			upper := ""
			for _, r := range entry.Prefix {
				if r >= 'a' && r <= 'z' {
					upper += string(r - 32)
				} else {
					upper += string(r)
				}
			}
			if got := ContextWindow(upper); got != entry.Tokens {
				t.Errorf("ContextWindow(%q upper) = %d, want %d", upper, got, entry.Tokens)
			}
		})
	}
}

// TestContextWindow_Unknown verifies the "unknown" sentinel path: an
// empty string and a model name that doesn't match any prefix must
// return 0 so callers can degrade their UI accordingly.
func TestContextWindow_Unknown(t *testing.T) {
	t.Parallel()

	cases := []string{
		"",
		"totally-made-up-model-xyz",
		"zzz-not-a-real-prefix",
	}
	for _, name := range cases {
		if got := ContextWindow(name); got != 0 {
			t.Errorf("ContextWindow(%q) = %d, want 0", name, got)
		}
	}
}

// TestContextWindow_SpecificityOrder checks that longer / more
// specific prefixes beat shorter stems sharing a common root. For
// example gpt-4o-mini must NOT resolve to whatever gpt-4 returns if
// the two happen to differ in the table.
func TestContextWindow_SpecificityOrder(t *testing.T) {
	t.Parallel()

	// Fetch the shared table once so we can cross-check without
	// hardcoding values.
	miniTokens := ContextWindow("gpt-4o-mini")
	if miniTokens == 0 {
		t.Fatal("gpt-4o-mini not in table; update the test")
	}
	fullTokens := ContextWindow("gpt-4o")
	if fullTokens == 0 {
		t.Fatal("gpt-4o not in table; update the test")
	}
	// Even if the values happen to be equal today, the point of
	// the test is that the mini prefix is matched first.
	if ContextWindow("gpt-4o-mini-2024-07-18") != miniTokens {
		t.Errorf("gpt-4o-mini-2024-07-18 did not resolve to gpt-4o-mini entry")
	}
	if ContextWindow("gpt-4o-2024-08-06") != fullTokens {
		t.Errorf("gpt-4o-2024-08-06 did not resolve to gpt-4o entry")
	}
}
