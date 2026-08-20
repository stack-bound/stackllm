package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

func TestLoadOpenAITokenRecord(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	t.Run("JSON record round-trips with expiry", func(t *testing.T) {
		t.Parallel()
		store := NewMemoryStore()
		expiry := time.Now().Add(time.Hour).Truncate(time.Second)
		in := openAITokenRecord{
			AccessToken:  "acc",
			RefreshToken: "ref",
			ExpiresAt:    expiry,
		}
		if err := saveOpenAITokenRecord(ctx, store, "k", in); err != nil {
			t.Fatalf("save: %v", err)
		}
		out, err := loadOpenAITokenRecord(ctx, store, "k")
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		if out.AccessToken != "acc" || out.RefreshToken != "ref" {
			t.Errorf("record = %+v", out)
		}
		if !out.ExpiresAt.Equal(expiry) {
			t.Errorf("ExpiresAt = %v, want %v — expiry must round-trip", out.ExpiresAt, expiry)
		}
	})

	t.Run("legacy bare token string", func(t *testing.T) {
		t.Parallel()
		store := NewMemoryStore()
		if err := store.Save(ctx, "k", "sk-legacy-raw"); err != nil {
			t.Fatalf("seed: %v", err)
		}
		out, err := loadOpenAITokenRecord(ctx, store, "k")
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		if out.AccessToken != "sk-legacy-raw" {
			t.Errorf("AccessToken = %q, want the raw legacy value", out.AccessToken)
		}
		if !out.ExpiresAt.IsZero() {
			t.Errorf("legacy record ExpiresAt = %v, want zero (never expires)", out.ExpiresAt)
		}
	})

	t.Run("empty stored value", func(t *testing.T) {
		t.Parallel()
		store := NewMemoryStore()
		if err := store.Save(ctx, "k", ""); err != nil {
			t.Fatalf("seed: %v", err)
		}
		if _, err := loadOpenAITokenRecord(ctx, store, "k"); err == nil {
			t.Error("expected error for empty token record")
		}
	})

	t.Run("missing key", func(t *testing.T) {
		t.Parallel()
		store := NewMemoryStore()
		if _, err := loadOpenAITokenRecord(ctx, store, "absent"); err == nil {
			t.Error("expected error for missing key")
		}
	})
}

func TestExchangeOpenAIToken(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	t.Run("success computes expiry from expires_in", func(t *testing.T) {
		t.Parallel()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewEncoder(w).Encode(map[string]any{
				"access_token":  "tok",
				"refresh_token": "ref",
				"expires_in":    1800,
			})
		}))
		defer srv.Close()

		before := time.Now()
		rec, err := exchangeOpenAIToken(ctx, srv.Client(), srv.URL, url.Values{"grant_type": {"authorization_code"}})
		if err != nil {
			t.Fatalf("exchange: %v", err)
		}
		if rec.AccessToken != "tok" || rec.RefreshToken != "ref" {
			t.Errorf("record = %+v", rec)
		}
		if rec.ExpiresAt.Before(before.Add(1700 * time.Second)) {
			t.Errorf("ExpiresAt = %v — expires_in=1800 not honoured (too early)", rec.ExpiresAt)
		}
		if rec.ExpiresAt.After(before.Add(1900 * time.Second)) {
			t.Errorf("ExpiresAt = %v — expires_in=1800 not honoured (too late)", rec.ExpiresAt)
		}
	})

	t.Run("no expires_in means never expires", func(t *testing.T) {
		t.Parallel()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewEncoder(w).Encode(map[string]any{"access_token": "tok"})
		}))
		defer srv.Close()

		rec, err := exchangeOpenAIToken(ctx, srv.Client(), srv.URL, url.Values{})
		if err != nil {
			t.Fatalf("exchange: %v", err)
		}
		if !rec.ExpiresAt.IsZero() {
			t.Errorf("ExpiresAt = %v, want zero when expires_in absent", rec.ExpiresAt)
		}
	})

	errorTests := []struct {
		name    string
		handler http.HandlerFunc
		wantErr string
	}{
		{
			name: "non-200 status",
			handler: func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, "denied", http.StatusForbidden)
			},
			wantErr: "status 403",
		},
		{
			name: "empty access token",
			handler: func(w http.ResponseWriter, r *http.Request) {
				json.NewEncoder(w).Encode(map[string]any{"expires_in": 60})
			},
			wantErr: "empty access token",
		},
		{
			name: "invalid JSON body",
			handler: func(w http.ResponseWriter, r *http.Request) {
				fmt.Fprint(w, "not-json{")
			},
			wantErr: "decode token response",
		},
	}
	for _, tt := range errorTests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(tt.handler)
			defer srv.Close()

			_, err := exchangeOpenAIToken(ctx, srv.Client(), srv.URL, url.Values{})
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error = %q, want substring %q", err, tt.wantErr)
			}
		})
	}
}
