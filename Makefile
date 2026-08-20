SHELL := /bin/bash

build:
	go build ./...

test:
	go test ./...

vet:
	go vet ./...

test-coverage:
	@echo "\n🧐 Running tests with coverage...\n"
	@go test -count=1 -coverpkg=./... ./... -coverprofile cover.out
	@grep -v '/stackllm/examples/' cover.out > cover.tmp
	@mv cover.tmp cover.out
	@go tool cover -func cover.out | grep -v '^total:'
	@tail -n +2 cover.out | awk '{ if (!($$1 in stmts)) stmts[$$1]=$$2; if ($$3+0 > cnt[$$1]+0) cnt[$$1]=$$3+0 } \
		END { for (k in stmts) { t+=stmts[k]; if (cnt[k]>0) c+=stmts[k] }; \
		line="══════════════════════════════════════════════════════════════"; \
		printf "\n%s\n  \033[1;32m●\033[0m Total coverage: \033[1;32m%.2f%%\033[0m  (%d of %d statements)\n%s\n\n", line, 100*c/t, c, t, line }'

.PHONY: build test vet test-coverage
