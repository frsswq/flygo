.PHONY: dev api build static browser check

dev: ## Run the browser app on Vite. No backend is needed.
	@npm --prefix web run dev

api: ## Run the FastAPI reference viewer in front of the built app
	@uv run fastapi dev

build: ## Build the static app for the FastAPI mount at /static
	@npm --prefix web run build

static: ## Build the deployable static app for a host that serves the root
	@env FLYGO_PUBLIC_BASE=/ FLYGO_OUT_DIR=dist npm --prefix web run build

browser: ## Run the browser regression when Chrome or Chromium is available
	@if [ -n "$$BROWSER_BIN" ] || ls "$$HOME"/.cache/ms-playwright/chromium-*/chrome-linux64/chrome >/dev/null 2>&1 || command -v chromium google-chrome >/dev/null 2>&1; then npm --prefix web run test:browser; else echo "browser regression skipped: no Chrome or Chromium found (set BROWSER_BIN)"; fi

check: ## Run every Python and web check
	@uv run ruff format --check .
	@uv run ruff check .
	@uv run vulture
	@uv run basedpyright
	@uv run pytest
	@npm --prefix web run check
	@npm --prefix web run typecheck
	@npm --prefix web run knip
	@npm --prefix web test
	@$(MAKE) --no-print-directory browser
