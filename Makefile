.PHONY: dev api build static browser check label label-status label-finalize

dev: ## Run the browser app on Vite. No backend is needed.
	@npm --prefix web run dev

api: ## Run the FastAPI reference viewer in front of the built app
	@uv run fastapi dev

build: ## Build the static app for the FastAPI mount at /static
	@npm --prefix web run build

static: ## Build the deployable static app for a host that serves the root
	@env FLYGO_PUBLIC_BASE=/ FLYGO_OUT_DIR=dist npm --prefix web run build

browser: ## Run the browser regression when Chrome or Chromium is available
	@scripts/run_browser_tests.sh

label: ## Complete one resumable feasibility teacher shard
	@uv run python scripts/teacher_feasibility.py run --max-shards 1

label-status: ## Show feasibility teacher labelling progress
	@uv run python scripts/teacher_feasibility.py status
label-finalize: ## Build the verified dataset after every teacher shard completes
	@uv run python scripts/teacher_feasibility.py finalize

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
	@${MAKE} --no-print-directory browser
