.PHONY: dev api build static check

dev: ## Run the browser app on Vite. No backend is needed.
	@npm --prefix web run dev

api: ## Run the FastAPI reference viewer in front of the built app
	@uv run fastapi dev

build: ## Build the static app for the FastAPI mount at /static
	@npm --prefix web run build

static: ## Build the deployable static app for a host that serves the root
	@env FLYGO_PUBLIC_BASE=/ FLYGO_OUT_DIR=dist npm --prefix web run build

check: ## Run every Python and web check
	@uv run ruff format --check .
	@uv run ruff check .
	@uv run basedpyright
	@uv run pytest
	@npm --prefix web run check
	@npm --prefix web run typecheck
	@npm --prefix web test
