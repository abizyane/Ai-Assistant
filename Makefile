SHELL := /bin/bash
.DEFAULT_GOAL := help

# ─────────────────────────────────────────────────────────────────────────────
# Project metadata
# ─────────────────────────────────────────────────────────────────────────────
PROJECT     := 1337-rag-assistant
VERSION     := 1.0.0
COMPOSE     := docker compose
PYTHON      := python
UV          := uv

# Eval settings
GOLDEN_SET  := evals/golden_set.jsonl
EVAL_REPORT := evals/runs/$(shell date +%Y%m%d-%H%M%S).json

# Colours
C_RED    := \033[1;31m
C_GREEN  := \033[1;32m
C_YELLOW := \033[1;33m
C_CYAN   := \033[1;36m
C_RESET  := \033[0m

# ─────────────────────────────────────────────────────────────────────────────
# .PHONY declarations
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: help \
        up down logs ps build rebuild clean fclean state demo \
        lint format type typecheck \
        test test-unit test-integration test-e2e smoke eval \
        db-migrate db-rollback db-revision db-shell \
        traces metrics dashboards \
        install

# ─────────────────────────────────────────────────────────────────────────────
# help  (default target — auto-generated from ## comments)
# ─────────────────────────────────────────────────────────────────────────────
help:  ## Show this help message
	@printf "\n$(C_CYAN)"
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║        $(PROJECT) v$(VERSION)        ║"
	@echo "║   Production-grade multilingual agentic RAG assistant   ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@printf "$(C_RESET)\n"
	@echo "Usage:  make <target>"
	@echo ""
	@printf "$(C_GREEN)Stack lifecycle:$(C_RESET)\n"
	@grep -E '^(up|down|logs|ps|build|rebuild|clean|fclean|state|demo)[[:space:]]*:.*## ' $(MAKEFILE_LIST) \
	    | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-22s %s\n", $$1, $$2}'
	@echo ""
	@printf "$(C_GREEN)Quality gates:$(C_RESET)\n"
	@grep -E '^(lint|format|type|typecheck|test|test-unit|test-integration|test-e2e|smoke|eval)[[:space:]]*:.*## ' $(MAKEFILE_LIST) \
	    | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-22s %s\n", $$1, $$2}'
	@echo ""
	@printf "$(C_GREEN)Database / migrations:$(C_RESET)\n"
	@grep -E '^(db-migrate|db-rollback|db-revision|db-shell)[[:space:]]*:.*## ' $(MAKEFILE_LIST) \
	    | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-22s %s\n", $$1, $$2}'
	@echo ""
	@printf "$(C_GREEN)Observability:$(C_RESET)\n"
	@grep -E '^(traces|metrics|dashboards)[[:space:]]*:.*## ' $(MAKEFILE_LIST) \
	    | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-22s %s\n", $$1, $$2}'
	@echo ""
	@printf "$(C_GREEN)Development:$(C_RESET)\n"
	@grep -E '^(install)[[:space:]]*:.*## ' $(MAKEFILE_LIST) \
	    | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-22s %s\n", $$1, $$2}'
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Stack lifecycle
# ─────────────────────────────────────────────────────────────────────────────
build:  ## Build images incrementally, start the stack, and apply migrations
	@printf "$(C_GREEN)Building images…$(C_RESET)\n"
	$(COMPOSE) build
	@printf "$(C_GREEN)Starting stack…$(C_RESET)\n"
	$(COMPOSE) up -d
	@printf "$(C_YELLOW)Waiting for the database to be healthy…$(C_RESET)\n"
	@until $(COMPOSE) exec db pg_isready -U $${POSTGRES_USER:-raguser} -q 2>/dev/null; do sleep 2; done
	@printf "$(C_YELLOW)Applying database migrations…$(C_RESET)\n"
	$(COMPOSE) exec app alembic upgrade head
	@printf "$(C_GREEN)Setup complete — stack is ready.$(C_RESET)\n"

up:  ## Start the full stack (detached, healthchecks)
	@printf "$(C_GREEN)Starting stack…$(C_RESET)\n"
	$(COMPOSE) up -d
	@printf "$(C_GREEN)Stack is up.$(C_RESET)\n"

down:  ## Stop the stack (containers only, volumes preserved)
	@printf "$(C_RED)Stopping stack…$(C_RESET)\n"
	$(COMPOSE) down
	@printf "$(C_RED)Stack stopped.$(C_RESET)\n"

logs:  ## Tail all container logs (Ctrl-C to quit)
	$(COMPOSE) logs -f --tail=100

ps:  ## Show running containers and their health
	$(COMPOSE) ps

state:  ## Alias for ps
	$(COMPOSE) ps

clean:  ## Remove build artifacts (keep containers + data)
	@printf "$(C_YELLOW)Removing build artifacts…$(C_RESET)\n"
	rm -rf dist/ build/ *.egg-info
	find . -type d -name '__pycache__' -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -not -path './.venv/*' -delete 2>/dev/null || true
	@printf "$(C_YELLOW)Done.$(C_RESET)\n"

fclean:  ## Destroy EVERYTHING: containers, volumes, images, caches, runtime data
	@printf "$(C_RED)Full clean — removing all docker resources and caches…$(C_RESET)\n"
	$(COMPOSE) down -v --rmi all --remove-orphans 2>/dev/null || true
	docker volume prune -f
	docker image prune -f
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov \
	       dist/ build/ *.egg-info \
	       .coverage coverage.xml \
	       evals/runs/ logs/ *.log
	find . -type d -name '__pycache__' -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -not -path './.venv/*' -delete 2>/dev/null || true
	@printf "$(C_RED)Full clean complete.$(C_RESET)\n"

rebuild:  ## Full clean → build → up (useful after major changes)
	$(MAKE) fclean
	$(MAKE) build
	$(MAKE) up

demo:  ## One-shot hiring-manager demo: spin up + seed + print URLs
	@printf "$(C_CYAN)Starting demo stack…$(C_RESET)\n"
	$(MAKE) up
	@printf "$(C_YELLOW)Waiting 30s for services to become healthy…$(C_RESET)\n"
	sleep 30
	@printf "$(C_GREEN)Seeding demo documents…$(C_RESET)\n"
	$(PYTHON) scripts/seed_demo.py
	@printf "\n$(C_CYAN)╔══════════════════════════════════════════════════════════╗$(C_RESET)\n"
	@printf "$(C_CYAN)║  Demo ready — open any of these URLs:                   ║$(C_RESET)\n"
	@printf "$(C_CYAN)║  Chainlit UI   →  http://localhost:8502                 ║$(C_RESET)\n"
	@printf "$(C_CYAN)║  FastAPI docs  →  http://localhost:8000/docs            ║$(C_RESET)\n"
	@printf "$(C_CYAN)║  Langfuse      →  http://localhost:3000                 ║$(C_RESET)\n"
	@printf "$(C_CYAN)║  Grafana       →  http://localhost:3001                 ║$(C_RESET)\n"
	@printf "$(C_CYAN)║  Prometheus    →  http://localhost:9090                 ║$(C_RESET)\n"
	@printf "$(C_CYAN)╚══════════════════════════════════════════════════════════╝$(C_RESET)\n"

# ─────────────────────────────────────────────────────────────────────────────
# Quality gates
# ─────────────────────────────────────────────────────────────────────────────
lint:  ## Run ruff format check + ruff lint (no auto-fix)
	@printf "$(C_YELLOW)Running ruff format check…$(C_RESET)\n"
	ruff format --check src/ tests/
	@printf "$(C_YELLOW)Running ruff lint…$(C_RESET)\n"
	ruff check src/ tests/
	@printf "$(C_GREEN)Lint passed.$(C_RESET)\n"

format:  ## Auto-format with ruff (modifies files in-place)
	@printf "$(C_YELLOW)Formatting…$(C_RESET)\n"
	ruff format src/ tests/
	ruff check --fix src/ tests/
	@printf "$(C_GREEN)Format complete.$(C_RESET)\n"

typecheck:  ## Run mypy --strict on src/
	@printf "$(C_YELLOW)Running mypy --strict…$(C_RESET)\n"
	mypy --strict src/
	@printf "$(C_GREEN)Type-check passed.$(C_RESET)\n"

type: typecheck  ## Alias for typecheck

test:  ## Run full test suite with coverage report (threshold 80%)
	@printf "$(C_YELLOW)Running test suite…$(C_RESET)\n"
	pytest tests/ \
	    --cov=src \
	    --cov-report=term \
	    --cov-report=html \
	    --cov-fail-under=80 \
	    -v
	@printf "$(C_GREEN)Tests passed.$(C_RESET)\n"

test-unit:  ## Run unit tests only (fast, no I/O)
	@printf "$(C_YELLOW)Running unit tests…$(C_RESET)\n"
	pytest tests/unit/ -v
	@printf "$(C_GREEN)Unit tests passed.$(C_RESET)\n"

test-integration:  ## Run integration tests (requires Docker / testcontainers)
	@printf "$(C_YELLOW)Running integration tests…$(C_RESET)\n"
	pytest tests/integration/ -v --maxfail=1
	@printf "$(C_GREEN)Integration tests passed.$(C_RESET)\n"

test-e2e:  ## Run e2e tests against running stack (requires make up first)
	@printf "$(C_YELLOW)Running e2e tests — stack must be up…$(C_RESET)\n"
	pytest tests/e2e/ -v --maxfail=1
	@printf "$(C_GREEN)E2e tests passed.$(C_RESET)\n"

smoke:  ## Run smoke test against live stack (3 hard-coded questions)
	@printf "$(C_YELLOW)Running smoke test against live stack…$(C_RESET)\n"
	$(PYTHON) scripts/smoke.py
	@printf "$(C_GREEN)Smoke test passed.$(C_RESET)\n"

eval:  ## Run Ragas evaluation against golden set; fail if thresholds breached
	@printf "$(C_YELLOW)Running Ragas evaluation…$(C_RESET)\n"
	@mkdir -p evals/runs
	$(PYTHON) -m src.application.use_cases.evaluate \
	    --dataset $(GOLDEN_SET) \
	    --report $(EVAL_REPORT) \
	    --gate
	@printf "$(C_GREEN)Eval passed — report written to $(EVAL_REPORT)$(C_RESET)\n"

# ─────────────────────────────────────────────────────────────────────────────
# Database / migrations
# ─────────────────────────────────────────────────────────────────────────────
db-migrate:  ## Apply all pending Alembic migrations (upgrade head)
	@printf "$(C_YELLOW)Applying migrations…$(C_RESET)\n"
	$(COMPOSE) exec app alembic upgrade head
	@printf "$(C_GREEN)Migrations applied.$(C_RESET)\n"

db-rollback:  ## Roll back one migration (downgrade -1)
	@printf "$(C_RED)Rolling back last migration…$(C_RESET)\n"
	$(COMPOSE) exec app alembic downgrade -1
	@printf "$(C_RED)Rollback done.$(C_RESET)\n"

db-revision:  ## Generate a new autogenerate migration (MSG="description" required)
ifndef MSG
	$(error MSG is required — usage: make db-revision MSG="your migration description")
endif
	@printf "$(C_YELLOW)Generating migration: $(MSG)…$(C_RESET)\n"
	alembic revision --autogenerate -m "$(MSG)"
	@printf "$(C_GREEN)Migration file created.$(C_RESET)\n"

db-shell:  ## Open a psql shell inside the running db container
	@printf "$(C_YELLOW)Opening psql shell…$(C_RESET)\n"
	$(COMPOSE) exec db psql -U $${POSTGRES_USER:-raguser} -d $${POSTGRES_DB:-ragdb}

# ─────────────────────────────────────────────────────────────────────────────
# Observability
# ─────────────────────────────────────────────────────────────────────────────
traces:  ## Open Langfuse traces UI in your default browser
	@printf "$(C_CYAN)Opening Langfuse at http://localhost:3000…$(C_RESET)\n"
	@command -v xdg-open >/dev/null 2>&1 && xdg-open http://localhost:3000 || \
	 command -v open      >/dev/null 2>&1 && open      http://localhost:3000 || \
	 echo "Navigate to http://localhost:3000"

metrics:  ## Open Prometheus UI in your default browser
	@printf "$(C_CYAN)Opening Prometheus at http://localhost:9090…$(C_RESET)\n"
	@command -v xdg-open >/dev/null 2>&1 && xdg-open http://localhost:9090 || \
	 command -v open      >/dev/null 2>&1 && open      http://localhost:9090 || \
	 echo "Navigate to http://localhost:9090"

dashboards:  ## Open Grafana dashboards in your default browser
	@printf "$(C_CYAN)Opening Grafana at http://localhost:3001…$(C_RESET)\n"
	@command -v xdg-open >/dev/null 2>&1 && xdg-open http://localhost:3001 || \
	 command -v open      >/dev/null 2>&1 && open      http://localhost:3001 || \
	 echo "Navigate to http://localhost:3001"

# ─────────────────────────────────────────────────────────────────────────────
# Development utilities
# ─────────────────────────────────────────────────────────────────────────────
install:  ## Install all Python dependencies (including dev extras) with uv
	@printf "$(C_YELLOW)Installing dependencies…$(C_RESET)\n"
	$(UV) sync --all-extras --dev
	@printf "$(C_GREEN)Dependencies installed.$(C_RESET)\n"
