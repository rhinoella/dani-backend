# ==============================================================================
# DANI Engine - Development & Deployment Makefile
# ==============================================================================

.PHONY: help setup env dev prod up down logs clean test lint build

# Default target
help:
	@echo "DANI Engine - Available Commands:"
	@echo ""
	@echo "  Setup:"
	@echo "    make setup     - Initial setup (creates .env, installs deps)"
	@echo "    make env       - Create .env from .env.example (won't overwrite)"
	@echo ""
	@echo "  Development:"
	@echo "    make dev       - Start development environment (with hot reload)"
	@echo "    make up        - Alias for 'make dev'"
	@echo "    make down      - Stop all containers"
	@echo "    make logs      - Tail logs from all containers"
	@echo "    make restart   - Restart all containers"
	@echo ""
	@echo "  Production:"
	@echo "    make prod      - Start production environment (multi-worker)"
	@echo "    make build     - Build production Docker image"
	@echo ""
	@echo "  Testing:"
	@echo "    make test      - Run all tests"
	@echo "    make test-fast - Run tests (skip slow integration tests)"
	@echo "    make lint      - Run linter"
	@echo ""
	@echo "  Database:"
	@echo "    make migrate   - Run database migrations"
	@echo "    make db-reset  - Reset database (DESTRUCTIVE)"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean     - Remove containers and volumes"

# ==============================================================================
# Setup
# ==============================================================================

# Create .env from example (won't overwrite existing)
env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅ Created .env from .env.example"; \
		echo "📝 Please edit .env with your configuration"; \
	else \
		echo "⚠️  .env already exists, skipping (use 'make env-force' to overwrite)"; \
	fi

# Force create .env (will overwrite)
env-force:
	cp .env.example .env
	@echo "✅ Created .env from .env.example (overwritten)"
	@echo "📝 Please edit .env with your configuration"

# Full setup
setup: env
	@echo ""
	@echo "🔧 Setting up DANI Engine..."
	@if [ -f requirements.txt ]; then \
		echo "📦 Installing Python dependencies..."; \
		pip install -r requirements.txt; \
	fi
	@echo ""
	@echo "✅ Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Edit .env with your configuration"
	@echo "  2. Run 'make dev' to start development environment"

# ==============================================================================
# Development
# ==============================================================================

dev: env
	cd docker && docker compose --profile dev up -d
	@echo ""
	@echo "✅ Development environment started!"
	@echo "   API: http://localhost:8000"
	@echo "   Docs: http://localhost:8000/docs"

up: dev

down:
	cd docker && docker compose --profile dev --profile prod down

logs:
	cd docker && docker compose logs -f

restart: down dev

# ==============================================================================
# Production
# ==============================================================================

build:
	cd docker && docker compose build api-prod

prod: env build
	cd docker && docker compose --profile prod up -d
	@echo ""
	@echo "✅ Production environment started!"
	@echo "   API: http://localhost:8000"

# ==============================================================================
# Testing
# ==============================================================================

test:
	python -m pytest tests/ -v

test-fast:
	python -m pytest tests/ -v -m "not integration and not performance"

lint:
	python -m ruff check app/
	python -m mypy app/ --ignore-missing-imports

# ==============================================================================
# Database
# ==============================================================================

migrate:
	alembic upgrade head
	@echo "✅ Migrations applied"

db-reset:
	@echo "⚠️  This will DESTROY all data in the database!"
	@read -p "Are you sure? (y/N) " confirm && [ "$$confirm" = "y" ]
	alembic downgrade base
	alembic upgrade head
	@echo "✅ Database reset complete"

# ==============================================================================
# Cleanup
# ==============================================================================

clean:
	cd docker && docker compose --profile dev --profile prod down -v
	@echo "✅ Containers and volumes removed"
