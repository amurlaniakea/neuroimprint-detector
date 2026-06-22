# neuroimprint-detector Makefile

.PHONY: help install test test-verbose lint format typecheck check clean all

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in dev mode with all dev dependencies
	pip install -e ".[dev]"

test: ## Run all tests
	python -m pytest tests/ -v --tb=short

test-verbose: ## Run tests with coverage report
	python -m pytest tests/ -v --tb=long --cov=src --cov-report=term-missing

lint: ## Run ruff linter
	ruff check src tests

format: ## Format code with black + ruff
	black src tests
	ruff check --fix src tests

typecheck: ## Run mypy type checker
	mypy src

check: lint typecheck test ## Run all checks (lint + typecheck + test)

clean: ## Remove build artifacts and cache
	rm -rf build/ dist/ *.egg-info .mypy_cache .ruff_cache .pytest_cache
	find src tests -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find src tests -name "*.pyc" -delete 2>/dev/null || true

all: format check ## Format, lint, typecheck, and test
