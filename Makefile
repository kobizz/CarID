.PHONY: dev start sync test flake8 pylint mypy bandit safety lint check format clean bump help

# Default target
help:
	@echo "Available commands:"
	@echo "  dev     - Start FastAPI server in development mode (auto-reload)"
	@echo "  start   - Start FastAPI server in production mode"
	@echo "  sync    - Sync local project with HA instance"
	@echo "  bump    - Bump version and sync (defaults to patch; usage: make bump [VERSION=1.2.3|patch|minor|major])"
	@echo "  test    - Run tests with pytest"
	@echo "  flake8  - Run flake8 linter"
	@echo "  pylint  - Run pylint linter"
	@echo "  mypy    - Run mypy type checker"
	@echo "  bandit  - Run bandit security scanner"
	@echo "  safety  - Run safety dependency scanner"
	@echo "  lint    - Run all linters (flake8 + pylint)"
	@echo "  check   - Run all quality checks (lint + type + security + test)"
	@echo "  format  - Format code with black"
	@echo "  clean   - Remove Python cache files"

# Development server with auto-reload
dev:
	uvicorn service.app:app --reload --host 0.0.0.0 --port 8000

# Production server
start:
	uvicorn service.app:app --host 0.0.0.0 --port 8000

# Sync with Home Assistant
sync:
	rsync -avz --delete --exclude-from=.syncignore . ha-ssh:/addons/carid

# Run tests
test:
	python -m pytest tests/ -v

# Lint code with flake8
flake8:
	python -m flake8 service/

# Lint code with pylint
pylint:
	python -m pylint service/

# Run mypy type checker
mypy:
	python -m mypy service/ --ignore-missing-imports

# Run bandit security scanner
bandit:
	python -m bandit -r service/

# Run safety dependency scanner  
safety:
	python -m safety check

# Run all linters
lint: flake8 pylint

# Run all quality checks
check: lint mypy bandit safety test

# Format code
format:
	python -m black service/

# Clean up Python cache files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

# Bump version and sync to HA
bump:
	$(eval VERSION ?= patch)
	@echo "Bumping version ($(VERSION))..."
	@python3 scripts/bump_version.py $(VERSION)
	@echo "Syncing to Home Assistant..."
	@make sync
	@echo "✅ Version bump and sync completed!"

