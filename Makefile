.PHONY: dev start sync test lint format clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  dev     - Start FastAPI server in development mode (auto-reload)"
	@echo "  start   - Start FastAPI server in production mode"
	@echo "  sync    - Sync local project with HA instance"
	@echo "  test    - Run tests with pytest"
	@echo "  lint    - Run flake8 linter"
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
	rsync -avz --delete --exclude-from=.syncignore . ha-ssh:/addons/parking

# Run tests
test:
	python -m pytest tests/ -v

# Lint code
lint:
	python -m flake8 service/

# Format code
format:
	python -m black service/

# Clean up Python cache files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

