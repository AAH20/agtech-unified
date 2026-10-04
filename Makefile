.PHONY: install install-dev test lint format security docker-build docker-up docker-down clean

# Default Python interpreter
PYTHON := python3

# Install production dependencies
install:
	$(PYTHON) -m pip install -e .

# Install development dependencies (includes test + lint tools)
install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

# Run all tests
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# Run tests with coverage
test-cov:
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

# Lint with ruff
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

# Auto-format with ruff
format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Security scan with bandit
security:
	bandit -r src/ -ll

# Build Docker image
docker-build:
	docker build -t agtech-unified:latest .

# Start all services with docker compose
docker-up:
	docker compose up -d

# Stop all services
docker-down:
	docker compose down

# View logs
docker-logs:
	docker compose logs -f

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
