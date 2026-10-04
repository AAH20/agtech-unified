# Contributing to AgTech Unified

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/AAH20/agtech-unified.git
cd agtech-unified

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[test,dev]"

# Run tests
pytest
```

## Code Style

- **Formatter**: [ruff](https://docs.astral.sh/ruff/) (line length 100)
- **Linter**: ruff with rules: E, F, I, N, W
- **Type hints**: All public functions must have type annotations
- **Docstrings**: Google-style docstrings for all public classes and methods

```bash
# Format and lint
ruff format .
ruff check .
```

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add GPU-accelerated VRP solver
fix: handle empty TSP instance edge case
docs: expand deployment guide
test: add property-based tests for consensus
refactor: extract validation logic from data pipeline
```

## Pull Request Process

1. **Fork and branch**: Create a feature branch from `main`
2. **Write tests first** (TDD): Add tests before implementation
3. **Implement**: Make your changes
4. **Verify**: Ensure all tests pass (`pytest`)
5. **Lint**: Run `ruff format . && ruff check .`
6. **Document**: Update relevant docs (README, docstrings, tutorials)
7. **Submit**: Open a PR with a clear description

## Testing Requirements

- All new features must have tests
- Bug fixes must include a regression test
- Target 90%+ code coverage
- Tests must pass on Python 3.10, 3.11, and 3.12

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Getting Help

- Open a [GitHub Discussion](https://github.com/AAH20/agtech-unified/discussions) for questions
- Join our [Discord](https://discord.gg/agtech-unified) for real-time chat
- Check the [documentation](docs/) for guides
