"""Tests verifying all critical infrastructure files exist and are valid."""

from pathlib import Path

import tomllib

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPyprojectToml:
    """Validate pyproject.toml structure and content."""

    def test_pyproject_toml_exists(self):
        """pyproject.toml must exist at project root."""
        assert (PROJECT_ROOT / "pyproject.toml").is_file()

    def test_pyproject_toml_valid_toml(self):
        """pyproject.toml must be valid TOML."""
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        assert "project" in data
        assert "build-system" in data

    def test_pyproject_toml_has_required_fields(self):
        """pyproject.toml must have name, version, dependencies."""
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        project = data["project"]
        assert project["name"] == "agtech-unified"
        assert project["version"] == "1.0.0"
        assert "dependencies" in project
        assert len(project["dependencies"]) > 0

    def test_pyproject_toml_dependencies_pinned(self):
        """All dependencies in pyproject.toml must be pinned with ==."""
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        deps = data["project"]["dependencies"]
        for dep in deps:
            assert "==" in dep, f"Dependency not pinned: {dep}"


class TestRequirementsTxt:
    """Validate requirements.txt structure and content."""

    def test_requirements_txt_exists(self):
        """requirements.txt must exist at project root."""
        assert (PROJECT_ROOT / "requirements.txt").is_file()

    def test_requirements_txt_has_pinned_deps(self):
        """requirements.txt must contain pinned dependencies."""
        content = (PROJECT_ROOT / "requirements.txt").read_text()
        lines = [l.strip() for l in content.splitlines() if l.strip() and not l.startswith("#")]
        assert len(lines) > 0
        for line in lines:
            assert "==" in line, f"Dependency not pinned: {line}"

    def test_requirements_txt_contains_core_deps(self):
        """requirements.txt must contain core runtime dependencies."""
        content = (PROJECT_ROOT / "requirements.txt").read_text()
        assert "fastapi" in content
        assert "pydantic" in content
        assert "numpy" in content
        assert "torch" in content


class TestLicense:
    """Validate LICENSE file."""

    def test_license_exists(self):
        """LICENSE file must exist at project root."""
        assert (PROJECT_ROOT / "LICENSE").is_file()

    def test_license_is_mit(self):
        """LICENSE must be MIT."""
        content = (PROJECT_ROOT / "LICENSE").read_text()
        assert "MIT License" in content
        assert "Permission is hereby granted, free of charge" in content


class TestDockerfile:
    """Validate Dockerfile structure and content."""

    def test_dockerfile_exists(self):
        """Dockerfile must exist at project root."""
        assert (PROJECT_ROOT / "Dockerfile").is_file()

    def test_dockerfile_uses_python_311_slim(self):
        """Dockerfile must use python:3.11-slim as base image."""
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "FROM python:3.11-slim" in content

    def test_dockerfile_has_workdir_and_copy(self):
        """Dockerfile must set WORKDIR and copy requirements."""
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "WORKDIR /app" in content
        assert "COPY requirements.txt" in content
        assert "pip install" in content


class TestDockerCompose:
    """Validate docker-compose.yml structure and content."""

    def test_docker_compose_exists(self):
        """docker-compose.yml must exist at project root."""
        assert (PROJECT_ROOT / "docker-compose.yml").is_file()

    def test_docker_compose_has_required_services(self):
        """docker-compose.yml must define app, redis, and timescaledb services."""
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        assert "app:" in content
        assert "redis:" in content
        assert "timescaledb:" in content

    def test_docker_compose_app_depends_on_services(self):
        """app service must depend on redis and timescaledb."""
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        assert "redis" in content
        assert "timescaledb" in content
        assert "depends_on" in content


class TestDockerignore:
    """Validate .dockerignore file."""

    def test_dockerignore_exists(self):
        """.dockerignore must exist at project root."""
        assert (PROJECT_ROOT / ".dockerignore").is_file()

    def test_dockerignore_excludes_git(self):
        """.dockerignore must exclude .git directory."""
        content = (PROJECT_ROOT / ".dockerignore").read_text()
        assert ".git" in content

    def test_dockerignore_excludes_pycache(self):
        """.dockerignore must exclude __pycache__."""
        content = (PROJECT_ROOT / ".dockerignore").read_text()
        assert "__pycache__" in content


class TestCIWorkflow:
    """Validate GitHub Actions CI workflow."""

    def test_ci_workflow_exists(self):
        """CI workflow file must exist."""
        ci_path = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
        assert ci_path.is_file()

    def test_ci_workflow_has_ruff(self):
        """CI workflow must run ruff linting."""
        content = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "ruff" in content

    def test_ci_workflow_has_pytest(self):
        """CI workflow must run pytest."""
        content = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "pytest" in content

    def test_ci_workflow_has_bandit(self):
        """CI workflow must run bandit security scan."""
        content = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "bandit" in content


class TestMakefile:
    """Validate Makefile structure and content."""

    def test_makefile_exists(self):
        """Makefile must exist at project root."""
        assert (PROJECT_ROOT / "Makefile").is_file()

    def test_makefile_has_install_target(self):
        """Makefile must have install target."""
        content = (PROJECT_ROOT / "Makefile").read_text()
        assert "install:" in content

    def test_makefile_has_test_target(self):
        """Makefile must have test target."""
        content = (PROJECT_ROOT / "Makefile").read_text()
        assert "test:" in content

    def test_makefile_has_lint_target(self):
        """Makefile must have lint target."""
        content = (PROJECT_ROOT / "Makefile").read_text()
        assert "lint:" in content

    def test_makefile_has_docker_targets(self):
        """Makefile must have docker build and up targets."""
        content = (PROJECT_ROOT / "Makefile").read_text()
        assert "docker-build:" in content
        assert "docker-up:" in content


class TestGapsDirectory:
    """Validate gaps/ directory exists for TDD workflow."""

    def test_gaps_directory_exists(self):
        """gaps/ directory must exist."""
        assert (PROJECT_ROOT / "gaps").is_dir()

    def test_gaps_directory_has_json_files(self):
        """gaps/ directory must contain gap analysis JSON files."""
        gaps_dir = PROJECT_ROOT / "gaps"
        json_files = list(gaps_dir.glob("*.json"))
        assert len(json_files) > 0
