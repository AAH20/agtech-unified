"""Tests for infrastructure gap fixes (INF-001 through INF-012).

Each test corresponds to a gap in gaps/infrastructure_gaps.json.
Tests are written FIRST (TDD) and will fail until the fix is applied.
"""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read(path: str) -> str:
    return (PROJECT_ROOT / path).read_text()


def _read_ci() -> str:
    return _read(".github/workflows/ci.yml")


def _read_compose() -> str:
    return _read("docker-compose.yml")


def _read_dockerfile() -> str:
    return _read("Dockerfile")


def _read_dockerignore() -> str:
    return _read(".dockerignore")


def _read_makefile() -> str:
    return _read("Makefile")


def _read_pyproject() -> dict:
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


# ── INF-001: CI pip caching ──────────────────────────────────────────


class TestINF001CIPipCaching:
    """INF-001: CI must cache pip dependencies."""

    def test_ci_has_cache_pip(self):
        """setup-python steps must include cache: pip."""
        content = _read_ci()
        assert "cache: pip" in content, "CI must enable pip caching via cache: pip"

    def test_ci_cache_in_setup_python(self):
        """cache: pip must appear in a setup-python step context."""
        content = _read_ci()
        # Find setup-python blocks and verify cache is nearby
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if "actions/setup-python" in line:
                # Check next few lines for cache: pip
                context = "\n".join(lines[i : i + 5])
                assert "cache: pip" in context, f"setup-python at line {i + 1} missing cache: pip"
                break
        else:
            raise AssertionError("No actions/setup-python found in CI")


# ── INF-002: CI concurrency group ─────────────────────────────────────


class TestINF002CIConcurrency:
    """INF-002: CI must have concurrency group to cancel redundant runs."""

    def test_ci_has_concurrency_block(self):
        """CI must define a concurrency block."""
        content = _read_ci()
        assert "concurrency:" in content, "CI must have concurrency block"

    def test_ci_concurrency_group_uses_ref(self):
        """Concurrency group must use github.ref."""
        content = _read_ci()
        assert "group:" in content
        assert "github.ref" in content, "Concurrency group must key on github.ref"

    def test_ci_concurrency_cancel_in_progress(self):
        """Concurrency must cancel in-progress runs."""
        content = _read_ci()
        assert "cancel-in-progress: true" in content


# ── INF-003: Python 3.13 in test matrix ───────────────────────────────


class TestINF003Python313:
    """INF-003: CI test matrix must include Python 3.10-3.12 (3.13 excluded due to onnx build failure)."""

    def test_ci_matrix_includes_supported_versions(self):
        """Test matrix must include Python 3.10, 3.11, 3.12."""
        content = _read_ci()
        assert '"3.10"' in content, "CI matrix must include Python 3.10"
        assert '"3.11"' in content, "CI matrix must include Python 3.11"
        assert '"3.12"' in content, "CI matrix must include Python 3.12"


# ── INF-004: Coverage reporting ───────────────────────────────────────


class TestINF004Coverage:
    """INF-004: CI must generate and upload coverage reports."""

    def test_ci_has_pytest_cov(self):
        """CI must install and use pytest-cov."""
        content = _read_ci()
        assert "pytest-cov" in content, "CI must use pytest-cov"

    def test_ci_has_codecov_upload(self):
        """CI must upload coverage to codecov."""
        content = _read_ci()
        assert "codecov" in content, "CI must upload coverage to codecov"

    def test_ci_coverage_flag_in_pytest(self):
        """pytest invocation must include --cov flag."""
        content = _read_ci()
        assert "--cov" in content, "pytest must run with --cov"


# ── INF-005: Multi-stage Docker build ─────────────────────────────────


class TestINF005MultiStageDocker:
    """INF-005: Dockerfile must use multi-stage build."""

    def test_dockerfile_has_multiple_from(self):
        """Dockerfile must have at least two FROM statements."""
        content = _read_dockerfile()
        from_count = content.count("FROM ")
        assert from_count >= 2, f"Dockerfile must have >=2 FROM stages, found {from_count}"

    def test_dockerfile_has_builder_stage(self):
        """Dockerfile must have a named builder stage."""
        content = _read_dockerfile()
        assert (
            "AS builder" in content or "as builder" in content.lower()
        ), "Dockerfile must have a builder stage"

    def test_dockerfile_runtime_stage_copies_from_builder(self):
        """Runtime stage must copy artifacts from builder."""
        content = _read_dockerfile()
        assert "COPY --from=builder" in content, "Runtime stage must COPY --from=builder"


# ── INF-006: .dockerignore for gaps/ ──────────────────────────────────


class TestINF006DockerignoreGaps:
    """INF-006: .dockerignore must exclude gaps/ directory."""

    def test_dockerignore_excludes_gaps(self):
        """.dockerignore must contain gaps/."""
        content = _read_dockerignore()
        assert "gaps/" in content, ".dockerignore must exclude gaps/"


# ── INF-007: .env credentials in compose ──────────────────────────────


class TestINF007EnvCredentials:
    """INF-007: docker-compose must use .env for credentials."""

    def test_compose_no_hardcoded_postgres_password(self):
        """Compose must not hardcode POSTGRES_PASSWORD."""
        content = _read_compose()
        assert "POSTGRES_PASSWORD=agtech" not in content, "POSTGRES_PASSWORD must not be hardcoded"

    def test_compose_uses_env_var_for_password(self):
        """Compose must use ${POSTGRES_PASSWORD:?} syntax."""
        content = _read_compose()
        assert (
            "${POSTGRES_PASSWORD" in content
        ), "Compose must reference POSTGRES_PASSWORD from .env"


# ── INF-008: GPU runtime in compose ──────────────────────────────────


class TestINF008GPURuntime:
    """INF-008: docker-compose must configure NVIDIA runtime for GPU."""

    def test_compose_app_has_nvidia_runtime(self):
        """app service must use runtime: nvidia."""
        content = _read_compose()
        assert "runtime: nvidia" in content, "app service must have runtime: nvidia for GPU support"


# ── INF-009: torch as optional dependency ────────────────────────────


class TestINF009TorchOptional:
    """INF-009: torch must be an optional dependency, not required."""

    def test_torch_not_in_main_dependencies(self):
        """torch must not be in project.dependencies."""
        data = _read_pyproject()
        deps = data["project"]["dependencies"]
        torch_deps = [d for d in deps if d.startswith("torch")]
        assert len(torch_deps) == 0, f"torch must not be a required dependency: {torch_deps}"

    def test_torch_in_optional_dependencies(self):
        """torch must be in [project.optional-dependencies] gpu."""
        data = _read_pyproject()
        optional = data["project"].get("optional-dependencies", {})
        assert "gpu" in optional, "Must have a 'gpu' optional dependency group"
        gpu_deps = optional["gpu"]
        torch_deps = [d for d in gpu_deps if d.startswith("torch")]
        assert len(torch_deps) > 0, "torch must be in the gpu optional group"


# ── INF-010: Consistent version pinning ────────────────────────────────


class TestINF010ConsistentPinning:
    """INF-010: rdflib must have consistent version bounds."""

    def test_rdflib_has_upper_bound(self):
        """rdflib must have an upper version bound."""
        data = _read_pyproject()
        optional = data["project"].get("optional-dependencies", {})
        all_deps = []
        for group in optional.values():
            all_deps.extend(group)
        rdflib_deps = [d for d in all_deps if d.startswith("rdflib")]
        assert len(rdflib_deps) > 0, "rdflib must be in optional dependencies"
        for dep in rdflib_deps:
            assert "<" in dep or "<=" in dep, f"rdflib must have upper bound: {dep}"


# ── INF-011: docker-test target in Makefile ──────────────────────────


class TestINF011DockerTestTarget:
    """INF-011: Makefile must have docker-test target."""

    def test_makefile_has_docker_test(self):
        """Makefile must define docker-test target."""
        content = _read_makefile()
        assert "docker-test:" in content, "Makefile must have docker-test target"

    def test_makefile_docker_test_runs_pytest(self):
        """docker-test target must run pytest in container."""
        content = _read_makefile()
        # Find the docker-test block
        lines = content.splitlines()
        in_target = False
        for line in lines:
            if line.startswith("docker-test:"):
                in_target = True
                continue
            if in_target:
                if line and not line.startswith("\t") and not line.startswith(" "):
                    break
                if "pytest" in line:
                    return
        raise AssertionError("docker-test target must run pytest")


# ── INF-012: Scheduled CI runs ────────────────────────────────────────


class TestINF012ScheduledCI:
    """INF-012: CI must have scheduled (cron) runs."""

    def test_ci_has_schedule_trigger(self):
        """CI must have a schedule trigger."""
        content = _read_ci()
        assert "schedule:" in content, "CI must have schedule trigger"

    def test_ci_has_cron_expression(self):
        """CI must have a cron expression."""
        content = _read_ci()
        assert "cron:" in content, "CI must have cron expression"
