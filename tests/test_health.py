"""Tests for onboarding health checks (src/onboarding/health.py).

Covers post-install verification that modules are functional:
module health checks, dependency validation, and overall system readiness.
"""

from __future__ import annotations

import pytest

from src.onboarding.health import HealthCheck, HealthStatus, ModuleHealthChecker, SystemHealth


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_status_values(self):
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"

    def test_status_comparison(self):
        assert HealthStatus.HEALTHY != HealthStatus.UNHEALTHY
        assert HealthStatus.HEALTHY == HealthStatus.HEALTHY


class TestHealthCheck:
    """Tests for HealthCheck dataclass."""

    def test_check_creation(self):
        check = HealthCheck(
            name="db_check",
            status=HealthStatus.HEALTHY,
            message="Database reachable",
        )
        assert check.name == "db_check"
        assert check.status == HealthStatus.HEALTHY
        assert check.message == "Database reachable"

    def test_check_with_details(self):
        check = HealthCheck(
            name="api_check",
            status=HealthStatus.DEGRADED,
            message="API slow",
            details={"latency_ms": 500},
        )
        assert check.details["latency_ms"] == 500

    def test_check_is_healthy(self):
        healthy = HealthCheck(name="x", status=HealthStatus.HEALTHY, message="ok")
        assert healthy.is_healthy() is True

    def test_check_is_not_healthy(self):
        unhealthy = HealthCheck(name="x", status=HealthStatus.UNHEALTHY, message="fail")
        assert unhealthy.is_healthy() is False

    def test_check_degraded_not_healthy(self):
        degraded = HealthCheck(name="x", status=HealthStatus.DEGRADED, message="slow")
        assert degraded.is_healthy() is False


class TestModuleHealthChecker:
    """Tests for ModuleHealthChecker."""

    @pytest.fixture
    def checker(self):
        return ModuleHealthChecker()

    def test_check_module_returns_healthcheck(self, checker):
        result = checker.check_module("basics")
        assert isinstance(result, HealthCheck)
        assert result.name == "basics"

    def test_check_module_healthy(self, checker):
        result = checker.check_module("basics")
        assert result.status == HealthStatus.HEALTHY

    def test_check_all_modules_returns_dict(self, checker):
        results = checker.check_all_modules()
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_check_all_modules_all_healthy(self, checker):
        results = checker.check_all_modules()
        for name, check in results.items():
            assert isinstance(check, HealthCheck)
            assert check.status == HealthStatus.HEALTHY, f"Module {name} is not healthy"

    def test_check_all_modules_covers_all_tiers(self, checker):
        """All modules from all tiers should be checked."""
        from src.onboarding.modules import ModuleRegistry

        registry = ModuleRegistry()
        results = checker.check_all_modules()
        for tier in registry.all_tiers():
            for mod in registry.get_modules(tier):
                assert mod in results, f"Module {mod} from tier {tier} not checked"

    def test_check_unknown_module(self, checker):
        result = checker.check_module("nonexistent_module")
        assert result.status == HealthStatus.UNHEALTHY
        assert "unknown" in result.message.lower() or "not found" in result.message.lower()

    def test_module_has_dependencies_checked(self, checker):
        """Modules with dependencies should verify them."""
        result = checker.check_module("analytics")
        assert isinstance(result, HealthCheck)
        # analytics depends on sensors, basics
        if result.details and "dependencies" in result.details:
            assert len(result.details["dependencies"]) > 0

    def test_overall_status_all_healthy(self, checker):
        status = checker.get_overall_status()
        assert status == HealthStatus.HEALTHY

    def test_overall_status_with_unhealthy(self, checker):
        """If any module is unhealthy, overall should be unhealthy."""
        checker._module_checks["broken"] = lambda: HealthCheck(
            name="broken", status=HealthStatus.UNHEALTHY, message="broken"
        )
        status = checker.get_overall_status()
        assert status == HealthStatus.UNHEALTHY

    def test_overall_status_with_degraded(self, checker):
        """If any module is degraded (but none unhealthy), overall is degraded."""
        checker._module_checks["slow"] = lambda: HealthCheck(
            name="slow", status=HealthStatus.DEGRADED, message="slow"
        )
        status = checker.get_overall_status()
        assert status == HealthStatus.DEGRADED


class TestSystemHealth:
    """Tests for SystemHealth."""

    @pytest.fixture
    def system(self):
        return SystemHealth()

    def test_system_initial_state(self, system):
        assert system.is_healthy() is True

    def test_system_run_checks(self, system):
        results = system.run_checks()
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_system_run_checks_all_healthy(self, system):
        results = system.run_checks()
        for name, check in results.items():
            assert isinstance(check, HealthCheck)
            assert check.status == HealthStatus.HEALTHY

    def test_system_get_status_healthy(self, system):
        status = system.get_status()
        assert status == HealthStatus.HEALTHY

    def test_system_get_status_with_unhealthy_module(self, system):
        """Injecting an unhealthy module should make system unhealthy."""
        checker = ModuleHealthChecker()
        checker._module_checks["broken"] = lambda: HealthCheck(
            name="broken", status=HealthStatus.UNHEALTHY, message="broken"
        )
        system = SystemHealth(checker=checker)
        status = system.get_status()
        assert status == HealthStatus.UNHEALTHY

    def test_system_is_healthy_false_when_unhealthy(self, system):
        checker = ModuleHealthChecker()
        checker._module_checks["broken"] = lambda: HealthCheck(
            name="broken", status=HealthStatus.UNHEALTHY, message="broken"
        )
        system = SystemHealth(checker=checker)
        assert system.is_healthy() is False

    def test_system_report(self, system):
        report = system.get_report()
        assert "status" in report
        assert "checks" in report
        assert "timestamp" in report

    def test_system_report_status_matches(self, system):
        report = system.get_report()
        assert report["status"] == system.get_status().value

    def test_system_report_checks_count(self, system):
        report = system.get_report()
        assert len(report["checks"]) > 0
