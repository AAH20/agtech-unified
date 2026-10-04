"""Health checks for onboarding modules.

Provides post-install verification that modules are functional,
including dependency validation and overall system readiness.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Callable

from src.onboarding.dependencies import ModuleDependencyGraph
from src.onboarding.modules import ModuleRegistry


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck:
    """Result of a single health check."""

    def __init__(
        self,
        name: str,
        status: HealthStatus,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.status = status
        self.message = message
        self.details = details or {}

    def is_healthy(self) -> bool:
        """Return True if the check passed."""
        return self.status == HealthStatus.HEALTHY

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


class ModuleHealthChecker:
    """Checks health of individual onboarding modules."""

    def __init__(self) -> None:
        self._registry = ModuleRegistry()
        self._graph = ModuleDependencyGraph.from_registry(self._registry)
        self._module_checks: dict[str, Callable[[], HealthCheck]] = {}
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default health checks for all known modules."""
        for tier in self._registry.all_tiers():
            for module in self._registry.get_modules(tier):
                self._module_checks[module] = lambda mod=module: self._check_module(mod)

    def check_module(self, module_name: str) -> HealthCheck:
        """Run health check for a single module.

        Args:
            module_name: Name of the module to check.

        Returns:
            HealthCheck result.
        """
        if module_name not in self._module_checks:
            return HealthCheck(
                name=module_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Unknown module: {module_name}",
            )
        return self._module_checks[module_name]()

    def check_all_modules(self) -> dict[str, HealthCheck]:
        """Run health checks for all known modules.

        Returns:
            Dictionary mapping module names to HealthCheck results.
        """
        return {name: check() for name, check in self._module_checks.items()}

    def get_overall_status(self) -> HealthStatus:
        """Get the overall health status across all modules.

        Returns:
            HEALTHY if all modules are healthy,
            UNHEALTHY if any module is unhealthy,
            DEGRADED if any module is degraded (but none unhealthy).
        """
        results = self.check_all_modules()
        statuses = [r.status for r in results.values()]

        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def _check_module(self, module_name: str) -> HealthCheck:
        """Default health check for a module.

        Verifies the module exists in the registry and its dependencies
        are satisfied.

        Args:
            module_name: Name of the module.

        Returns:
            HealthCheck result.
        """
        # Verify module exists in registry
        all_modules: set[str] = set()
        for tier in self._registry.all_tiers():
            all_modules.update(self._registry.get_modules(tier))

        if module_name not in all_modules:
            return HealthCheck(
                name=module_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Module '{module_name}' not found in registry",
            )

        # Check dependencies
        deps = self._graph.get_dependencies(module_name)
        dep_results: dict[str, str] = {}
        all_deps_healthy = True

        for dep in deps:
            dep_check = self.check_module(dep)
            dep_results[dep] = dep_check.status.value
            if not dep_check.is_healthy():
                all_deps_healthy = False

        if not all_deps_healthy:
            return HealthCheck(
                name=module_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Module '{module_name}' has unhealthy dependencies",
                details={"dependencies": dep_results},
            )

        return HealthCheck(
            name=module_name,
            status=HealthStatus.HEALTHY,
            message=f"Module '{module_name}' is functional",
            details={"dependencies": dep_results} if dep_results else None,
        )


class SystemHealth:
    """Overall system health monitor for onboarding."""

    def __init__(self, checker: ModuleHealthChecker | None = None) -> None:
        self._checker = checker or ModuleHealthChecker()

    def is_healthy(self) -> bool:
        """Return True if the system is healthy."""
        return self.get_status() == HealthStatus.HEALTHY

    def get_status(self) -> HealthStatus:
        """Get the overall system health status."""
        return self._checker.get_overall_status()

    def run_checks(self) -> dict[str, HealthCheck]:
        """Run all health checks.

        Returns:
            Dictionary mapping check names to HealthCheck results.
        """
        return self._checker.check_all_modules()

    def get_report(self) -> dict[str, Any]:
        """Get a full health report.

        Returns:
            Dictionary with status, checks, and timestamp.
        """
        checks = self.run_checks()
        return {
            "status": self.get_status().value,
            "checks": {name: check.to_dict() for name, check in checks.items()},
            "timestamp": time.time(),
        }
