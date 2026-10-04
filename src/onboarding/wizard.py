"""Guided setup wizard for onboarding.

Provides a step-by-step wizard that collects organization profile
information, recommends a tier, and generates setup instructions.
"""

from __future__ import annotations

from typing import Any

from src.onboarding.dependencies import ModuleDependencyGraph
from src.onboarding.modules import ModuleRegistry
from src.onboarding.sizing import OrganizationProfiler
from src.onboarding.templates import ConfigTemplate


class SetupWizard:
    """Interactive setup wizard for new onboarding users.

    Collects organization profile data, recommends a tier using
    multi-factor scoring, and produces ordered setup steps and
    configuration templates.
    """

    def __init__(self) -> None:
        """Initialize the wizard with default components."""
        self._profiler = OrganizationProfiler()
        self._registry = ModuleRegistry()
        self._templates = ConfigTemplate()
        self._graph = ModuleDependencyGraph.from_registry(self._registry)
        self._profile: dict[str, Any] | None = None

    def set_organization_profile(
        self,
        employees: int,
        revenue: float,
        fields: int = 0,
        sensors: int = 0,
        has_dedicated_it: bool = False,
    ) -> None:
        """Set the organization profile for the wizard.

        Args:
            employees: Number of employees.
            revenue: Annual revenue in USD.
            fields: Number of fields/managed plots.
            sensors: Number of deployed or planned sensors.
            has_dedicated_it: Whether the org has dedicated IT staff.
        """
        self._profile = {
            "employees": employees,
            "revenue": revenue,
            "fields": fields,
            "sensors": sensors,
            "has_dedicated_it": has_dedicated_it,
        }

    def is_complete(self) -> bool:
        """Check if the wizard has all required profile data.

        Returns:
            True if the organization profile has been set.
        """
        return self._profile is not None

    def get_recommended_tier(self) -> str:
        """Get the recommended tier based on the organization profile.

        Returns:
            Recommended tier name.

        Raises:
            RuntimeError: If the profile has not been set.
        """
        if self._profile is None:
            raise RuntimeError("Organization profile not set")
        return self._profiler.recommend_tier_scored(**self._profile)

    def get_setup_steps(self) -> list[str]:
        """Get ordered setup steps for the recommended tier.

        Returns:
            List of module names in install order.

        Raises:
            RuntimeError: If the profile has not been set.
        """
        if self._profile is None:
            raise RuntimeError("Organization profile not set")
        tier = self.get_recommended_tier()
        modules = self._registry.get_modules(tier)
        return self._graph.get_install_order(modules)

    def get_config_template(self) -> dict[str, Any]:
        """Get the configuration template for the recommended tier.

        Returns:
            Configuration dictionary.

        Raises:
            RuntimeError: If the profile has not been set.
        """
        if self._profile is None:
            raise RuntimeError("Organization profile not set")
        tier = self.get_recommended_tier()
        return self._templates.get_template(tier)

    def validate_setup(self, config: dict[str, Any]) -> list[str]:
        """Validate a setup configuration.

        Args:
            config: Configuration dictionary to validate.

        Returns:
            List of error messages (empty if valid).
        """
        return self._templates.validate_template(config)

    def get_progress(self) -> float:
        """Get wizard completion progress.

        Returns:
            Progress as a float from 0.0 to 1.0.
        """
        if self._profile is None:
            return 0.0
        return 1.0

    def reset(self) -> None:
        """Reset the wizard to its initial state."""
        self._profile = None
