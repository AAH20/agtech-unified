"""Configuration templates for onboarding tiers.

Provides tier-specific configuration templates with validation.
"""

from __future__ import annotations

from typing import Any

from src.onboarding.modules import ModuleRegistry


class ConfigTemplate:
    """Tier-specific configuration templates.

    Each tier (startup, smb, enterprise) has a predefined
    configuration with modules, features, database, and API
    settings appropriate for that scale.
    """

    _REQUIRED_KEYS = ("tier", "modules", "features", "database", "api")

    def __init__(self) -> None:
        """Initialize with a ModuleRegistry for module lookups."""
        self._registry = ModuleRegistry()

    def get_template(self, tier: str) -> dict[str, Any]:
        """Return a configuration template for the given tier.

        Args:
            tier: One of 'startup', 'smb', 'enterprise'.

        Returns:
            Configuration dictionary.

        Raises:
            ValueError: If the tier is not recognized.
        """
        if tier not in self._registry.all_tiers():
            raise ValueError(f"Unknown tier: {tier}")

        modules = self._registry.get_modules(tier)
        features = self._get_features(tier)
        database = self._get_database_config(tier)
        api = self._get_api_config(tier)

        return {
            "tier": tier,
            "modules": modules,
            "features": features,
            "database": database,
            "api": api,
        }

    def get_all_templates(self) -> dict[str, dict[str, Any]]:
        """Return all tier templates.

        Returns:
            Dictionary mapping tier names to config dictionaries.
        """
        return {tier: self.get_template(tier) for tier in self._registry.all_tiers()}

    def validate_template(self, config: dict[str, Any]) -> list[str]:
        """Validate a configuration template.

        Args:
            config: Configuration dictionary to validate.

        Returns:
            List of error messages (empty if valid).
        """
        errors: list[str] = []

        for key in self._REQUIRED_KEYS:
            if key not in config:
                errors.append(f"Missing required key: {key}")

        if "modules" in config and not isinstance(config["modules"], list):
            errors.append("'modules' must be a list")

        if "features" in config and not isinstance(config["features"], list):
            errors.append("'features' must be a list")

        if "tier" in config and config["tier"] not in self._registry.all_tiers():
            errors.append(f"Unknown tier: {config['tier']}")

        return errors

    def _get_features(self, tier: str) -> list[str]:
        """Return feature flags for a tier."""
        base = ["sensor_monitoring", "basic_alerts"]
        if tier == "startup":
            return base
        smb = base + ["analytics_dashboard", "reporting", "multi_field"]
        if tier == "smb":
            return smb
        return smb + ["integrations", "sla_monitoring", "advanced_analytics", "api_access"]

    def _get_database_config(self, tier: str) -> dict[str, Any]:
        """Return database configuration for a tier."""
        if tier == "startup":
            return {"type": "sqlite", "path": "./data/agtech.db", "pool_size": 1}
        elif tier == "smb":
            return {"type": "postgresql", "host": "localhost", "port": 5432, "pool_size": 5}
        else:
            return {
                "type": "postgresql",
                "host": "localhost",
                "port": 5432,
                "pool_size": 20,
                "replicas": 2,
            }

    def _get_api_config(self, tier: str) -> dict[str, Any]:
        """Return API configuration for a tier."""
        if tier == "startup":
            return {"host": "localhost", "port": 8000, "workers": 1, "auth": "basic"}
        elif tier == "smb":
            return {"host": "0.0.0.0", "port": 8000, "workers": 4, "auth": "jwt"}
        else:
            return {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 8,
                "auth": "oauth2",
                "rate_limiting": True,
            }
