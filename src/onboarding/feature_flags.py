"""Feature flags for onboarding modules.

Provides gradual rollout, user-bucketing, and per-environment overrides
for onboarding features.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeatureFlag:
    """A single feature flag with rollout and environment override support."""

    name: str
    default_enabled: bool = False
    rollout_percentage: int = 0
    environment_overrides: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.rollout_percentage <= 100:
            raise ValueError(f"rollout_percentage must be 0-100, got {self.rollout_percentage}")


class FeatureFlagManager:
    """Manages feature flags with rollout, bucketing, and environment overrides."""

    def __init__(self) -> None:
        self._flags: dict[str, FeatureFlag] = {}
        self._env_overrides: dict[str, dict[str, bool]] = {}

    def register_flag(self, flag: FeatureFlag) -> None:
        """Register a feature flag.

        Args:
            flag: The FeatureFlag to register.

        Raises:
            ValueError: If a flag with the same name is already registered.
        """
        if flag.name in self._flags:
            raise ValueError(f"Flag '{flag.name}' is already registered")
        self._flags[flag.name] = flag
        self._env_overrides[flag.name] = dict(flag.environment_overrides)

    def is_enabled(
        self,
        flag_name: str,
        user_id: str | None = None,
        environment: str | None = None,
    ) -> bool:
        """Check if a feature flag is enabled.

        Args:
            flag_name: Name of the flag.
            user_id: Optional user identifier for bucketing.
            environment: Optional environment name for overrides.

        Returns:
            True if the flag is enabled for the given context.

        Raises:
            KeyError: If the flag is not registered.
        """
        if flag_name not in self._flags:
            raise KeyError(f"Unknown flag: {flag_name}")

        flag = self._flags[flag_name]

        # Environment override takes highest priority
        if environment and environment in self._env_overrides.get(flag_name, {}):
            return self._env_overrides[flag_name][environment]

        # Rollout percentage determines bucketing
        if flag.rollout_percentage == 0:
            return False
        if flag.rollout_percentage == 100:
            return True

        # If no user_id, fall back to default_enabled
        if user_id is None:
            return flag.default_enabled

        # Deterministic bucketing via hash
        bucket = self._bucket_user(flag_name, user_id)
        return bucket < flag.rollout_percentage

    def set_environment_override(self, flag_name: str, environment: str, enabled: bool) -> None:
        """Set an environment-specific override for a flag.

        Args:
            flag_name: Name of the flag.
            environment: Environment name (e.g., 'dev', 'staging', 'prod').
            enabled: Whether the flag should be enabled in that environment.

        Raises:
            KeyError: If the flag is not registered.
        """
        if flag_name not in self._flags:
            raise KeyError(f"Unknown flag: {flag_name}")
        if flag_name not in self._env_overrides:
            self._env_overrides[flag_name] = {}
        self._env_overrides[flag_name][environment] = enabled

    def clear_environment_override(self, flag_name: str, environment: str) -> None:
        """Clear an environment-specific override.

        Args:
            flag_name: Name of the flag.
            environment: Environment name.

        Raises:
            KeyError: If the flag is not registered.
        """
        if flag_name not in self._flags:
            raise KeyError(f"Unknown flag: {flag_name}")
        if flag_name in self._env_overrides:
            self._env_overrides[flag_name].pop(environment, None)

    def set_rollout_percentage(self, flag_name: str, percentage: int) -> None:
        """Set the rollout percentage for a flag.

        Args:
            flag_name: Name of the flag.
            percentage: Rollout percentage (0-100).

        Raises:
            KeyError: If the flag is not registered.
            ValueError: If percentage is not 0-100.
        """
        if flag_name not in self._flags:
            raise KeyError(f"Unknown flag: {flag_name}")
        if not 0 <= percentage <= 100:
            raise ValueError(f"rollout_percentage must be 0-100, got {percentage}")
        self._flags[flag_name].rollout_percentage = percentage

    def get_flag_status(
        self,
        flag_name: str,
        user_id: str | None = None,
        environment: str | None = None,
    ) -> dict[str, Any]:
        """Get the full status of a flag.

        Args:
            flag_name: Name of the flag.
            user_id: Optional user identifier.
            environment: Optional environment name.

        Returns:
            Dictionary with flag status information.

        Raises:
            KeyError: If the flag is not registered.
        """
        if flag_name not in self._flags:
            raise KeyError(f"Unknown flag: {flag_name}")
        flag = self._flags[flag_name]
        return {
            "name": flag.name,
            "default_enabled": flag.default_enabled,
            "rollout_percentage": flag.rollout_percentage,
            "environment_overrides": dict(self._env_overrides.get(flag_name, {})),
            "enabled": self.is_enabled(flag_name, user_id=user_id, environment=environment),
            "environment": environment,
        }

    def list_flags(self) -> list[FeatureFlag]:
        """Return all registered flags."""
        return list(self._flags.values())

    @staticmethod
    def _bucket_user(flag_name: str, user_id: str) -> int:
        """Deterministically bucket a user into 0-99.

        Args:
            flag_name: Flag name (used as hash salt).
            user_id: User identifier.

        Returns:
            Bucket number 0-99.
        """
        key = f"{flag_name}:{user_id}"
        hash_bytes = hashlib.sha256(key.encode()).digest()
        return int.from_bytes(hash_bytes[:4], "big") % 100
