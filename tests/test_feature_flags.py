"""Tests for onboarding feature flags (src/onboarding/feature_flags.py).

Covers gradual rollout, user-bucketing, per-environment overrides,
and flag registration.
"""

from __future__ import annotations

import pytest

from src.onboarding.feature_flags import FeatureFlag, FeatureFlagManager


class TestFeatureFlag:
    """Tests for FeatureFlag dataclass."""

    def test_flag_creation(self):
        flag = FeatureFlag(name="test_flag", default_enabled=False)
        assert flag.name == "test_flag"
        assert flag.default_enabled is False
        assert flag.rollout_percentage == 0

    def test_flag_with_rollout(self):
        flag = FeatureFlag(name="test_flag", default_enabled=True, rollout_percentage=50)
        assert flag.rollout_percentage == 50

    def test_flag_with_env_overrides(self):
        flag = FeatureFlag(
            name="test_flag",
            default_enabled=False,
            environment_overrides={"dev": True, "staging": True},
        )
        assert flag.environment_overrides["dev"] is True
        assert flag.environment_overrides["staging"] is True

    def test_flag_rollout_bounds(self):
        """Rollout percentage must be 0-100."""
        with pytest.raises(ValueError):
            FeatureFlag(name="bad", rollout_percentage=-1)
        with pytest.raises(ValueError):
            FeatureFlag(name="bad", rollout_percentage=101)

    def test_flag_default_rollout_is_zero(self):
        flag = FeatureFlag(name="test", default_enabled=True)
        assert flag.rollout_percentage == 0


class TestFeatureFlagManager:
    """Tests for FeatureFlagManager."""

    @pytest.fixture
    def manager(self):
        return FeatureFlagManager()

    def test_register_flag(self, manager):
        flag = FeatureFlag(name="new_feature", default_enabled=False)
        manager.register_flag(flag)
        assert manager.is_enabled("new_feature") is False

    def test_register_duplicate_flag_raises(self, manager):
        flag = FeatureFlag(name="dup", default_enabled=False)
        manager.register_flag(flag)
        with pytest.raises(ValueError, match="already registered"):
            manager.register_flag(flag)

    def test_is_enabled_default_false(self, manager):
        manager.register_flag(FeatureFlag(name="f1", default_enabled=False))
        assert manager.is_enabled("f1") is False

    def test_is_enabled_default_true(self, manager):
        manager.register_flag(FeatureFlag(name="f2", default_enabled=True, rollout_percentage=100))
        assert manager.is_enabled("f2") is True

    def test_unknown_flag_raises(self, manager):
        with pytest.raises(KeyError):
            manager.is_enabled("nonexistent")

    def test_rollout_zero_percent_disables_for_all(self, manager):
        manager.register_flag(FeatureFlag(name="f", default_enabled=True, rollout_percentage=0))
        for uid in ["user1", "user2", "user3", "abc", "xyz"]:
            assert manager.is_enabled("f", user_id=uid) is False

    def test_rollout_100_percent_enables_for_all(self, manager):
        manager.register_flag(FeatureFlag(name="f", default_enabled=False, rollout_percentage=100))
        for uid in ["user1", "user2", "user3", "abc", "xyz"]:
            assert manager.is_enabled("f", user_id=uid) is True

    def test_rollout_partial_deterministic(self, manager):
        """Same user always gets same result."""
        manager.register_flag(FeatureFlag(name="f", default_enabled=False, rollout_percentage=50))
        result1 = manager.is_enabled("f", user_id="stable_user")
        result2 = manager.is_enabled("f", user_id="stable_user")
        assert result1 == result2

    def test_rollout_partial_some_enabled(self, manager):
        """With 50% rollout, some users should be enabled."""
        manager.register_flag(FeatureFlag(name="f", default_enabled=False, rollout_percentage=50))
        users = [f"user_{i}" for i in range(100)]
        enabled_count = sum(1 for u in users if manager.is_enabled("f", user_id=u))
        assert 0 < enabled_count < 100

    def test_rollout_monotonic_with_percentage(self, manager):
        """Higher rollout percentage should enable at least as many users."""
        manager.register_flag(FeatureFlag(name="f", default_enabled=False, rollout_percentage=10))
        users = [f"user_{i}" for i in range(200)]
        count_10 = sum(1 for u in users if manager.is_enabled("f", user_id=u))

        manager.register_flag(FeatureFlag(name="g", default_enabled=False, rollout_percentage=90))
        count_90 = sum(1 for u in users if manager.is_enabled("g", user_id=u))

        assert count_90 > count_10

    def test_environment_override_enables(self, manager):
        manager.register_flag(FeatureFlag(name="f", default_enabled=False, rollout_percentage=0))
        manager.set_environment_override("f", "dev", True)
        assert manager.is_enabled("f", user_id="anyone", environment="dev") is True

    def test_environment_override_disables(self, manager):
        manager.register_flag(FeatureFlag(name="f", default_enabled=True, rollout_percentage=100))
        manager.set_environment_override("f", "prod", False)
        assert manager.is_enabled("f", user_id="anyone", environment="prod") is False

    def test_environment_override_does_not_affect_other_envs(self, manager):
        manager.register_flag(FeatureFlag(name="f", default_enabled=False, rollout_percentage=0))
        manager.set_environment_override("f", "dev", True)
        assert manager.is_enabled("f", user_id="anyone", environment="prod") is False

    def test_no_user_id_uses_default(self, manager):
        """Without user_id, falls back to default_enabled."""
        manager.register_flag(FeatureFlag(name="f", default_enabled=True, rollout_percentage=50))
        assert manager.is_enabled("f") is True

    def test_no_user_id_respects_zero_rollout(self, manager):
        manager.register_flag(FeatureFlag(name="f", default_enabled=True, rollout_percentage=0))
        assert manager.is_enabled("f") is False

    def test_set_rollout_percentage(self, manager):
        manager.register_flag(FeatureFlag(name="f", default_enabled=False, rollout_percentage=0))
        manager.set_rollout_percentage("f", 100)
        assert manager.is_enabled("f", user_id="anyone") is True

    def test_set_rollout_percentage_bounds(self, manager):
        manager.register_flag(FeatureFlag(name="f", default_enabled=False))
        with pytest.raises(ValueError):
            manager.set_rollout_percentage("f", -1)
        with pytest.raises(ValueError):
            manager.set_rollout_percentage("f", 101)

    def test_get_flag_status(self, manager):
        manager.register_flag(FeatureFlag(name="f", default_enabled=False, rollout_percentage=50))
        status = manager.get_flag_status("f", user_id="user1", environment="dev")
        assert status["name"] == "f"
        assert status["default_enabled"] is False
        assert status["rollout_percentage"] == 50
        assert "enabled" in status
        assert "environment" in status

    def test_list_flags(self, manager):
        manager.register_flag(FeatureFlag(name="a", default_enabled=True))
        manager.register_flag(FeatureFlag(name="b", default_enabled=False))
        flags = manager.list_flags()
        assert len(flags) == 2
        names = {f.name for f in flags}
        assert names == {"a", "b"}

    def test_clear_environment_override(self, manager):
        manager.register_flag(FeatureFlag(name="f", default_enabled=False, rollout_percentage=0))
        manager.set_environment_override("f", "dev", True)
        assert manager.is_enabled("f", user_id="u", environment="dev") is True
        manager.clear_environment_override("f", "dev")
        assert manager.is_enabled("f", user_id="u", environment="dev") is False

    def test_bucketing_consistent_across_managers(self, manager):
        """Two managers with same flag config should bucket identically."""
        manager.register_flag(FeatureFlag(name="f", default_enabled=False, rollout_percentage=50))
        manager2 = FeatureFlagManager()
        manager2.register_flag(FeatureFlag(name="f", default_enabled=False, rollout_percentage=50))
        for uid in ["alice", "bob", "charlie"]:
            assert manager.is_enabled("f", user_id=uid) == manager2.is_enabled("f", user_id=uid)
