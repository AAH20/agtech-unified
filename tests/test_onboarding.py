"""Tests for onboarding and sizing module."""

from src.onboarding.modules import ModuleRegistry
from src.onboarding.sizing import OrganizationProfiler


def test_startup_tier():
    profiler = OrganizationProfiler()
    assert profiler.recommend_tier(5, 100000.0) == "startup"


def test_smb_tier():
    profiler = OrganizationProfiler()
    assert profiler.recommend_tier(100, 5000000.0) == "smb"


def test_enterprise_tier():
    profiler = OrganizationProfiler()
    assert profiler.recommend_tier(1000, 100000000.0) == "enterprise"


def test_module_registry_has_all_tiers():
    registry = ModuleRegistry()
    tiers = registry.all_tiers()
    assert "startup" in tiers
    assert "smb" in tiers
    assert "enterprise" in tiers


def test_recommend_tier_boundary():
    profiler = OrganizationProfiler()
    assert profiler.recommend_tier(9, 0.0) == "startup"
    assert profiler.recommend_tier(10, 0.0) == "smb"
    assert profiler.recommend_tier(500, 0.0) == "smb"
    assert profiler.recommend_tier(501, 0.0) == "enterprise"
