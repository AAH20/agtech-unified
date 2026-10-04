"""Dedicated unit tests for onboarding modules (src/onboarding/).

Covers ModuleRegistry tier filtering, module listing, invalid tier handling,
and OrganizationProfiler tier recommendation boundaries.
"""

from __future__ import annotations


class TestModuleRegistry:
    """Tests for ModuleRegistry."""

    def test_get_modules_startup(self, module_registry):
        modules = module_registry.get_modules("startup")
        assert modules == ["basics", "sensors", "alerts"]

    def test_get_modules_smb(self, module_registry):
        modules = module_registry.get_modules("smb")
        assert modules == ["basics", "sensors", "alerts", "analytics", "reporting"]

    def test_get_modules_enterprise(self, module_registry):
        modules = module_registry.get_modules("enterprise")
        assert modules == [
            "basics",
            "sensors",
            "alerts",
            "analytics",
            "reporting",
            "integrations",
            "sla",
        ]

    def test_get_modules_invalid_tier_returns_empty(self, module_registry):
        modules = module_registry.get_modules("nonexistent")
        assert modules == []

    def test_get_modules_empty_string_returns_empty(self, module_registry):
        modules = module_registry.get_modules("")
        assert modules == []

    def test_all_tiers_returns_all(self, module_registry):
        tiers = module_registry.all_tiers()
        assert set(tiers) == {"startup", "smb", "enterprise"}

    def test_all_tiers_returns_list(self, module_registry):
        tiers = module_registry.all_tiers()
        assert isinstance(tiers, list)
        assert len(tiers) == 3

    def test_tier_hierarchy_superset(self, module_registry):
        """Each higher tier should be a superset of lower tiers."""
        startup = set(module_registry.get_modules("startup"))
        smb = set(module_registry.get_modules("smb"))
        enterprise = set(module_registry.get_modules("enterprise"))
        assert startup < smb < enterprise

    def test_common_modules_in_all_tiers(self, module_registry):
        """basics, sensors, alerts should be in all tiers."""
        for tier in module_registry.all_tiers():
            modules = module_registry.get_modules(tier)
            assert "basics" in modules
            assert "sensors" in modules
            assert "alerts" in modules

    def test_enterprise_has_integrations(self, module_registry):
        modules = module_registry.get_modules("enterprise")
        assert "integrations" in modules

    def test_enterprise_has_sla(self, module_registry):
        modules = module_registry.get_modules("enterprise")
        assert "sla" in modules

    def test_startup_lacks_analytics(self, module_registry):
        modules = module_registry.get_modules("startup")
        assert "analytics" not in modules

    def test_smb_lacks_integrations(self, module_registry):
        modules = module_registry.get_modules("smb")
        assert "integrations" not in modules


class TestOrganizationProfiler:
    """Tests for OrganizationProfiler."""

    def test_startup_for_small_org(self, org_profiler):
        tier = org_profiler.recommend_tier(employees=5, revenue=50000.0)
        assert tier == "startup"

    def test_startup_boundary_9_employees(self, org_profiler):
        tier = org_profiler.recommend_tier(employees=9, revenue=100000.0)
        assert tier == "startup"

    def test_smb_for_medium_org(self, org_profiler):
        tier = org_profiler.recommend_tier(employees=100, revenue=1000000.0)
        assert tier == "smb"

    def test_smb_boundary_10_employees(self, org_profiler):
        tier = org_profiler.recommend_tier(employees=10, revenue=100000.0)
        assert tier == "smb"

    def test_smb_boundary_500_employees(self, org_profiler):
        tier = org_profiler.recommend_tier(employees=500, revenue=10000000.0)
        assert tier == "smb"

    def test_enterprise_for_large_org(self, org_profiler):
        tier = org_profiler.recommend_tier(employees=501, revenue=100000000.0)
        assert tier == "enterprise"

    def test_enterprise_boundary_501_employees(self, org_profiler):
        tier = org_profiler.recommend_tier(employees=501, revenue=100000000.0)
        assert tier == "enterprise"

    def test_zero_employees_startup(self, org_profiler):
        tier = org_profiler.recommend_tier(employees=0, revenue=0.0)
        assert tier == "startup"

    def test_very_large_org_enterprise(self, org_profiler):
        tier = org_profiler.recommend_tier(employees=10000, revenue=1000000000.0)
        assert tier == "enterprise"

    def test_revenue_ignored(self, org_profiler):
        """Revenue should not affect tier recommendation."""
        tier_low_rev = org_profiler.recommend_tier(employees=5, revenue=0.0)
        tier_high_rev = org_profiler.recommend_tier(employees=5, revenue=1e12)
        assert tier_low_rev == tier_high_rev

    def test_negative_employees_startup(self, org_profiler):
        """Negative employees should still return a tier."""
        tier = org_profiler.recommend_tier(employees=-1, revenue=0.0)
        assert tier == "startup"

    def test_boundary_10_exact(self, org_profiler):
        """Exactly 10 employees → smb."""
        tier = org_profiler.recommend_tier(employees=10, revenue=0.0)
        assert tier == "smb"

    def test_boundary_500_exact(self, org_profiler):
        """Exactly 500 employees → smb."""
        tier = org_profiler.recommend_tier(employees=500, revenue=0.0)
        assert tier == "smb"

    def test_boundary_501_exact(self, org_profiler):
        """Exactly 501 employees → enterprise."""
        tier = org_profiler.recommend_tier(employees=501, revenue=0.0)
        assert tier == "enterprise"
