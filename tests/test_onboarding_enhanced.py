"""Tests for enhanced onboarding: multi-factor scoring, dependency graph, wizard, and templates.

Covers:
- OrganizationProfiler multi-factor scoring
- ModuleDependencyGraph topological ordering and cycle detection
- SetupWizard guided flow
- ConfigTemplate validation and tier-specific configs
"""

from __future__ import annotations

import pytest

from src.onboarding.dependencies import ModuleDependencyGraph
from src.onboarding.templates import ConfigTemplate
from src.onboarding.wizard import SetupWizard

# ---------------------------------------------------------------------------
# Multi-factor scoring
# ---------------------------------------------------------------------------


class TestMultiFactorScoring:
    """Tests for OrganizationProfiler multi-factor scoring."""

    def test_score_organization_returns_dict(self, org_profiler):
        result = org_profiler.score_organization(employees=50, revenue=1_000_000.0)
        assert isinstance(result, dict)
        assert "total_score" in result
        assert "factors" in result

    def test_score_organization_factors_present(self, org_profiler):
        result = org_profiler.score_organization(employees=50, revenue=1_000_000.0)
        factors = result["factors"]
        assert "employees" in factors
        assert "revenue" in factors
        assert "fields" in factors
        assert "sensors" in factors
        assert "dedicated_it" in factors

    def test_score_organization_total_is_sum(self, org_profiler):
        result = org_profiler.score_organization(
            employees=50, revenue=1_000_000.0, fields=5, sensors=10, has_dedicated_it=True
        )
        factors = result["factors"]
        expected = (
            factors["employees"]
            + factors["revenue"]
            + factors["fields"]
            + factors["sensors"]
            + factors["dedicated_it"]
        )
        assert result["total_score"] == expected

    def test_larger_org_scores_higher(self, org_profiler):
        small = org_profiler.score_organization(employees=5, revenue=50_000.0)
        large = org_profiler.score_organization(employees=500, revenue=10_000_000.0)
        assert large["total_score"] > small["total_score"]

    def test_more_fields_increases_score(self, org_profiler):
        few = org_profiler.score_organization(employees=50, revenue=1_000_000.0, fields=1)
        many = org_profiler.score_organization(employees=50, revenue=1_000_000.0, fields=50)
        assert many["total_score"] > few["total_score"]

    def test_more_sensors_increases_score(self, org_profiler):
        few = org_profiler.score_organization(employees=50, revenue=1_000_000.0, sensors=1)
        many = org_profiler.score_organization(employees=50, revenue=1_000_000.0, sensors=100)
        assert many["total_score"] > few["total_score"]

    def test_dedicated_it_increases_score(self, org_profiler):
        without_it = org_profiler.score_organization(
            employees=50, revenue=1_000_000.0, has_dedicated_it=False
        )
        with_it = org_profiler.score_organization(
            employees=50, revenue=1_000_000.0, has_dedicated_it=True
        )
        assert with_it["total_score"] > without_it["total_score"]

    def test_recommend_tier_scored_startup(self, org_profiler):
        tier = org_profiler.recommend_tier_scored(employees=5, revenue=50_000.0)
        assert tier == "startup"

    def test_recommend_tier_scored_smb(self, org_profiler):
        tier = org_profiler.recommend_tier_scored(employees=100, revenue=1_000_000.0)
        assert tier == "smb"

    def test_recommend_tier_scored_enterprise(self, org_profiler):
        tier = org_profiler.recommend_tier_scored(employees=1000, revenue=100_000_000.0)
        assert tier == "enterprise"

    def test_recommend_tier_scored_with_factors(self, org_profiler):
        """A small org with many fields/sensors could be pushed to smb."""
        tier = org_profiler.recommend_tier_scored(
            employees=8, revenue=200_000.0, fields=30, sensors=50, has_dedicated_it=True
        )
        assert tier in ("startup", "smb", "enterprise")

    def test_recommend_tier_scored_returns_valid_tier(self, org_profiler):
        """Always returns one of the three valid tiers."""
        for emp in [0, 5, 50, 100, 500, 1000, 5000]:
            for rev in [0.0, 100_000.0, 1_000_000.0, 100_000_000.0]:
                tier = org_profiler.recommend_tier_scored(employees=emp, revenue=rev)
                assert tier in ("startup", "smb", "enterprise")

    def test_score_organization_zero_values(self, org_profiler):
        result = org_profiler.score_organization(
            employees=0, revenue=0.0, fields=0, sensors=0, has_dedicated_it=False
        )
        assert result["total_score"] >= 0

    def test_score_organization_monotonic_employees(self, org_profiler):
        """More employees should never decrease the score."""
        prev = -1
        for emp in [1, 5, 10, 50, 100, 500, 1000]:
            score = org_profiler.score_organization(employees=emp, revenue=0.0)["total_score"]
            assert score >= prev
            prev = score


# ---------------------------------------------------------------------------
# Module dependency graph
# ---------------------------------------------------------------------------


class TestModuleDependencyGraph:
    """Tests for ModuleDependencyGraph."""

    def test_add_dependency(self):
        graph = ModuleDependencyGraph()
        graph.add_dependency("analytics", "sensors")
        assert "sensors" in graph.get_dependencies("analytics")

    def test_get_dependencies_empty(self):
        graph = ModuleDependencyGraph()
        assert graph.get_dependencies("nonexistent") == []

    def test_get_all_dependencies_transitive(self):
        graph = ModuleDependencyGraph()
        graph.add_dependency("analytics", "sensors")
        graph.add_dependency("sensors", "basics")
        all_deps = graph.get_all_dependencies("analytics")
        assert "sensors" in all_deps
        assert "basics" in all_deps

    def test_get_all_dependencies_no_deps(self):
        graph = ModuleDependencyGraph()
        assert graph.get_all_dependencies("basics") == []

    def test_topological_order(self):
        graph = ModuleDependencyGraph()
        graph.add_dependency("analytics", "sensors")
        graph.add_dependency("sensors", "basics")
        graph.add_dependency("reporting", "analytics")
        order = graph.get_install_order(["basics", "sensors", "analytics", "reporting"])
        assert order.index("basics") < order.index("sensors")
        assert order.index("sensors") < order.index("analytics")
        assert order.index("analytics") < order.index("reporting")

    def test_topological_order_empty(self):
        graph = ModuleDependencyGraph()
        assert graph.get_install_order([]) == []

    def test_topological_order_single(self):
        graph = ModuleDependencyGraph()
        assert graph.get_install_order(["basics"]) == ["basics"]

    def test_cycle_detection_raises(self):
        graph = ModuleDependencyGraph()
        graph.add_dependency("a", "b")
        graph.add_dependency("b", "a")
        with pytest.raises(ValueError, match="cycle"):
            graph.get_install_order(["a", "b"])

    def test_validate_no_cycles(self):
        graph = ModuleDependencyGraph()
        graph.add_dependency("analytics", "sensors")
        graph.add_dependency("sensors", "basics")
        # Should not raise
        graph.validate()

    def test_validate_with_cycles_raises(self):
        graph = ModuleDependencyGraph()
        graph.add_dependency("a", "b")
        graph.add_dependency("b", "c")
        graph.add_dependency("c", "a")
        with pytest.raises(ValueError, match="cycle"):
            graph.validate()

    def test_default_graph_from_registry(self):
        """ModuleDependencyGraph can be built from ModuleRegistry tiers."""
        from src.onboarding.modules import ModuleRegistry

        registry = ModuleRegistry()
        graph = ModuleDependencyGraph.from_registry(registry)
        # All modules from all tiers should be present
        all_modules = set()
        for tier in registry.all_tiers():
            all_modules.update(registry.get_modules(tier))
        for mod in all_modules:
            # Should not raise
            graph.get_dependencies(mod)

    def test_from_registry_enterprise_depends_on_smb(self):
        """Enterprise modules should depend on SMB modules."""
        from src.onboarding.modules import ModuleRegistry

        registry = ModuleRegistry()
        graph = ModuleDependencyGraph.from_registry(registry)
        # Enterprise-only modules should have dependencies
        enterprise_modules = set(registry.get_modules("enterprise"))
        smb_modules = set(registry.get_modules("smb"))
        enterprise_only = enterprise_modules - smb_modules
        for mod in enterprise_only:
            deps = graph.get_all_dependencies(mod)
            assert len(deps) > 0, f"Enterprise module {mod} should have dependencies"


# ---------------------------------------------------------------------------
# Guided setup wizard
# ---------------------------------------------------------------------------


class TestSetupWizard:
    """Tests for SetupWizard."""

    def test_wizard_initial_state(self):
        wizard = SetupWizard()
        assert wizard.is_complete() is False

    def test_wizard_set_profile(self):
        wizard = SetupWizard()
        wizard.set_organization_profile(
            employees=100, revenue=1_000_000.0, fields=10, sensors=20, has_dedicated_it=True
        )
        assert wizard.is_complete() is True

    def test_wizard_get_recommended_tier(self):
        wizard = SetupWizard()
        wizard.set_organization_profile(employees=500, revenue=10_000_000.0)
        tier = wizard.get_recommended_tier()
        assert tier in ("startup", "smb", "enterprise")

    def test_wizard_get_setup_steps(self):
        wizard = SetupWizard()
        wizard.set_organization_profile(employees=100, revenue=1_000_000.0)
        steps = wizard.get_setup_steps()
        assert isinstance(steps, list)
        assert len(steps) > 0

    def test_wizard_setup_steps_ordered(self):
        """Setup steps should be in dependency order."""
        wizard = SetupWizard()
        wizard.set_organization_profile(employees=100, revenue=1_000_000.0)
        steps = wizard.get_setup_steps()
        # basics should come before sensors, sensors before alerts
        if "basics" in steps and "sensors" in steps:
            assert steps.index("basics") < steps.index("sensors")
        if "sensors" in steps and "alerts" in steps:
            assert steps.index("sensors") < steps.index("alerts")

    def test_wizard_get_config_template(self):
        wizard = SetupWizard()
        wizard.set_organization_profile(employees=100, revenue=1_000_000.0)
        config = wizard.get_config_template()
        assert isinstance(config, dict)
        assert "tier" in config

    def test_wizard_validate_setup_valid(self):
        wizard = SetupWizard()
        wizard.set_organization_profile(employees=100, revenue=1_000_000.0)
        config = wizard.get_config_template()
        errors = wizard.validate_setup(config)
        assert errors == []

    def test_wizard_validate_setup_invalid(self):
        wizard = SetupWizard()
        wizard.set_organization_profile(employees=100, revenue=1_000_000.0)
        errors = wizard.validate_setup({})
        assert len(errors) > 0

    def test_wizard_progress_tracking(self):
        wizard = SetupWizard()
        assert wizard.get_progress() == 0.0
        wizard.set_organization_profile(employees=100, revenue=1_000_000.0)
        assert wizard.get_progress() > 0.0

    def test_wizard_reset(self):
        wizard = SetupWizard()
        wizard.set_organization_profile(employees=100, revenue=1_000_000.0)
        assert wizard.is_complete() is True
        wizard.reset()
        assert wizard.is_complete() is False

    def test_wizard_tier_specific_steps(self):
        """Different tiers should produce different step lists."""
        wizard_startup = SetupWizard()
        wizard_startup.set_organization_profile(employees=5, revenue=50_000.0)
        startup_steps = wizard_startup.get_setup_steps()

        wizard_enterprise = SetupWizard()
        wizard_enterprise.set_organization_profile(employees=1000, revenue=100_000_000.0)
        enterprise_steps = wizard_enterprise.get_setup_steps()

        assert set(startup_steps) <= set(enterprise_steps)


# ---------------------------------------------------------------------------
# Configuration templates
# ---------------------------------------------------------------------------


class TestConfigTemplate:
    """Tests for ConfigTemplate."""

    def test_get_template_startup(self):
        template = ConfigTemplate()
        config = template.get_template("startup")
        assert isinstance(config, dict)
        assert config["tier"] == "startup"

    def test_get_template_smb(self):
        template = ConfigTemplate()
        config = template.get_template("smb")
        assert isinstance(config, dict)
        assert config["tier"] == "smb"

    def test_get_template_enterprise(self):
        template = ConfigTemplate()
        config = template.get_template("enterprise")
        assert isinstance(config, dict)
        assert config["tier"] == "enterprise"

    def test_get_template_invalid_tier(self):
        template = ConfigTemplate()
        with pytest.raises(ValueError, match="Unknown tier"):
            template.get_template("nonexistent")

    def test_get_all_templates(self):
        template = ConfigTemplate()
        templates = template.get_all_templates()
        assert "startup" in templates
        assert "smb" in templates
        assert "enterprise" in templates

    def test_templates_have_required_keys(self):
        template = ConfigTemplate()
        for tier in ("startup", "smb", "enterprise"):
            config = template.get_template(tier)
            assert "tier" in config
            assert "modules" in config
            assert "features" in config

    def test_enterprise_template_has_more_features(self):
        template = ConfigTemplate()
        startup = template.get_template("startup")
        enterprise = template.get_template("enterprise")
        assert len(enterprise["features"]) > len(startup["features"])

    def test_validate_template_valid(self):
        template = ConfigTemplate()
        config = template.get_template("startup")
        errors = template.validate_template(config)
        assert errors == []

    def test_validate_template_missing_key(self):
        template = ConfigTemplate()
        errors = template.validate_template({"tier": "startup"})
        assert len(errors) > 0

    def test_validate_template_wrong_type(self):
        template = ConfigTemplate()
        config = template.get_template("startup")
        config["modules"] = "not_a_list"
        errors = template.validate_template(config)
        assert len(errors) > 0

    def test_template_modules_match_registry(self):
        """Template modules should match ModuleRegistry modules for that tier."""
        from src.onboarding.modules import ModuleRegistry

        registry = ModuleRegistry()
        template = ConfigTemplate()
        for tier in ("startup", "smb", "enterprise"):
            config = template.get_template(tier)
            assert set(config["modules"]) == set(registry.get_modules(tier))

    def test_template_has_database_config(self):
        template = ConfigTemplate()
        for tier in ("startup", "smb", "enterprise"):
            config = template.get_template(tier)
            assert "database" in config

    def test_template_has_api_config(self):
        template = ConfigTemplate()
        for tier in ("startup", "smb", "enterprise"):
            config = template.get_template(tier)
            assert "api" in config
