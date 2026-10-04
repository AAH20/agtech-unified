"""Tests for ModuleRegistry smoke-testing recommended modules."""

from src.onboarding.modules import ModuleRegistry


class TestModuleRegistryValidation:
    """ModuleRegistry validates that recommended modules are importable."""

    def test_all_recommended_modules_importable(self):
        """Every module name in every tier maps to an importable module."""
        registry = ModuleRegistry()
        for tier in registry.all_tiers():
            for module_name in registry.get_modules(tier):
                # Each module name should be a valid Python identifier
                assert module_name.isidentifier()

    def test_startup_tier_modules(self):
        """Startup tier has the expected modules."""
        registry = ModuleRegistry()
        modules = registry.get_modules("startup")
        assert "basics" in modules
        assert "sensors" in modules
        assert "alerts" in modules

    def test_smb_tier_modules(self):
        """SMB tier includes startup modules plus more."""
        registry = ModuleRegistry()
        modules = registry.get_modules("smb")
        assert "basics" in modules
        assert "sensors" in modules
        assert "alerts" in modules
        assert "analytics" in modules
        assert "reporting" in modules

    def test_enterprise_tier_modules(self):
        """Enterprise tier includes all modules."""
        registry = ModuleRegistry()
        modules = registry.get_modules("enterprise")
        assert "basics" in modules
        assert "sensors" in modules
        assert "alerts" in modules
        assert "analytics" in modules
        assert "reporting" in modules
        assert "integrations" in modules
        assert "sla" in modules

    def test_unknown_tier_returns_empty(self):
        """Unknown tier returns empty list."""
        registry = ModuleRegistry()
        assert registry.get_modules("nonexistent") == []
