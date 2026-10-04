"""Module registry mapping tiers to onboarding modules."""


class ModuleRegistry:
    """Maps tier names to lists of module names."""

    MODULES = {
        "startup": ["basics", "sensors", "alerts"],
        "smb": ["basics", "sensors", "alerts", "analytics", "reporting"],
        "enterprise": [
            "basics",
            "sensors",
            "alerts",
            "analytics",
            "reporting",
            "integrations",
            "sla",
        ],
    }

    def get_modules(self, tier: str) -> list:
        """Return module list for a tier."""
        return self.MODULES.get(tier, [])

    def all_tiers(self) -> list:
        """Return all tier names."""
        return list(self.MODULES.keys())
