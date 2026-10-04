"""Organization profiling and tier recommendation."""


class OrganizationProfiler:
    """Recommends an onboarding tier based on organization size."""

    def recommend_tier(self, employees: int, revenue: float) -> str:
        """Return 'startup', 'smb', or 'enterprise' based on employee count.

        Args:
            employees: Number of employees.
            revenue: Annual revenue in USD.

        Returns:
            Tier name string.
        """
        if employees < 10:
            return "startup"
        elif employees <= 500:
            return "smb"
        else:
            return "enterprise"
