"""Organization profiling and tier recommendation with multi-factor scoring."""

from __future__ import annotations

from typing import Any


class OrganizationProfiler:
    """Recommends an onboarding tier based on organization attributes.

    Supports both the original single-factor (employee count) method
    and a new multi-factor scoring model that considers employees,
    revenue, number of fields, number of sensors, and whether the
    organization has dedicated IT staff.
    """

    # Tier thresholds for the legacy single-factor method
    _STARTUP_MAX_EMPLOYEES = 9
    _SMB_MAX_EMPLOYEES = 500

    # Multi-factor scoring weights
    _WEIGHT_EMPLOYEES = 0.30
    _WEIGHT_REVENUE = 0.20
    _WEIGHT_FIELDS = 0.20
    _WEIGHT_SENSORS = 0.20
    _WEIGHT_DEDICATED_IT = 0.10

    # Score normalization bounds
    _MAX_EMPLOYEES_SCORE = 500
    _MAX_REVENUE_SCORE = 10_000_000
    _MAX_FIELDS_SCORE = 100
    _MAX_SENSORS_SCORE = 500
    _DEDICATED_IT_SCORE = 100

    # Scored tier thresholds (total weighted score out of 100)
    _SCORED_SMB_THRESHOLD = 5
    _SCORED_ENTERPRISE_THRESHOLD = 40

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

    def score_organization(
        self,
        employees: int,
        revenue: float,
        fields: int = 0,
        sensors: int = 0,
        has_dedicated_it: bool = False,
    ) -> dict[str, Any]:
        """Compute a multi-factor score for the organization.

        Each factor is normalized to 0–100, then combined using
        fixed weights. The total score ranges from 0 to 100.

        Args:
            employees: Number of employees.
            revenue: Annual revenue in USD.
            fields: Number of fields/managed plots.
            sensors: Number of deployed or planned sensors.
            has_dedicated_it: Whether the org has dedicated IT staff.

        Returns:
            Dictionary with 'total_score' and per-factor breakdown.
        """
        emp_score = self._normalize(employees, self._MAX_EMPLOYEES_SCORE)
        rev_score = self._normalize(revenue, self._MAX_REVENUE_SCORE)
        field_score = self._normalize(fields, self._MAX_FIELDS_SCORE)
        sensor_score = self._normalize(sensors, self._MAX_SENSORS_SCORE)
        it_score = self._DEDICATED_IT_SCORE if has_dedicated_it else 0

        factors = {
            "employees": round(emp_score * self._WEIGHT_EMPLOYEES, 2),
            "revenue": round(rev_score * self._WEIGHT_REVENUE, 2),
            "fields": round(field_score * self._WEIGHT_FIELDS, 2),
            "sensors": round(sensor_score * self._WEIGHT_SENSORS, 2),
            "dedicated_it": round(it_score * self._WEIGHT_DEDICATED_IT, 2),
        }

        total = sum(factors.values())
        return {"total_score": round(total, 2), "factors": factors}

    def recommend_tier_scored(
        self,
        employees: int,
        revenue: float,
        fields: int = 0,
        sensors: int = 0,
        has_dedicated_it: bool = False,
    ) -> str:
        """Recommend a tier using multi-factor scoring.

        Uses the same weighted score as score_organization but maps
        the total to a tier using scored thresholds.

        Args:
            employees: Number of employees.
            revenue: Annual revenue in USD.
            fields: Number of fields/managed plots.
            sensors: Number of deployed or planned sensors.
            has_dedicated_it: Whether the org has dedicated IT staff.

        Returns:
            Tier name: 'startup', 'smb', or 'enterprise'.
        """
        result = self.score_organization(
            employees=employees,
            revenue=revenue,
            fields=fields,
            sensors=sensors,
            has_dedicated_it=has_dedicated_it,
        )
        total = result["total_score"]

        if total >= self._SCORED_ENTERPRISE_THRESHOLD:
            return "enterprise"
        elif total >= self._SCORED_SMB_THRESHOLD:
            return "smb"
        else:
            return "startup"

    def _normalize(self, value: float, maximum: float) -> float:
        """Clamp a value to [0, maximum] and normalize to [0, 100].

        Args:
            value: Raw value.
            maximum: Value that maps to 100.

        Returns:
            Normalized score in [0, 100].
        """
        if maximum <= 0:
            return 0.0
        if value <= 0:
            return 0.0
        if value >= maximum:
            return 100.0
        return (value / maximum) * 100.0
