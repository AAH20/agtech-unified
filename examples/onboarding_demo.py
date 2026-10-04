"""Onboarding demo: module registry, profiling, and guided setup."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.onboarding.modules import ModuleRegistry
from src.onboarding.sizing import OrganizationProfiler
from src.onboarding.wizard import SetupWizard


def main():
    registry = ModuleRegistry()
    profiler = OrganizationProfiler()

    # Multi-factor scoring
    result = profiler.score_organization(
        employees=50, revenue=1_000_000.0, fields=10, sensors=20, has_dedicated_it=True
    )
    print(f"Score: {result['total_score']}")
    print(f"Factors: {result['factors']}")

    # Tier recommendation
    tier = profiler.recommend_tier_scored(
        employees=50, revenue=1_000_000.0, fields=10, sensors=20, has_dedicated_it=True
    )
    print(f"Recommended tier: {tier}")

    # Guided setup wizard
    wizard = SetupWizard()
    wizard.set_organization_profile(
        employees=50, revenue=1_000_000.0, fields=10, sensors=20, has_dedicated_it=True
    )
    steps = wizard.get_setup_steps()
    print(f"Setup steps: {steps}")

    config = wizard.get_config_template()
    print(f"Config tier: {config['tier']}")
    print(f"Config modules: {config['modules']}")


if __name__ == "__main__":
    main()
