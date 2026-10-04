"""Onboarding demo: module registry and organization profiling."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.onboarding.modules import ModuleRegistry
from src.onboarding.sizing import OrganizationProfiler


def main():
    registry = ModuleRegistry()
    modules = registry.get_modules(tier="starter")
    print(f"Starter modules: {modules}")

    profiler = OrganizationProfiler()
    tier = profiler.recommend_tier(employees=50, fields=10)
    print(f"Recommended tier: {tier}")


if __name__ == "__main__":
    main()
