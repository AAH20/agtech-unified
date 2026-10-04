"""Path planning demo: coverage planning for agricultural drones."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.path_planning.coverage import CoverageInstance, CoveragePlanner


def main():
    instance = CoverageInstance(
        width=100.0,
        height=100.0,
        swath_width=5.0,
        start=(0.0, 0.0),
    )
    planner = CoveragePlanner("boustrophedon")
    result = planner.plan(instance)
    print(f"Path Length: {result.path_length:.1f}m")
    print(f"Coverage: {result.coverage_ratio:.0%}")
    print(f"Waypoints: {len(result.waypoints)}")


if __name__ == "__main__":
    main()
