"""Decision support example: agricultural recommendations."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.decision_support.recommender import FarmState, DecisionEngine


def main():
    state = FarmState(
        soil_moisture=0.15,
        temperature=40.0,
        crop_height=0.05,
        nutrient_level=0.1,
        pest_pressure=0.8,
    )
    engine = DecisionEngine("rule_based")
    result = engine.recommend(state)
    print(f"Priority Score: {result.priority_score:.1f}")
    for rec in result.recommendations:
        print(f"  [{rec.priority}] {rec.action}: {rec.reason}")


if __name__ == "__main__":
    main()
