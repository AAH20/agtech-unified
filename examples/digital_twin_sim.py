"""Digital twin simulation example: crop growth modeling."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.digital_twin.simulator import SimulationState, DigitalTwin


def main():
    state = SimulationState(
        soil_moisture=0.6,
        temperature=25.0,
        crop_height=0.1,
        nutrient_level=0.8,
    )
    twin = DigitalTwin("logistic_growth")
    result = twin.simulate(state, days=30)
    print(f"Days Simulated: {result.days_simulated}")
    print(f"Final Height: {result.final_state.crop_height:.2f}m")
    print(f"Final Biomass: {result.final_state.biomass:.2f}")


if __name__ == "__main__":
    main()
