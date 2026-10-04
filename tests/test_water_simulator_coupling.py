"""Tests for water balance ↔ simulator coupling."""

from src.digital_twin.shared_state import DigitalTwinState
from src.digital_twin.simulator import DigitalTwin, SimulationState
from src.digital_twin.water_balance import WaterBalanceModel, WaterBalanceState


class TestWaterSimulatorCoupling:
    """Test water balance and simulator integration."""

    def test_water_balance_provides_moisture(self):
        """Water balance model provides moisture to simulator."""
        wb = WaterBalanceModel()
        state = WaterBalanceState(soil_moisture=0.3)
        weather = [{"precipitation": 0.0, "evapotranspiration": 5.0} for _ in range(5)]
        result = wb.simulate(state, weather)
        # Result moisture should be usable by simulator
        assert 0.0 <= result.final_state.soil_moisture <= 1.0

    def test_simulator_uses_water_balance_moisture(self):
        """Simulator can use water balance moisture output."""
        wb = WaterBalanceModel()
        wb_state = WaterBalanceState(soil_moisture=0.3)
        weather = [{"precipitation": 0.0, "evapotranspiration": 3.0} for _ in range(3)]
        wb_result = wb.simulate(wb_state, weather)

        # Use water balance result as simulator input
        sim = DigitalTwin()
        sim_state = SimulationState(
            soil_moisture=wb_result.final_state.soil_moisture,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        sim_result = sim.simulate(sim_state, days=5)
        assert sim_result.final_state.crop_height > 0.1

    def test_coupled_simulation_improves_growth(self):
        """Coupled simulation produces better growth than uncoupled."""
        # Uncoupled: fixed moisture
        sim_uncoupled = DigitalTwin()
        uncoupled_state = SimulationState(
            soil_moisture=0.15,  # Low moisture
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        uncoupled_result = sim_uncoupled.simulate(uncoupled_state, days=10)

        # Coupled: water balance maintains moisture
        wb = WaterBalanceModel()
        wb_state = WaterBalanceState(soil_moisture=0.3)
        weather = [{"precipitation": 2.0, "evapotranspiration": 3.0} for _ in range(10)]
        wb_result = wb.simulate(wb_state, weather)

        sim_coupled = DigitalTwin()
        coupled_state = SimulationState(
            soil_moisture=wb_result.final_state.soil_moisture,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        coupled_result = sim_coupled.simulate(coupled_state, days=10)

        # Coupled should have equal or better growth
        assert coupled_result.final_state.crop_height >= uncoupled_result.final_state.crop_height

    def test_water_stress_reduces_growth_coupled(self):
        """Water stress in water balance reduces simulator growth."""
        # Well-watered
        wb_good = WaterBalanceModel()
        wb_state_good = WaterBalanceState(soil_moisture=0.35)
        weather_good = [{"precipitation": 5.0, "evapotranspiration": 2.0} for _ in range(5)]
        wb_result_good = wb_good.simulate(wb_state_good, weather_good)

        # Drought
        wb_bad = WaterBalanceModel()
        wb_state_bad = WaterBalanceState(soil_moisture=0.12)
        weather_bad = [{"precipitation": 0.0, "evapotranspiration": 5.0} for _ in range(5)]
        wb_result_bad = wb_bad.simulate(wb_state_bad, weather_bad)

        sim = DigitalTwin()
        good_state = SimulationState(
            soil_moisture=wb_result_good.final_state.soil_moisture,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        bad_state = SimulationState(
            soil_moisture=wb_result_bad.final_state.soil_moisture,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        good_result = sim.simulate(good_state, days=10)
        bad_result = sim.simulate(bad_state, days=10)
        assert good_result.final_state.crop_height > bad_result.final_state.crop_height

    def test_shared_state_water_coupling(self):
        """Shared state model enables water coupling."""
        state = DigitalTwinState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.1,
        )
        # Convert to water balance state
        wb_state = state.to_water_balance_state()
        assert wb_state.soil_moisture == 0.3

        # Run water balance
        wb = WaterBalanceModel()
        weather = [{"precipitation": 0.0, "evapotranspiration": 3.0} for _ in range(3)]
        wb_result = wb.simulate(wb_state, weather)

        # Update shared state
        state.soil_moisture = wb_result.final_state.soil_moisture
        assert state.soil_moisture != 0.3  # Should have changed

    def test_irrigation_event_coupling(self):
        """Irrigation event in water balance affects simulator."""
        wb = WaterBalanceModel()
        wb_state = WaterBalanceState(soil_moisture=0.15)
        weather = [{"precipitation": 0.0, "evapotranspiration": 4.0} for _ in range(5)]
        irrigation = [20.0, 0.0, 0.0, 0.0, 0.0]  # Irrigate on day 0
        wb_result = wb.simulate(wb_state, weather, irrigation_schedule=irrigation)

        # After irrigation on day 0, moisture should be higher
        assert wb_result.history[0].soil_moisture > 0.15

        # Simulator should benefit
        sim = DigitalTwin()
        sim_state = SimulationState(
            soil_moisture=wb_result.final_state.soil_moisture,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        result = sim.simulate(sim_state, days=5)
        assert result.final_state.crop_height > 0.1
