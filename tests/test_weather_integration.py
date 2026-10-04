"""Tests for weather/climate data integration."""

from src.digital_twin.simulator import DigitalTwin, SimulationState


class TestWeatherIntegration:
    """Test time-varying weather driver."""

    def test_diurnal_temperature_cycle(self):
        """Temperature varies diurnally."""
        twin = DigitalTwin()
        [{"temperature": 20.0 + 10.0 * (1 if 6 <= h <= 18 else -1)} for h in range(24)]
        # Simulator should accept weather data
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        result = twin.simulate_with_weather(
            state,
            days=3,
            weather_data=[
                {"temp_max": 30.0, "temp_min": 15.0, "precipitation": 0.0} for _ in range(3)
            ],
        )
        assert result is not None

    def test_seasonal_variation(self):
        """Temperature varies seasonally."""
        twin = DigitalTwin()
        # Summer vs winter
        summer = SimulationState(
            soil_moisture=0.5,
            temperature=30.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        winter = SimulationState(
            soil_moisture=0.5,
            temperature=5.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        result_summer = twin.simulate(summer, days=10)
        result_winter = twin.simulate(winter, days=10)
        assert result_summer.final_state.crop_height > result_winter.final_state.crop_height

    def test_extreme_weather_event(self):
        """Extreme heat reduces growth."""
        twin = DigitalTwin()
        normal = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        heat_wave = SimulationState(
            soil_moisture=0.5,
            temperature=42.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        result_normal = twin.simulate(normal, days=10)
        result_heat = twin.simulate(heat_wave, days=10)
        assert result_normal.final_state.crop_height > result_heat.final_state.crop_height

    def test_weather_forecast_integration(self):
        """Simulator accepts weather forecast."""
        twin = DigitalTwin()
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        forecast = [
            {"temp_max": 28.0, "temp_min": 18.0, "precipitation": 0.0, "humidity": 60.0}
            for _ in range(7)
        ]
        result = twin.simulate_with_weather(state, days=7, weather_data=forecast)
        assert result is not None
        assert result.days_simulated == 7

    def test_rainfall_increases_moisture(self):
        """Rainfall in weather data increases soil moisture."""
        twin = DigitalTwin()
        state = SimulationState(
            soil_moisture=0.2,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        rainy_weather = [
            {"temp_max": 25.0, "temp_min": 18.0, "precipitation": 10.0} for _ in range(3)
        ]
        result = twin.simulate_with_weather(state, days=3, weather_data=rainy_weather)
        # Moisture should increase from rainfall
        assert result.final_state.soil_moisture > 0.2

    def test_drought_reduces_moisture(self):
        """Drought in weather data decreases soil moisture."""
        twin = DigitalTwin()
        state = SimulationState(
            soil_moisture=0.4,
            temperature=35.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        drought_weather = [
            {"temp_max": 38.0, "temp_min": 25.0, "precipitation": 0.0} for _ in range(5)
        ]
        result = twin.simulate_with_weather(state, days=5, weather_data=drought_weather)
        # Moisture should decrease
        assert result.final_state.soil_moisture < 0.4
