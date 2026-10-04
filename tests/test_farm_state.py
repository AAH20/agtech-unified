"""Test shared FarmState model with validation."""

import pytest

from src.integration.farm_state import FarmState


class TestFarmStateCreation:
    """Valid construction."""

    def test_create_valid_state(self):
        """A FarmState with in-range values is accepted."""
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.3,
            nutrient_level=0.6,
            pest_pressure=0.2,
        )

        assert state.soil_moisture == 0.5
        assert state.temperature == 25.0
        assert state.crop_height == 0.3
        assert state.nutrient_level == 0.6
        assert state.pest_pressure == 0.2

    def test_boundary_values_accepted(self):
        """Values at the exact boundaries of each range are valid."""
        state = FarmState(
            soil_moisture=0.0,
            temperature=-50.0,
            crop_height=0.0,
            nutrient_level=0.0,
            pest_pressure=0.0,
        )
        assert state.soil_moisture == 0.0

        state2 = FarmState(
            soil_moisture=1.0,
            temperature=60.0,
            crop_height=10.0,
            nutrient_level=1.0,
            pest_pressure=1.0,
        )
        assert state2.pest_pressure == 1.0


class TestFarmStateValidation:
    """Out-of-range values raise ValueError."""

    def test_soil_moisture_above_one_rejected(self):
        """soil_moisture > 1.0 is invalid."""
        with pytest.raises(ValueError, match="soil_moisture"):
            FarmState(
                soil_moisture=1.5,
                temperature=25.0,
                crop_height=0.3,
                nutrient_level=0.5,
                pest_pressure=0.1,
            )

    def test_soil_moisture_negative_rejected(self):
        """Negative soil_moisture is invalid."""
        with pytest.raises(ValueError, match="soil_moisture"):
            FarmState(
                soil_moisture=-0.1,
                temperature=25.0,
                crop_height=0.3,
                nutrient_level=0.5,
                pest_pressure=0.1,
            )

    def test_temperature_below_range_rejected(self):
        """temperature < -50 is invalid."""
        with pytest.raises(ValueError, match="temperature"):
            FarmState(
                soil_moisture=0.5,
                temperature=-51.0,
                crop_height=0.3,
                nutrient_level=0.5,
                pest_pressure=0.1,
            )

    def test_temperature_above_range_rejected(self):
        """temperature > 60 is invalid."""
        with pytest.raises(ValueError, match="temperature"):
            FarmState(
                soil_moisture=0.5,
                temperature=61.0,
                crop_height=0.3,
                nutrient_level=0.5,
                pest_pressure=0.1,
            )

    def test_crop_height_negative_rejected(self):
        """Negative crop_height is invalid."""
        with pytest.raises(ValueError, match="crop_height"):
            FarmState(
                soil_moisture=0.5,
                temperature=25.0,
                crop_height=-0.1,
                nutrient_level=0.5,
                pest_pressure=0.1,
            )

    def test_crop_height_above_range_rejected(self):
        """crop_height > 10 is invalid."""
        with pytest.raises(ValueError, match="crop_height"):
            FarmState(
                soil_moisture=0.5,
                temperature=25.0,
                crop_height=10.1,
                nutrient_level=0.5,
                pest_pressure=0.1,
            )

    def test_nutrient_level_out_of_range_rejected(self):
        """nutrient_level outside [0, 1] is invalid."""
        with pytest.raises(ValueError, match="nutrient_level"):
            FarmState(
                soil_moisture=0.5,
                temperature=25.0,
                crop_height=0.3,
                nutrient_level=1.5,
                pest_pressure=0.1,
            )

    def test_pest_pressure_out_of_range_rejected(self):
        """pest_pressure outside [0, 1] is invalid."""
        with pytest.raises(ValueError, match="pest_pressure"):
            FarmState(
                soil_moisture=0.5,
                temperature=25.0,
                crop_height=0.3,
                nutrient_level=0.5,
                pest_pressure=-0.5,
            )

    def test_is_valid_helper(self):
        """is_valid() returns False for out-of-range values without raising."""
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.3,
            nutrient_level=0.5,
            pest_pressure=0.1,
        )
        assert state.is_valid() is True

        state.soil_moisture = 2.0
        assert state.is_valid() is False
