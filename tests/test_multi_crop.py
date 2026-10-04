"""Tests for multi-crop and crop rotation support."""

from src.digital_twin.simulator import CropType, DigitalTwin, SimulationState


class TestMultiCropSupport:
    """Test crop-specific parameterization."""

    def test_wheat_parameters(self):
        """Wheat has specific growth parameters."""
        twin = DigitalTwin(crop_type=CropType.WHEAT)
        assert twin.max_crop_height == 1.5
        assert twin.optimal_temp == 20.0

    def test_corn_parameters(self):
        """Corn has specific growth parameters."""
        twin = DigitalTwin(crop_type=CropType.CORN)
        assert twin.max_crop_height == 3.0
        assert twin.optimal_temp == 28.0

    def test_rice_parameters(self):
        """Rice has specific growth parameters."""
        twin = DigitalTwin(crop_type=CropType.RICE)
        assert twin.max_crop_height == 1.2
        assert twin.optimal_temp == 30.0

    def test_crop_specific_growth(self):
        """Different crops grow differently."""
        wheat_twin = DigitalTwin(crop_type=CropType.WHEAT)
        corn_twin = DigitalTwin(crop_type=CropType.CORN)

        wheat_state = SimulationState(
            soil_moisture=0.5,
            temperature=20.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        corn_state = SimulationState(
            soil_moisture=0.5,
            temperature=28.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        wheat_result = wheat_twin.simulate(wheat_state, days=30)
        corn_result = corn_twin.simulate(corn_state, days=30)
        # Corn should grow taller
        assert corn_result.final_state.crop_height > wheat_result.final_state.crop_height

    def test_phenological_stages(self):
        """Crop has phenological stages."""
        twin = DigitalTwin(crop_type=CropType.WHEAT)
        stages = twin.get_phenological_stages()
        assert len(stages) > 0
        assert "emergence" in stages
        assert "maturity" in stages

    def test_rotation_carryover(self):
        """Rotation affects soil nutrients."""
        twin = DigitalTwin(crop_type=CropType.WHEAT)
        # After legume rotation, nitrogen should be higher
        state_after_legume = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.7,  # Higher N from legume
        )
        state_no_rotation = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.3,
        )
        result_rotation = twin.simulate(state_after_legume, days=10)
        result_no_rotation = twin.simulate(state_no_rotation, days=10)
        assert result_rotation.final_state.crop_height > result_no_rotation.final_state.crop_height

    def test_crop_coefficient_by_crop(self):
        """Different crops have different Kc values."""
        wheat_twin = DigitalTwin(crop_type=CropType.WHEAT)
        corn_twin = DigitalTwin(crop_type=CropType.CORN)
        # Corn typically has higher Kc
        assert corn_twin.crop_coefficient >= wheat_twin.crop_coefficient
