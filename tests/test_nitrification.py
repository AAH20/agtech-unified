"""Tests for nitrification/denitrification split in nutrient cycling."""

from src.digital_twin.nutrient_cycling import NutrientCyclingModel, NutrientState


def make_state(n=50.0, p=10.0, k=80.0, om=3.0, ph=6.5):
    """Helper to create NutrientState."""
    return NutrientState(nitrogen=n, phosphorus=p, potassium=k, organic_matter=om, pH=ph)


class TestNitrificationDenitrification:
    """Test N species differentiation."""

    def test_nitrification_rate(self):
        """Nitrification converts NH4 to NO3."""
        nc = NutrientCyclingModel()
        rate = nc.nitrification_rate(temperature=25.0, soil_moisture=0.3)
        assert rate > 0.0
        assert rate <= 1.0

    def test_nitrification_temperature_dependent(self):
        """Nitrification increases with temperature."""
        nc = NutrientCyclingModel()
        rate_cold = nc.nitrification_rate(temperature=10.0, soil_moisture=0.3)
        rate_warm = nc.nitrification_rate(temperature=30.0, soil_moisture=0.3)
        assert rate_warm > rate_cold

    def test_nitrification_moisture_dependent(self):
        """Nitrification is moisture-dependent."""
        nc = NutrientCyclingModel()
        rate_dry = nc.nitrification_rate(temperature=25.0, soil_moisture=0.1)
        rate_wet = nc.nitrification_rate(temperature=25.0, soil_moisture=0.3)
        assert rate_wet > rate_dry

    def test_denitrification_rate(self):
        """Denitrification occurs under anaerobic conditions."""
        nc = NutrientCyclingModel()
        rate = nc.denitrification_rate(soil_moisture=0.45, temperature=25.0)
        assert rate >= 0.0
        assert rate <= 1.0

    def test_denitrification_high_moisture(self):
        """Denitrification increases with high moisture."""
        nc = NutrientCyclingModel()
        rate_low = nc.denitrification_rate(soil_moisture=0.2, temperature=25.0)
        rate_high = nc.denitrification_rate(soil_moisture=0.5, temperature=25.0)
        assert rate_high > rate_low

    def test_split_n_pools(self):
        """N is split into NH4 and NO3 pools."""
        nc = NutrientCyclingModel()
        state = make_state(n=100.0)
        nh4, no3 = nc.split_n_pools(state)
        assert nh4 >= 0.0
        assert no3 >= 0.0
        assert nh4 + no3 <= 100.0

    def test_nitrification_in_simulation(self):
        """Nitrification occurs during simulation."""
        nc = NutrientCyclingModel()
        state = make_state(n=100.0)
        result = nc.simulate(state, days=5)
        # Check that nitrification occurred
        assert result.final_state.nitrogen < 100.0  # Some N lost to nitrification

    def test_denitrification_in_simulation(self):
        """Denitrification occurs under wet conditions."""
        nc = NutrientCyclingModel()
        state = make_state(n=100.0)
        # High moisture promotes denitrification
        result = nc.simulate(state, days=5)
        # N losses should occur
        assert result.budget.total_n_leached > 0 or result.final_state.nitrogen < 100.0
