"""Tests for weed competition model: density-dependent resource preemption."""

import pytest

from src.digital_twin.weed_competition import (
    WeedCompetitionModel,
    WeedState,
)


class TestLightCompetition:
    """Light preemption by weeds."""

    def test_light_competition_increases_with_weed_density(self):
        """More weeds = more light competition."""
        model = WeedCompetitionModel()
        c_low = model.light_competition(weed_density=5.0, crop_lai=3.0)
        c_high = model.light_competition(weed_density=50.0, crop_lai=3.0)
        assert c_high > c_low

    def test_light_competition_zero_without_weeds(self):
        """No weeds means no light competition."""
        model = WeedCompetitionModel()
        assert model.light_competition(weed_density=0.0, crop_lai=3.0) == 0.0

    def test_light_competition_bounded(self):
        """Light competition factor stays in [0, 1]."""
        model = WeedCompetitionModel()
        c = model.light_competition(weed_density=500.0, crop_lai=0.5)
        assert 0.0 <= c <= 1.0

    def test_higher_crop_lai_reduces_competition(self):
        """Taller crop canopy reduces weed light competition."""
        model = WeedCompetitionModel()
        c_low = model.light_competition(weed_density=30.0, crop_lai=1.0)
        c_high = model.light_competition(weed_density=30.0, crop_lai=5.0)
        assert c_high < c_low


class TestWaterCompetition:
    """Water preemption by weeds."""

    def test_water_competition_increases_with_weed_density(self):
        """More weeds = more water competition."""
        model = WeedCompetitionModel()
        c_low = model.water_competition(weed_density=5.0, soil_moisture=0.3)
        c_high = model.water_competition(weed_density=50.0, soil_moisture=0.3)
        assert c_high > c_low

    def test_water_competition_zero_without_weeds(self):
        """No weeds means no water competition."""
        model = WeedCompetitionModel()
        assert model.water_competition(weed_density=0.0, soil_moisture=0.3) == 0.0

    def test_water_competition_higher_in_dry_soil(self):
        """Water competition more severe in dry soil."""
        model = WeedCompetitionModel()
        c_wet = model.water_competition(weed_density=30.0, soil_moisture=0.4)
        c_dry = model.water_competition(weed_density=30.0, soil_moisture=0.1)
        assert c_dry > c_wet


class TestNutrientCompetition:
    """Nutrient preemption by weeds."""

    def test_nutrient_competition_increases_with_weed_density(self):
        """More weeds = more nutrient competition."""
        model = WeedCompetitionModel()
        c_low = model.nutrient_competition(weed_density=5.0, soil_nitrogen=100.0)
        c_high = model.nutrient_competition(weed_density=50.0, soil_nitrogen=100.0)
        assert c_high > c_low

    def test_nutrient_competition_zero_without_weeds(self):
        """No weeds means no nutrient competition."""
        model = WeedCompetitionModel()
        assert model.nutrient_competition(weed_density=0.0, soil_nitrogen=100.0) == 0.0

    def test_nutrient_competition_higher_in_low_n_soil(self):
        """Nutrient competition more severe in N-deficient soil."""
        model = WeedCompetitionModel()
        c_rich = model.nutrient_competition(weed_density=30.0, soil_nitrogen=200.0)
        c_poor = model.nutrient_competition(weed_density=30.0, soil_nitrogen=20.0)
        assert c_poor > c_rich


class TestCombinedCompetitionIndex:
    """Combined competition index."""

    def test_combined_index_zero_without_weeds(self):
        """No weeds means zero competition index."""
        model = WeedCompetitionModel()
        assert model.competition_index(weed_density=0.0, crop_lai=3.0) == 0.0

    def test_combined_index_increases_with_density(self):
        """Competition index increases with weed density."""
        model = WeedCompetitionModel()
        c_low = model.competition_index(weed_density=5.0, crop_lai=3.0)
        c_high = model.competition_index(weed_density=50.0, crop_lai=3.0)
        assert c_high > c_low

    def test_combined_index_bounded(self):
        """Competition index stays in [0, 1]."""
        model = WeedCompetitionModel()
        c = model.competition_index(weed_density=1000.0, crop_lai=0.1)
        assert 0.0 <= c <= 1.0


class TestYieldLoss:
    """Yield loss estimation from weed competition."""

    def test_yield_loss_zero_without_weeds(self):
        """No weeds means no yield loss."""
        model = WeedCompetitionModel()
        result = model.yield_loss(weed_density=0.0, crop_lai=3.0)
        assert result.yield_loss_fraction == 0.0
        assert result.yield_loss_percent == 0.0

    def test_yield_loss_increases_with_density(self):
        """More weeds = more yield loss."""
        model = WeedCompetitionModel()
        r_low = model.yield_loss(weed_density=5.0, crop_lai=3.0)
        r_high = model.yield_loss(weed_density=50.0, crop_lai=3.0)
        assert r_high.yield_loss_fraction > r_low.yield_loss_fraction

    def test_yield_loss_bounded(self):
        """Yield loss fraction stays in [0, 1]."""
        model = WeedCompetitionModel()
        result = model.yield_loss(weed_density=1000.0, crop_lai=0.1)
        assert 0.0 <= result.yield_loss_fraction <= 1.0

    def test_yield_loss_percent_is_fraction_times_100(self):
        """Yield loss percent = fraction * 100."""
        model = WeedCompetitionModel()
        result = model.yield_loss(weed_density=30.0, crop_lai=3.0)
        assert result.yield_loss_percent == pytest.approx(result.yield_loss_fraction * 100.0)

    def test_higher_crop_lai_reduces_yield_loss(self):
        """Taller crop reduces yield loss from same weed density."""
        model = WeedCompetitionModel()
        r_low = model.yield_loss(weed_density=30.0, crop_lai=1.0)
        r_high = model.yield_loss(weed_density=30.0, crop_lai=5.0)
        assert r_high.yield_loss_fraction < r_low.yield_loss_fraction


class TestWeedPopulationDynamics:
    """Weed population growth dynamics."""

    def test_weed_population_grows(self):
        """Weed population increases over time."""
        model = WeedCompetitionModel()
        state = WeedState(density=10.0, biomass_g_m2=50.0)
        result = model.simulate(initial_state=state, days=30)
        assert result.final_state.density > 10.0
        assert result.days_simulated == 30

    def test_weed_population_logistic_growth(self):
        """Weed population follows logistic growth."""
        model = WeedCompetitionModel()
        state = WeedState(density=10.0, biomass_g_m2=50.0)
        result = model.simulate(initial_state=state, days=60)
        assert result.final_state.density <= model.carrying_capacity

    def test_weed_population_zero_stays_zero(self):
        """Zero weed density stays zero."""
        model = WeedCompetitionModel()
        state = WeedState(density=0.0, biomass_g_m2=0.0)
        result = model.simulate(initial_state=state, days=30)
        assert result.final_state.density == 0.0

    def test_weed_biomass_increases(self):
        """Weed biomass increases over time."""
        model = WeedCompetitionModel()
        state = WeedState(density=10.0, biomass_g_m2=50.0)
        result = model.simulate(initial_state=state, days=30)
        assert result.final_state.biomass_g_m2 > 50.0

    def test_negative_days_raises(self):
        """Negative days raises ValueError."""
        model = WeedCompetitionModel()
        state = WeedState(density=10.0, biomass_g_m2=50.0)
        with pytest.raises(ValueError, match="Days must be non-negative"):
            model.simulate(initial_state=state, days=-1)

    def test_zero_days_returns_initial_state(self):
        """Zero simulation days returns initial state."""
        model = WeedCompetitionModel()
        state = WeedState(density=10.0, biomass_g_m2=50.0)
        result = model.simulate(initial_state=state, days=0)
        assert result.days_simulated == 0
        assert result.final_state.density == 10.0
        assert len(result.history) == 0


class TestCompetitionThresholds:
    """Economic threshold and critical period."""

    def test_economic_threshold_density(self):
        """Economic threshold returns a positive density."""
        model = WeedCompetitionModel()
        threshold = model.economic_threshold(crop_value=500.0, control_cost=50.0)
        assert threshold > 0.0

    def test_economic_threshold_higher_crop_value(self):
        """Higher crop value lowers economic threshold."""
        model = WeedCompetitionModel()
        t_low = model.economic_threshold(crop_value=200.0, control_cost=50.0)
        t_high = model.economic_threshold(crop_value=1000.0, control_cost=50.0)
        assert t_high < t_low

    def test_economic_threshold_higher_control_cost(self):
        """Higher control cost raises economic threshold."""
        model = WeedCompetitionModel()
        t_low = model.economic_threshold(crop_value=500.0, control_cost=20.0)
        t_high = model.economic_threshold(crop_value=500.0, control_cost=100.0)
        assert t_high > t_low

    def test_critical_period(self):
        """Critical period returns start and end days."""
        model = WeedCompetitionModel()
        start, end = model.critical_period(crop_type="corn")
        assert start < end
        assert start >= 0
