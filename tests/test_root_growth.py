"""Tests for root growth model (ROOT-001, SIM-003)."""

import pytest

from src.digital_twin.root_growth import (
    RootGrowthModel,
    RootGrowthResult,
    RootGrowthState,
    RootLayerState,
)


@pytest.fixture
def root_model():
    return RootGrowthModel(
        max_root_depth=1.5,
        initial_root_depth=0.1,
        root_growth_rate=0.08,
        n_layers=5,
    )


def _moisture_profile(n_layers, root_depth, base=0.3):
    """Build a moisture profile covering the current root zone."""
    thickness = root_depth / n_layers
    return [
        RootLayerState(
            depth_top=i * thickness,
            depth_bottom=(i + 1) * thickness,
            root_length_density=1.0,
            soil_moisture=base,
        )
        for i in range(n_layers)
    ]


# ---------------------------------------------------------------------------
# Dynamic root zone expansion
# ---------------------------------------------------------------------------


def test_root_depth_increases_over_time(root_model):
    """Root depth grows during simulation."""
    result = root_model.simulate(days=30, daily_water_demand=5.0)
    assert result.final_state.root_depth > root_model.initial_root_depth


def test_root_depth_bounded_by_max(root_model):
    """Root depth never exceeds the crop-specific maximum."""
    result = root_model.simulate(days=365, daily_water_demand=5.0)
    assert result.final_state.root_depth <= root_model.max_root_depth


def test_root_depth_monotonic_non_decreasing(root_model):
    """Root depth never shrinks day over day."""
    result = root_model.simulate(days=60, daily_water_demand=5.0)
    depths = [s.root_depth for s in result.history]
    assert all(b >= a for a, b in zip(depths, depths[1:]))


def test_root_depth_logistic_shape(root_model):
    """Expansion is slow early, fast mid, slow near max (S-curve)."""
    d10 = root_model.root_depth_at_day(10)
    d50 = root_model.root_depth_at_day(50)
    d100 = root_model.root_depth_at_day(100)
    early_rate = (d50 - d10) / 40.0
    late_rate = (d100 - d50) / 50.0
    assert early_rate > late_rate


def test_root_depth_at_day_zero_is_initial(root_model):
    """Day 0 depth equals the initial root depth."""
    assert root_model.root_depth_at_day(0) == pytest.approx(root_model.initial_root_depth)


# ---------------------------------------------------------------------------
# Root distribution
# ---------------------------------------------------------------------------


def test_root_distribution_decays_with_depth(root_model):
    """Root length density is highest at the surface and declines with depth."""
    surface = root_model.root_distribution(0.0)
    deep = root_model.root_distribution(1.0)
    assert surface > deep


def test_root_distribution_non_negative(root_model):
    """Root distribution is never negative at any depth."""
    for d in (0.0, 0.25, 0.5, 1.0, 1.5):
        assert root_model.root_distribution(d) >= 0.0


def test_root_length_density_increases_over_time(root_model):
    """Total root length density grows as the root system develops."""
    result = root_model.simulate(days=45, daily_water_demand=5.0)
    assert result.final_state.total_root_length > result.history[0].total_root_length


# ---------------------------------------------------------------------------
# Root water uptake by depth
# ---------------------------------------------------------------------------


def test_wet_layer_uptake_exceeds_dry_layer(root_model):
    """A wet soil layer supplies more water than a dry one at the same depth."""
    wet = RootLayerState(0.0, 0.1, 2.0, soil_moisture=0.35)
    dry = RootLayerState(0.0, 0.1, 2.0, soil_moisture=0.05)
    uptake_wet = root_model.layer_water_uptake(wet, demand=5.0)
    uptake_dry = root_model.layer_water_uptake(dry, demand=5.0)
    assert uptake_wet > uptake_dry


def test_layer_uptake_bounded_by_demand(root_model):
    """A layer cannot supply more water than the crop demands."""
    layer = RootLayerState(0.0, 0.1, 2.0, soil_moisture=0.4)
    uptake = root_model.layer_water_uptake(layer, demand=3.0)
    assert uptake <= 3.0 + 1e-9


def test_profile_uptake_bounded_by_demand_and_capacity(root_model):
    """Profile uptake never exceeds demand or total layer capacity."""
    layers = _moisture_profile(5, root_model.initial_root_depth)
    demand = 6.0
    total = root_model.profile_water_uptake(layers, demand)
    capacity = sum(root_model.layer_uptake(l, demand) for l in layers)
    assert total <= demand + 1e-9
    assert total <= capacity + 1e-9


def test_profile_uptake_meets_demand_when_capacity_sufficient(root_model):
    """When layers can supply enough, profile uptake equals demand."""
    layers = [RootLayerState(0.0, 0.5, 5.0, soil_moisture=0.4) for _ in range(5)]
    demand = 2.0
    total = root_model.profile_water_uptake(layers, demand)
    assert total == pytest.approx(demand)


def test_uptake_zero_without_roots(root_model):
    """A layer with no roots contributes no uptake."""
    layer = RootLayerState(0.0, 0.1, root_length_density=0.0, soil_moisture=0.4)
    assert root_model.layer_water_uptake(layer, demand=5.0) == 0.0


def test_uptake_zero_at_wilting_point(root_model):
    """No uptake from soil at the wilting point."""
    layer = RootLayerState(0.0, 0.1, 2.0, soil_moisture=root_model.wilting_point)
    assert root_model.layer_water_uptake(layer, demand=5.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------


def test_simulate_history_length(root_model):
    """History has one entry per simulated day."""
    result = root_model.simulate(days=20, daily_water_demand=5.0)
    assert len(result.history) == 20


def test_simulate_result_fields(root_model):
    """Result exposes documented aggregate fields."""
    result = root_model.simulate(days=10, daily_water_demand=5.0)
    assert isinstance(result, RootGrowthResult)
    assert result.days_simulated == 10
    assert result.total_water_uptake >= 0.0
    assert result.max_root_depth <= root_model.max_root_depth


def test_simulate_zero_days(root_model):
    """Zero-day simulation returns the initial state."""
    result = root_model.simulate(days=0, daily_water_demand=5.0)
    assert result.days_simulated == 0
    assert result.final_state.root_depth == pytest.approx(root_model.initial_root_depth)


def test_final_state_type(root_model):
    """Final state is a RootGrowthState with layers."""
    result = root_model.simulate(days=5, daily_water_demand=5.0)
    assert isinstance(result.final_state, RootGrowthState)
    assert len(result.final_state.layers) == root_model.n_layers


def test_negative_days_raises(root_model):
    """Negative simulation days raise ValueError."""
    with pytest.raises(ValueError, match="Days must be non-negative"):
        root_model.simulate(days=-1, daily_water_demand=5.0)
