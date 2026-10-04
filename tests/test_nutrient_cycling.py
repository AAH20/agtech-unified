"""Test nutrient cycling model for agricultural digital twin."""

import pytest

from src.digital_twin.nutrient_cycling import (
    FertilizerApplication,
    NutrientCyclingModel,
    NutrientState,
)


def test_nutrient_state_initial():
    """Initial nutrient state is correctly stored."""
    state = NutrientState(nitrogen=50.0, phosphorus=10.0, potassium=40.0, organic_matter=3.0)
    assert state.nitrogen == 50.0
    assert state.phosphorus == 10.0
    assert state.potassium == 40.0
    assert state.organic_matter == 3.0


def test_simulate_zero_days():
    """Zero-day simulation returns initial state."""
    model = NutrientCyclingModel()
    state = NutrientState(nitrogen=50.0, phosphorus=10.0, potassium=40.0, organic_matter=3.0)
    result = model.simulate(state, days=0)
    assert result.days_simulated == 0
    assert result.final_state.nitrogen == 50.0
    assert result.history == []


def test_simulate_negative_days_raises():
    """Negative days raises ValueError."""
    model = NutrientCyclingModel()
    state = NutrientState(nitrogen=50.0, phosphorus=10.0, potassium=40.0, organic_matter=3.0)
    with pytest.raises(ValueError, match="Days must be non-negative"):
        model.simulate(state, days=-1)


def test_fertilizer_increases_nutrients():
    """Fertilizer application increases nutrient levels."""
    model = NutrientCyclingModel()
    state = NutrientState(nitrogen=20.0, phosphorus=5.0, potassium=20.0, organic_matter=2.0)
    fert = [FertilizerApplication(nitrogen=30.0, phosphorus=10.0, potassium=20.0, day=0)]
    result = model.simulate(state, days=1, fertilizer_schedule=fert)
    assert result.budget.total_n_applied == 30.0
    assert result.budget.total_p_applied == 10.0
    assert result.budget.total_k_applied == 20.0


def test_nutrient_uptake_reduces_levels():
    """Crop uptake reduces soil nutrient levels."""
    model = NutrientCyclingModel()
    state = NutrientState(nitrogen=100.0, phosphorus=20.0, potassium=80.0, organic_matter=3.0)
    demand = {"n": 5.0, "p": 1.0, "k": 3.0}
    result = model.simulate(state, days=5, crop_demand=demand)
    assert result.budget.total_n_uptake > 0.0
    assert result.budget.total_p_uptake > 0.0
    assert result.budget.total_k_uptake > 0.0


def test_mineralization_adds_nitrogen():
    """Mineralization adds plant-available nitrogen."""
    model = NutrientCyclingModel(mineralization_rate=0.01)
    state = NutrientState(nitrogen=10.0, phosphorus=5.0, potassium=20.0, organic_matter=5.0)
    result = model.simulate(state, days=10)
    assert result.budget.total_n_mineralized > 0.0


def test_nutrient_deficiency_detection():
    """Nutrient deficiency is correctly detected."""
    model = NutrientCyclingModel()
    deficient = NutrientState(nitrogen=10.0, phosphorus=2.0, potassium=15.0, organic_matter=2.0)
    result = model.nutrient_deficiency(deficient)
    assert result["n"] is True
    assert result["p"] is True
    assert result["k"] is True


def test_nutrient_no_deficiency():
    """No deficiency when nutrient levels are adequate."""
    model = NutrientCyclingModel()
    adequate = NutrientState(nitrogen=50.0, phosphorus=15.0, potassium=60.0, organic_matter=3.0)
    result = model.nutrient_deficiency(adequate)
    assert result["n"] is False
    assert result["p"] is False
    assert result["k"] is False


def test_recommend_fertilizer():
    """Fertilizer recommendation fills deficit to target."""
    model = NutrientCyclingModel()
    state = NutrientState(nitrogen=30.0, phosphorus=8.0, potassium=40.0, organic_matter=2.5)
    rec = model.recommend_fertilizer(state, target_n=100.0, target_p=20.0, target_k=80.0)
    assert rec["n"] == pytest.approx(70.0)
    assert rec["p"] == pytest.approx(12.0)
    assert rec["k"] == pytest.approx(40.0)


def test_ph_factor_optimal():
    """pH factor is 1.0 at optimal pH."""
    model = NutrientCyclingModel(optimal_ph=6.5)
    assert model._ph_factor(6.5) == pytest.approx(1.0)


def test_ph_factor_deviation():
    """pH factor decreases with deviation from optimal."""
    model = NutrientCyclingModel(optimal_ph=6.5)
    factor_7_5 = model._ph_factor(7.5)
    factor_5_5 = model._ph_factor(5.5)
    assert factor_7_5 < 1.0
    assert factor_5_5 < 1.0
    assert factor_7_5 == pytest.approx(factor_5_5)


def test_n_use_efficiency():
    """N use efficiency is calculated correctly."""
    model = NutrientCyclingModel()
    state = NutrientState(nitrogen=50.0, phosphorus=10.0, potassium=40.0, organic_matter=3.0)
    fert = [FertilizerApplication(nitrogen=50.0, phosphorus=10.0, potassium=20.0, day=0)]
    demand = {"n": 3.0, "p": 0.5, "k": 1.5}
    result = model.simulate(state, days=10, crop_demand=demand, fertilizer_schedule=fert)
    assert result.budget.n_use_efficiency >= 0.0
    assert result.budget.n_use_efficiency <= 1.0
