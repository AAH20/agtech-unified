"""Tests for yield prediction model (YIELD-001)."""

import pytest

from src.digital_twin.yield_prediction import (
    GrainQuality,
    YieldModel,
    YieldPrediction,
)


@pytest.fixture
def wheat_model():
    return YieldModel(crop_type="wheat")


@pytest.fixture
def corn_model():
    return YieldModel(crop_type="corn")


# ---------------------------------------------------------------------------
# Harvest index × biomass
# ---------------------------------------------------------------------------


def test_grain_yield_is_biomass_times_harvest_index(wheat_model):
    """Core identity: grain_yield = total_biomass × harvest_index."""
    pred = wheat_model.predict_yield(total_biomass=10000.0, harvest_index=0.5)
    assert pred.grain_yield == pytest.approx(5000.0)


def test_grain_yield_tons_conversion(wheat_model):
    """Tonnes conversion is consistent with kg/ha value."""
    pred = wheat_model.predict_yield(total_biomass=8000.0, harvest_index=0.4)
    assert pred.grain_yield_tons == pytest.approx(pred.grain_yield / 1000.0)


def test_zero_biomass_gives_zero_yield(wheat_model):
    """No biomass means no grain regardless of harvest index."""
    pred = wheat_model.predict_yield(total_biomass=0.0, harvest_index=0.5)
    assert pred.grain_yield == 0.0
    assert pred.grain_yield_tons == 0.0


def test_harvest_index_within_bounds(wheat_model):
    """Harvest index stays in (0, 1] even under stress."""
    pred = wheat_model.predict_yield(
        total_biomass=10000.0, harvest_index=0.5, water_stress=0.0, nutrient_stress=0.0
    )
    assert 0.0 < pred.harvest_index <= 1.0


# ---------------------------------------------------------------------------
# Stress effects
# ---------------------------------------------------------------------------


def test_water_stress_reduces_harvest_index(wheat_model):
    """Water stress lowers the effective harvest index."""
    good = wheat_model.predict_yield(total_biomass=10000.0, water_stress=1.0)
    bad = wheat_model.predict_yield(total_biomass=10000.0, water_stress=0.0)
    assert good.harvest_index > bad.harvest_index
    assert good.grain_yield > bad.grain_yield


def test_nutrient_stress_reduces_harvest_index(wheat_model):
    """Nutrient stress lowers the effective harvest index."""
    good = wheat_model.predict_yield(total_biomass=10000.0, nutrient_stress=1.0)
    bad = wheat_model.predict_yield(total_biomass=10000.0, nutrient_stress=0.0)
    assert good.harvest_index > bad.harvest_index


def test_limiting_factor_is_water(wheat_model):
    """When water is the strongest stress, it is reported as limiting."""
    pred = wheat_model.predict_yield(total_biomass=10000.0, water_stress=0.2, nutrient_stress=0.9)
    assert pred.limiting_factor == "water"


def test_limiting_factor_is_nutrient(wheat_model):
    """When nutrient is the strongest stress, it is reported as limiting."""
    pred = wheat_model.predict_yield(total_biomass=10000.0, water_stress=0.9, nutrient_stress=0.3)
    assert pred.limiting_factor == "nutrient"


def test_no_stress_reports_no_limiting_factor(wheat_model):
    """Without stress the limiting factor is 'none'."""
    pred = wheat_model.predict_yield(total_biomass=10000.0, water_stress=1.0, nutrient_stress=1.0)
    assert pred.limiting_factor == "none"
    assert pred.stress_adjusted is False


# ---------------------------------------------------------------------------
# Crop-specific parameters
# ---------------------------------------------------------------------------


def test_crop_specific_harvest_index(wheat_model, corn_model):
    """Different crops have different base harvest indices."""
    w = wheat_model.predict_yield(total_biomass=10000.0)
    c = corn_model.predict_yield(total_biomass=10000.0)
    assert w.harvest_index != c.harvest_index


def test_unknown_crop_falls_back_to_generic():
    """Unknown crop name uses generic parameters without error."""
    model = YieldModel(crop_type="quinoa")
    pred = model.predict_yield(total_biomass=5000.0)
    assert pred.grain_yield > 0.0


# ---------------------------------------------------------------------------
# Biomass estimation
# ---------------------------------------------------------------------------


def test_biomass_estimation_positive():
    """Biomass estimation from height/LAI is positive for positive inputs."""
    model = YieldModel(crop_type="wheat")
    biomass = model.estimate_biomass(crop_height=1.0, lai=3.0)
    assert biomass > 0.0


def test_biomass_increases_with_lai():
    """More leaf area means more biomass."""
    model = YieldModel(crop_type="wheat")
    low = model.estimate_biomass(crop_height=1.0, lai=1.0)
    high = model.estimate_biomass(crop_height=1.0, lai=4.0)
    assert high > low


def test_biomass_zero_when_no_leaf_area():
    """No leaf area means no biomass accumulation."""
    model = YieldModel(crop_type="wheat")
    assert model.estimate_biomass(crop_height=1.0, lai=0.0) == 0.0


# ---------------------------------------------------------------------------
# Grain quality
# ---------------------------------------------------------------------------


def test_grain_quality_metrics_present(wheat_model):
    """Prediction includes protein, moisture, test weight, TKW."""
    pred = wheat_model.predict_yield(total_biomass=10000.0)
    q = pred.grain_quality
    assert isinstance(q, GrainQuality)
    assert q.protein_content > 0.0
    assert 0.0 <= q.moisture_content <= 100.0
    assert q.test_weight > 0.0
    assert q.thousand_kernel_weight > 0.0


def test_grain_quality_has_grade(wheat_model):
    """Prediction assigns a quality grade string."""
    pred = wheat_model.predict_yield(total_biomass=10000.0)
    assert isinstance(pred.grain_quality.grade, str)
    assert len(pred.grain_quality.grade) > 0


def test_water_stress_increases_protein_concentration(wheat_model):
    """Water stress concentrates grain protein (smaller grains, higher %).

    Well-documented wheat physiology: drought during grain fill reduces
    kernel size more than protein accumulation, raising protein percentage.
    water_stress=0.0 is severe stress; 1.0 is no stress.
    """
    unstressed = wheat_model.predict_yield(total_biomass=10000.0, water_stress=1.0)
    stressed = wheat_model.predict_yield(total_biomass=10000.0, water_stress=0.0)
    assert stressed.grain_quality.protein_content > unstressed.grain_quality.protein_content


def test_moisture_content_within_bounds(wheat_model):
    """Grain moisture stays within physical limits under any stress."""
    for ws in (0.0, 0.5, 1.0):
        pred = wheat_model.predict_yield(total_biomass=10000.0, water_stress=ws)
        assert 0.0 <= pred.grain_quality.moisture_content <= 100.0


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


def test_prediction_result_fields(wheat_model):
    """YieldPrediction exposes all documented fields."""
    pred = wheat_model.predict_yield(total_biomass=10000.0)
    assert isinstance(pred, YieldPrediction)
    assert pred.total_biomass == 10000.0
    assert pred.days_to_maturity > 0
