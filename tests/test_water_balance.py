"""Test water balance and irrigation model for agricultural digital twin."""

import pytest

from src.digital_twin.water_balance import (
    WaterBalanceModel,
    WaterBalanceState,
)


def test_water_balance_initial_state():
    """Initial state is correctly stored."""
    state = WaterBalanceState(soil_moisture=0.25)
    assert state.soil_moisture == 0.25
    assert state.field_capacity == 0.30
    assert state.wilting_point == 0.10


def test_water_stress_factor_no_stress():
    """Stress factor is 1.0 at or above field capacity."""
    model = WaterBalanceModel(field_capacity=0.30, wilting_point=0.10)
    assert model.water_stress_factor(0.30) == 1.0
    assert model.water_stress_factor(0.35) == 1.0


def test_water_stress_factor_severe_stress():
    """Stress factor is 0.0 at or below wilting point."""
    model = WaterBalanceModel(field_capacity=0.30, wilting_point=0.10)
    assert model.water_stress_factor(0.10) == 0.0
    assert model.water_stress_factor(0.05) == 0.0


def test_water_stress_factor_intermediate():
    """Stress factor is linear between wilting point and field capacity."""
    model = WaterBalanceModel(field_capacity=0.30, wilting_point=0.10)
    # Midpoint: (0.20 - 0.10) / (0.30 - 0.10) = 0.5
    assert model.water_stress_factor(0.20) == pytest.approx(0.5)


def test_simulate_zero_days():
    """Zero-day simulation returns initial state."""
    model = WaterBalanceModel()
    state = WaterBalanceState(soil_moisture=0.25)
    result = model.simulate(state, daily_weather=[])
    assert result.days_simulated == 0
    assert result.final_state.soil_moisture == 0.25
    assert result.history == []


def test_simulate_rain_increases_moisture():
    """Precipitation increases soil moisture."""
    model = WaterBalanceModel()
    state = WaterBalanceState(soil_moisture=0.15)
    weather = [{"precipitation": 20.0, "evapotranspiration": 2.0}]
    result = model.simulate(state, daily_weather=weather)
    assert result.final_state.soil_moisture > 0.15
    assert result.total_precipitation == 20.0


def test_simulate_et_decreases_moisture():
    """Evapotranspiration decreases soil moisture."""
    model = WaterBalanceModel()
    state = WaterBalanceState(soil_moisture=0.25)
    weather = [{"precipitation": 0.0, "evapotranspiration": 10.0}]
    result = model.simulate(state, daily_weather=weather)
    assert result.final_state.soil_moisture < 0.25
    assert result.total_evapotranspiration > 0.0


def test_simulate_drainage_above_field_capacity():
    """Excess water above field capacity drains."""
    model = WaterBalanceModel(field_capacity=0.30)
    state = WaterBalanceState(soil_moisture=0.29)
    weather = [{"precipitation": 10.0, "evapotranspiration": 0.0}]
    result = model.simulate(state, daily_weather=weather)
    assert result.total_drainage > 0.0
    # After one step with 5% drainage rate, moisture is still above FC
    assert result.final_state.soil_moisture > 0.30


def test_irrigation_recommendation_no_stress():
    """No irrigation recommended when moisture is adequate."""
    model = WaterBalanceModel(field_capacity=0.30, wilting_point=0.10)
    rec = model.irrigation_recommendation(current_moisture=0.28, forecast_et=5.0)
    assert rec.should_irrigate is False
    assert rec.amount_mm == 0.0


def test_irrigation_recommendation_severe_stress():
    """Irrigation recommended under severe water stress."""
    model = WaterBalanceModel(field_capacity=0.30, wilting_point=0.10)
    rec = model.irrigation_recommendation(current_moisture=0.11, forecast_et=8.0)
    assert rec.should_irrigate is True
    assert rec.amount_mm > 0.0
    assert rec.priority <= 2


def test_water_use_efficiency():
    """WUE is calculated correctly."""
    model = WaterBalanceModel()
    wue = model.calculate_water_use_efficiency(total_irrigation=100.0, total_et=80.0)
    assert wue == pytest.approx(0.8)


def test_water_use_efficiency_no_irrigation():
    """WUE is 0 when no irrigation was applied."""
    model = WaterBalanceModel()
    wue = model.calculate_water_use_efficiency(total_irrigation=0.0, total_et=50.0)
    assert wue == 0.0


def test_simulate_water_stress_days_counted():
    """Water stress days are counted correctly."""
    model = WaterBalanceModel(field_capacity=0.30, wilting_point=0.10)
    state = WaterBalanceState(soil_moisture=0.12)
    weather = [{"precipitation": 0.0, "evapotranspiration": 5.0}] * 5
    result = model.simulate(state, daily_weather=weather)
    assert result.water_stress_days > 0
