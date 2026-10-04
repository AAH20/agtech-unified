"""Tests for FarmState history, serialization, diff, and transitions."""

import json

import pytest

from src.integration.farm_state import (
    FarmState,
    FarmStateDiff,
    FarmStateHistory,
    StateTransitionValidator,
)


class TestFarmStateSerialization:
    """FarmState to_dict/from_dict/JSON round-trip."""

    def test_to_dict(self):
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.3,
            nutrient_level=0.6,
            pest_pressure=0.2,
        )
        d = state.to_dict()
        assert d["soil_moisture"] == 0.5
        assert d["temperature"] == 25.0
        assert d["crop_height"] == 0.3
        assert d["nutrient_level"] == 0.6
        assert d["pest_pressure"] == 0.2

    def test_from_dict(self):
        d = {
            "soil_moisture": 0.5,
            "temperature": 25.0,
            "crop_height": 0.3,
            "nutrient_level": 0.6,
            "pest_pressure": 0.2,
        }
        state = FarmState.from_dict(d)
        assert state.soil_moisture == 0.5
        assert state.temperature == 25.0

    def test_to_json(self):
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.3,
            nutrient_level=0.6,
            pest_pressure=0.2,
        )
        j = state.to_json()
        assert isinstance(j, str)
        parsed = json.loads(j)
        assert parsed["soil_moisture"] == 0.5

    def test_from_json(self):
        j = json.dumps(
            {
                "soil_moisture": 0.5,
                "temperature": 25.0,
                "crop_height": 0.3,
                "nutrient_level": 0.6,
                "pest_pressure": 0.2,
            }
        )
        state = FarmState.from_json(j)
        assert state.soil_moisture == 0.5
        assert state.temperature == 25.0

    def test_round_trip(self):
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.3,
            nutrient_level=0.6,
            pest_pressure=0.2,
        )
        j = state.to_json()
        restored = FarmState.from_json(j)
        assert restored.soil_moisture == state.soil_moisture
        assert restored.temperature == state.temperature
        assert restored.crop_height == state.crop_height
        assert restored.nutrient_level == state.nutrient_level
        assert restored.pest_pressure == state.pest_pressure


class TestFarmStateDiff:
    """FarmState diff between two snapshots."""

    def test_no_change(self):
        s1 = FarmState(0.5, 25.0, 0.3, 0.6, 0.2)
        s2 = FarmState(0.5, 25.0, 0.3, 0.6, 0.2)
        diff = FarmStateDiff.compute(s1, s2)
        assert diff.changed_fields == {}

    def test_single_field_change(self):
        s1 = FarmState(0.5, 25.0, 0.3, 0.6, 0.2)
        s2 = FarmState(0.4, 25.0, 0.3, 0.6, 0.2)
        diff = FarmStateDiff.compute(s1, s2)
        assert "soil_moisture" in diff.changed_fields
        delta, new_val, pct = diff.changed_fields["soil_moisture"]
        assert delta == pytest.approx(-0.1)
        assert new_val == pytest.approx(0.4)

    def test_multiple_field_changes(self):
        s1 = FarmState(0.5, 25.0, 0.3, 0.6, 0.2)
        s2 = FarmState(0.4, 26.0, 0.5, 0.6, 0.2)
        diff = FarmStateDiff.compute(s1, s2)
        assert len(diff.changed_fields) == 3
        assert "soil_moisture" in diff.changed_fields
        assert "temperature" in diff.changed_fields
        assert "crop_height" in diff.changed_fields

    def test_magnitude(self):
        s1 = FarmState(0.5, 25.0, 0.3, 0.6, 0.2)
        s2 = FarmState(0.3, 25.0, 0.3, 0.6, 0.2)
        diff = FarmStateDiff.compute(s1, s2)
        assert diff.changed_fields["soil_moisture"][0] == pytest.approx(-0.2)

    def test_percent_change(self):
        s1 = FarmState(0.5, 25.0, 0.3, 0.6, 0.2)
        s2 = FarmState(0.25, 25.0, 0.3, 0.6, 0.2)
        diff = FarmStateDiff.compute(s1, s2)
        assert diff.changed_fields["soil_moisture"][2] == pytest.approx(50.0)

    def test_has_changes(self):
        s1 = FarmState(0.5, 25.0, 0.3, 0.6, 0.2)
        s2 = FarmState(0.5, 25.0, 0.3, 0.6, 0.2)
        diff = FarmStateDiff.compute(s1, s2)
        assert diff.has_changes() is False

    def test_has_changes_true(self):
        s1 = FarmState(0.5, 25.0, 0.3, 0.6, 0.2)
        s2 = FarmState(0.4, 25.0, 0.3, 0.6, 0.2)
        diff = FarmStateDiff.compute(s1, s2)
        assert diff.has_changes() is True

    def test_to_dict(self):
        s1 = FarmState(0.5, 25.0, 0.3, 0.6, 0.2)
        s2 = FarmState(0.4, 25.0, 0.3, 0.6, 0.2)
        diff = FarmStateDiff.compute(s1, s2)
        d = diff.to_dict()
        assert "soil_moisture" in d
        assert "old" in d["soil_moisture"]
        assert "new" in d["soil_moisture"]


class TestFarmStateHistory:
    """FarmState time-series history."""

    def test_append_and_get(self):
        history = FarmStateHistory()
        state = FarmState(0.5, 25.0, 0.3, 0.6, 0.2)
        history.append(state)
        assert len(history.get_all()) == 1

    def test_get_by_time_range(self):
        history = FarmStateHistory()
        state = FarmState(0.5, 25.0, 0.3, 0.6, 0.2)
        history.append(state)
        results = history.get_by_range(state.timestamp, state.timestamp)
        assert len(results) == 1

    def test_get_latest(self):
        history = FarmStateHistory()
        s1 = FarmState(0.5, 25.0, 0.3, 0.6, 0.2)
        s2 = FarmState(0.4, 26.0, 0.5, 0.6, 0.2)
        history.append(s1)
        history.append(s2)
        latest = history.get_latest()
        assert latest.soil_moisture == 0.4
        assert latest.temperature == 26.0

    def test_trend_analysis(self):
        history = FarmStateHistory()
        history.append(FarmState(0.8, 25.0, 0.3, 0.6, 0.2))
        history.append(FarmState(0.6, 25.0, 0.3, 0.6, 0.2))
        history.append(FarmState(0.4, 25.0, 0.3, 0.6, 0.2))
        trend = history.get_trend("soil_moisture")
        assert trend == "decreasing"

    def test_trend_increasing(self):
        history = FarmStateHistory()
        history.append(FarmState(0.2, 25.0, 0.3, 0.6, 0.2))
        history.append(FarmState(0.4, 25.0, 0.3, 0.6, 0.2))
        history.append(FarmState(0.6, 25.0, 0.3, 0.6, 0.2))
        trend = history.get_trend("soil_moisture")
        assert trend == "increasing"

    def test_trend_stable(self):
        history = FarmStateHistory()
        history.append(FarmState(0.5, 25.0, 0.3, 0.6, 0.2))
        history.append(FarmState(0.5, 25.0, 0.3, 0.6, 0.2))
        history.append(FarmState(0.5, 25.0, 0.3, 0.6, 0.2))
        trend = history.get_trend("soil_moisture")
        assert trend == "stable"

    def test_rate_of_change(self):
        history = FarmStateHistory()
        history.append(FarmState(0.8, 25.0, 0.3, 0.6, 0.2))
        history.append(FarmState(0.4, 25.0, 0.3, 0.6, 0.2))
        rate = history.get_rate_of_change("soil_moisture")
        assert rate < 0  # decreasing

    def test_clear(self):
        history = FarmStateHistory()
        history.append(FarmState(0.5, 25.0, 0.3, 0.6, 0.2))
        history.clear()
        assert len(history.get_all()) == 0

    def test_max_size(self):
        history = FarmStateHistory(max_size=3)
        for i in range(5):
            history.append(FarmState(0.1 * i, 25.0, 0.3, 0.6, 0.2))
        assert len(history.get_all()) == 3


class TestStateTransitionValidator:
    """State transition validation."""

    def test_valid_transition(self):
        validator = StateTransitionValidator()
        s1 = FarmState(0.5, 25.0, 0.3, 0.6, 0.2)
        s2 = FarmState(0.4, 25.0, 0.3, 0.6, 0.2)
        assert validator.validate(s1, s2) is True

    def test_moisture_increase_without_irrigation_flagged(self):
        validator = StateTransitionValidator()
        s1 = FarmState(0.3, 25.0, 0.3, 0.6, 0.2)
        s2 = FarmState(0.6, 25.0, 0.3, 0.6, 0.2)
        # Moisture increased significantly without irrigation event
        assert validator.validate(s1, s2) is False

    def test_moisture_increase_with_irrigation_allowed(self):
        validator = StateTransitionValidator()
        s1 = FarmState(0.3, 25.0, 0.3, 0.6, 0.2)
        s2 = FarmState(0.6, 25.0, 0.3, 0.6, 0.2)
        assert validator.validate(s1, s2, context={"irrigation_event": True}) is True

    def test_crop_height_decrease_flagged(self):
        validator = StateTransitionValidator()
        s1 = FarmState(0.5, 25.0, 0.5, 0.6, 0.2)
        s2 = FarmState(0.5, 25.0, 0.2, 0.6, 0.2)
        # Crop height decreased without harvest/stress event
        assert validator.validate(s1, s2) is False

    def test_crop_height_decrease_with_harvest_allowed(self):
        validator = StateTransitionValidator()
        s1 = FarmState(0.5, 25.0, 0.5, 0.6, 0.2)
        s2 = FarmState(0.5, 25.0, 0.2, 0.6, 0.2)
        assert validator.validate(s1, s2, context={"harvest_event": True}) is True

    def test_crop_height_increase_allowed(self):
        validator = StateTransitionValidator()
        s1 = FarmState(0.5, 25.0, 0.3, 0.6, 0.2)
        s2 = FarmState(0.5, 25.0, 0.5, 0.6, 0.2)
        assert validator.validate(s1, s2) is True

    def test_small_moisture_change_allowed(self):
        validator = StateTransitionValidator()
        s1 = FarmState(0.5, 25.0, 0.3, 0.6, 0.2)
        s2 = FarmState(0.52, 25.0, 0.3, 0.6, 0.2)
        assert validator.validate(s1, s2) is True

    def test_nutrient_decrease_flagged(self):
        validator = StateTransitionValidator()
        s1 = FarmState(0.5, 25.0, 0.3, 0.8, 0.2)
        s2 = FarmState(0.5, 25.0, 0.3, 0.2, 0.2)
        # Nutrient level dropped significantly
        assert validator.validate(s1, s2) is False

    def test_get_violations(self):
        validator = StateTransitionValidator()
        s1 = FarmState(0.3, 25.0, 0.5, 0.8, 0.2)
        s2 = FarmState(0.6, 25.0, 0.2, 0.2, 0.2)
        violations = validator.get_violations(s1, s2)
        assert len(violations) > 0
        assert any("soil_moisture" in v for v in violations)
        assert any("crop_height" in v for v in violations)
        assert any("nutrient_level" in v for v in violations)
