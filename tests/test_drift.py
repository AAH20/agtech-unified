"""Tests for drift detection in decision support."""

import pytest

from src.decision_support.drift import DriftDetector


class TestDriftDetector:
    """Test the drift detection framework."""

    def test_create_detector(self):
        """Create a drift detector with reference data."""
        ref = {"soil_moisture": [0.3, 0.4, 0.5, 0.6, 0.7]}
        det = DriftDetector(ref)
        assert det.reference_data is not None

    def test_no_drift_similar_distribution(self):
        """Similar distribution produces low PSI."""
        ref = [0.3, 0.4, 0.5, 0.6, 0.7, 0.35, 0.45, 0.55, 0.65, 0.75]
        det = DriftDetector({"soil_moisture": ref})
        current = [0.32, 0.42, 0.52, 0.62, 0.72, 0.37, 0.47, 0.57, 0.67, 0.77]
        psi = det.compute_psi("soil_moisture", current)
        assert psi < 0.1  # Very low PSI for similar distributions

    def test_drift_detected_different_distribution(self):
        """Very different distribution produces high PSI."""
        ref = [0.1, 0.15, 0.2, 0.25, 0.3, 0.12, 0.18, 0.22, 0.28, 0.35]
        det = DriftDetector({"soil_moisture": ref})
        current = [0.8, 0.85, 0.9, 0.95, 0.99, 0.82, 0.88, 0.92, 0.97, 0.93]
        psi = det.compute_psi("soil_moisture", current)
        assert psi > 0.5  # High PSI for very different distributions

    def test_psi_symmetric(self):
        """PSI is symmetric: PSI(a,b) == PSI(b,a)."""
        ref = [0.1, 0.2, 0.3, 0.4, 0.5]
        current = [0.6, 0.7, 0.8, 0.9, 1.0]
        det = DriftDetector({"f": ref})
        psi1 = det.compute_psi("f", current)
        det2 = DriftDetector({"f": current})
        psi2 = det2.compute_psi("f", ref)
        assert abs(psi1 - psi2) < 0.01

    def test_ks_statistic_identical(self):
        """KS statistic is 0 for identical distributions."""
        data = [0.1, 0.2, 0.3, 0.4, 0.5]
        det = DriftDetector({"f": data})
        ks = det.compute_ks_statistic("f", data)
        assert abs(ks) < 0.01

    def test_ks_statistic_different(self):
        """KS statistic is large for very different distributions."""
        ref = [0.1, 0.2, 0.3, 0.4, 0.5]
        current = [0.6, 0.7, 0.8, 0.9, 1.0]
        det = DriftDetector({"f": ref})
        ks = det.compute_ks_statistic("f", current)
        assert ks > 0.5

    def test_detect_drift_all_features(self):
        """detect_drift returns drift status for all features."""
        ref = {
            "soil_moisture": [0.3, 0.4, 0.5, 0.6, 0.7],
            "temperature": [20.0, 22.0, 24.0, 26.0, 28.0],
        }
        det = DriftDetector(ref)
        current = {
            "soil_moisture": [0.32, 0.42, 0.52, 0.62, 0.72],
            "temperature": [35.0, 37.0, 39.0, 41.0, 43.0],
        }
        result = det.detect_drift(current, psi_threshold=0.2)
        assert "soil_moisture" in result
        assert "temperature" in result
        # soil_moisture similar → no drift; temperature very different → drift
        assert result["soil_moisture"] is False
        assert result["temperature"] is True

    def test_detect_drift_with_custom_threshold(self):
        """Custom PSI threshold affects drift detection."""
        ref = [0.3, 0.4, 0.5, 0.6, 0.7]
        det = DriftDetector({"f": ref})
        current = [0.35, 0.45, 0.55, 0.65, 0.75]
        # With very low threshold, even small differences trigger drift
        result_low = det.detect_drift({"f": current}, psi_threshold=0.01)
        # With very high threshold, nothing triggers drift
        result_high = det.detect_drift({"f": current}, psi_threshold=10.0)
        assert result_low["f"] is True
        assert result_high["f"] is False

    def test_drift_detector_multiple_features(self):
        """DriftDetector handles multiple features independently."""
        ref = {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 1.2],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0, 15.0, 25.0, 35.0, 45.0, 12.0],
            "c": [0.1, 0.2, 0.3, 0.4, 0.5, 0.15, 0.25, 0.35, 0.45, 0.12],
        }
        det = DriftDetector(ref)
        assert det.compute_psi("a", [1.1, 2.1, 3.1, 4.1, 5.1, 1.6, 2.6, 3.6, 4.6, 1.3]) < 0.1
        assert (
            det.compute_psi(
                "b", [100.0, 200.0, 300.0, 400.0, 500.0, 150.0, 250.0, 350.0, 450.0, 120.0]
            )
            > 0.5
        )
        assert (
            det.compute_psi("c", [0.11, 0.21, 0.31, 0.41, 0.51, 0.16, 0.26, 0.36, 0.46, 0.13]) < 0.1
        )

    def test_psi_zero_for_identical(self):
        """PSI is 0 for identical distributions."""
        data = [0.1, 0.2, 0.3, 0.4, 0.5]
        det = DriftDetector({"f": data})
        psi = det.compute_psi("f", data)
        assert abs(psi) < 0.01

    def test_psi_non_negative(self):
        """PSI is always non-negative."""
        ref = [0.1, 0.2, 0.3]
        current = [0.7, 0.8, 0.9]
        det = DriftDetector({"f": ref})
        psi = det.compute_psi("f", current)
        assert psi >= 0.0

    def test_ks_statistic_range(self):
        """KS statistic is in [0, 1]."""
        ref = [0.1, 0.2, 0.3, 0.4, 0.5]
        det = DriftDetector({"f": ref})
        for current in [[0.1, 0.2, 0.3], [0.6, 0.7, 0.8], [0.3, 0.3, 0.3]]:
            ks = det.compute_ks_statistic("f", current)
            assert 0.0 <= ks <= 1.0

    def test_detect_drift_returns_dict_of_bools(self):
        """detect_drift returns a dict mapping feature names to bools."""
        ref = {"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}
        det = DriftDetector(ref)
        current = {"x": [1.1, 2.1, 3.1], "y": [40.0, 50.0, 60.0]}
        result = det.detect_drift(current)
        assert isinstance(result, dict)
        for key, val in result.items():
            assert isinstance(key, str)
            assert isinstance(val, bool)

    def test_drift_detector_with_empty_current(self):
        """Empty current data raises ValueError."""
        det = DriftDetector({"f": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError):
            det.compute_psi("f", [])

    def test_drift_detector_unknown_feature(self):
        """Unknown feature name raises KeyError."""
        det = DriftDetector({"f": [1.0, 2.0, 3.0]})
        with pytest.raises(KeyError):
            det.compute_psi("nonexistent", [1.0, 2.0])
