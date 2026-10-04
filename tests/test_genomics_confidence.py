"""Test confidence intervals for prediction scores."""

import pytest

from src.genomics.confidence import (
    ConfidenceInterval,
    add_confidence_interval,
    bootstrap_confidence_interval,
    wilson_score_interval,
)


class TestWilsonScoreInterval:
    """Tests for Wilson score confidence interval."""

    def test_basic_interval(self):
        """Wilson interval for 50/100 successes."""
        ci = wilson_score_interval(50, 100)
        assert ci.lower < ci.upper
        assert 0.0 <= ci.lower <= 1.0
        assert 0.0 <= ci.upper <= 1.0

    def test_perfect_success(self):
        """All successes gives upper bound near 1."""
        ci = wilson_score_interval(100, 100)
        assert ci.lower > 0.9
        assert ci.upper <= 1.0

    def test_zero_success(self):
        """No successes gives lower bound of 0."""
        ci = wilson_score_interval(0, 100)
        assert ci.lower == 0.0
        assert ci.upper < 0.1

    def test_symmetry(self):
        """Wilson interval is approximately symmetric for p=0.5."""
        ci = wilson_score_interval(50, 100)
        mid = (ci.lower + ci.upper) / 2
        assert mid == pytest.approx(0.5, abs=0.05)

    def test_n_zero_raises(self):
        """Zero trials raises ValueError."""
        with pytest.raises(ValueError, match="n must be positive"):
            wilson_score_interval(0, 0)

    def test_successes_exceed_trials_raises(self):
        """More successes than trials raises ValueError."""
        with pytest.raises(ValueError, match="k cannot exceed n"):
            wilson_score_interval(101, 100)

    def test_custom_confidence(self):
        """Custom confidence level works."""
        ci_95 = wilson_score_interval(50, 100, confidence=0.95)
        ci_99 = wilson_score_interval(50, 100, confidence=0.99)
        # 99% CI should be wider
        width_95 = ci_95.upper - ci_95.lower
        width_99 = ci_99.upper - ci_99.lower
        assert width_99 > width_95


class TestBootstrapConfidenceInterval:
    """Tests for bootstrap confidence interval."""

    def test_basic_interval(self):
        """Bootstrap interval for sample data."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = bootstrap_confidence_interval(data)
        assert ci.lower <= ci.upper

    def test_interval_contains_mean(self):
        """Bootstrap interval approximately contains the mean."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = bootstrap_confidence_interval(data, n_bootstrap=500)
        mean = sum(data) / len(data)
        assert ci.lower <= mean <= ci.upper

    def test_empty_data_raises(self):
        """Empty data raises ValueError."""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            bootstrap_confidence_interval([])

    def test_single_value(self):
        """Single value gives zero-width interval."""
        ci = bootstrap_confidence_interval([5.0])
        assert ci.lower == 5.0
        assert ci.upper == 5.0

    def test_custom_n_bootstrap(self):
        """Custom number of bootstrap samples works."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = bootstrap_confidence_interval(data, n_bootstrap=100)
        assert ci.lower <= ci.upper

    def test_custom_confidence(self):
        """Custom confidence level works."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci_95 = bootstrap_confidence_interval(data, confidence=0.95, n_bootstrap=500)
        ci_99 = bootstrap_confidence_interval(data, confidence=0.99, n_bootstrap=500)
        width_95 = ci_95.upper - ci_95.lower
        width_99 = ci_99.upper - ci_99.lower
        assert width_99 > width_95


class TestAddConfidenceInterval:
    """Tests for adding CI to prediction scores."""

    def test_adds_ci_fields(self):
        """add_confidence_interval adds lower and upper bounds."""
        ci = add_confidence_interval(0.7, 100, 70)
        assert hasattr(ci, "lower")
        assert hasattr(ci, "upper")
        assert hasattr(ci, "score")
        assert ci.lower <= ci.score <= ci.upper

    def test_score_preserved(self):
        """Original score is preserved."""
        ci = add_confidence_interval(0.7, 100, 70)
        assert ci.score == pytest.approx(0.7)

    def test_n_zero_raises(self):
        """Zero trials raises ValueError."""
        with pytest.raises(ValueError, match="n must be positive"):
            add_confidence_interval(0.5, 0, 0)


class TestConfidenceIntervalDataclass:
    """Tests for ConfidenceInterval dataclass."""

    def test_creation(self):
        """ConfidenceInterval can be created."""
        ci = ConfidenceInterval(lower=0.1, upper=0.9, confidence=0.95)
        assert ci.lower == 0.1
        assert ci.upper == 0.9
        assert ci.confidence == 0.95

    def test_width(self):
        """Width is upper minus lower."""
        ci = ConfidenceInterval(lower=0.1, upper=0.9, confidence=0.95)
        assert ci.width == pytest.approx(0.8)

    def test_contains(self):
        """Contains checks if value is in interval."""
        ci = ConfidenceInterval(lower=0.1, upper=0.9, confidence=0.95)
        assert ci.contains(0.5)
        assert not ci.contains(0.05)
        assert not ci.contains(0.95)

    def test_str_representation(self):
        """String representation includes bounds."""
        ci = ConfidenceInterval(lower=0.1, upper=0.9, confidence=0.95)
        s = str(ci)
        assert "0.1" in s
        assert "0.9" in s
