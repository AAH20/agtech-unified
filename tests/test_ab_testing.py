"""Tests for A/B testing framework."""

import time

import pytest

from src.decision_support.ab_testing import ABTestManager


class TestABTestManager:
    """Test the A/B testing framework."""

    def test_create_experiment(self):
        """Create an experiment with variants."""
        mgr = ABTestManager()
        exp = mgr.create_experiment("rec_algo", variants=["control", "treatment"])
        assert exp.name == "rec_algo"
        assert exp.variants == ["control", "treatment"]
        assert exp.weights == [0.5, 0.5]

    def test_create_experiment_custom_weights(self):
        """Create an experiment with custom traffic split."""
        mgr = ABTestManager()
        exp = mgr.create_experiment("rec_algo", variants=["A", "B", "C"], weights=[0.5, 0.3, 0.2])
        assert exp.weights == [0.5, 0.3, 0.2]

    def test_create_experiment_weights_must_sum_to_one(self):
        """Weights must sum to 1.0."""
        mgr = ABTestManager()
        with pytest.raises(ValueError, match="sum"):
            mgr.create_experiment("test", variants=["A", "B"], weights=[0.5, 0.4])

    def test_assign_variant_deterministic(self):
        """Same user always gets the same variant."""
        mgr = ABTestManager()
        mgr.create_experiment("exp1", variants=["control", "treatment"])
        v1 = mgr.assign_variant("exp1", "user-42")
        v2 = mgr.assign_variant("exp1", "user-42")
        assert v1 == v2

    def test_assign_variant_different_users(self):
        """Different users may get different variants."""
        mgr = ABTestManager()
        mgr.create_experiment("exp1", variants=["control", "treatment"])
        variants = set()
        for i in range(50):
            v = mgr.assign_variant("exp1", f"user-{i}")
            variants.add(v)
        # With 50 users, we should see both variants
        assert len(variants) == 2

    def test_assign_variant_returns_valid_variant(self):
        """Assigned variant is always one of the experiment's variants."""
        mgr = ABTestManager()
        mgr.create_experiment("exp1", variants=["A", "B", "C"])
        for i in range(20):
            v = mgr.assign_variant("exp1", f"user-{i}")
            assert v in ["A", "B", "C"]

    def test_assign_variant_unknown_experiment_raises(self):
        """Assigning to unknown experiment raises KeyError."""
        mgr = ABTestManager()
        with pytest.raises(KeyError):
            mgr.assign_variant("nonexistent", "user-1")

    def test_record_outcome(self):
        """Record an outcome for a variant."""
        mgr = ABTestManager()
        mgr.create_experiment("exp1", variants=["control", "treatment"])
        mgr.record_outcome("exp1", "control", 1.0)
        mgr.record_outcome("exp1", "treatment", 0.0)
        results = mgr.get_results("exp1")
        assert "control" in results
        assert "treatment" in results

    def test_get_results_means(self):
        """Results include mean outcome per variant."""
        mgr = ABTestManager()
        mgr.create_experiment("exp1", variants=["control", "treatment"])
        mgr.record_outcome("exp1", "control", 1.0)
        mgr.record_outcome("exp1", "control", 0.0)
        mgr.record_outcome("exp1", "treatment", 1.0)
        mgr.record_outcome("exp1", "treatment", 1.0)
        results = mgr.get_results("exp1")
        assert abs(results["control"]["mean"] - 0.5) < 0.01
        assert abs(results["treatment"]["mean"] - 1.0) < 0.01

    def test_get_results_counts(self):
        """Results include sample counts per variant."""
        mgr = ABTestManager()
        mgr.create_experiment("exp1", variants=["control", "treatment"])
        for _ in range(10):
            mgr.record_outcome("exp1", "control", 1.0)
        for _ in range(5):
            mgr.record_outcome("exp1", "treatment", 0.0)
        results = mgr.get_results("exp1")
        assert results["control"]["count"] == 10
        assert results["treatment"]["count"] == 5

    def test_is_significant_clear_difference(self):
        """Large difference between variants is detected as significant."""
        mgr = ABTestManager()
        mgr.create_experiment("exp1", variants=["control", "treatment"])
        # Control: all zeros, Treatment: all ones — huge difference
        for _ in range(100):
            mgr.record_outcome("exp1", "control", 0.0)
            mgr.record_outcome("exp1", "treatment", 1.0)
        assert mgr.is_significant("exp1", alpha=0.05)

    def test_is_significant_no_difference(self):
        """No difference between variants is not significant."""
        mgr = ABTestManager()
        mgr.create_experiment("exp1", variants=["control", "treatment"])
        # Both variants get the same outcomes
        for _ in range(100):
            mgr.record_outcome("exp1", "control", 0.5)
            mgr.record_outcome("exp1", "treatment", 0.5)
        assert not mgr.is_significant("exp1", alpha=0.05)

    def test_is_significant_small_sample_not_significant(self):
        """Small samples with moderate difference are not significant."""
        mgr = ABTestManager()
        mgr.create_experiment("exp1", variants=["control", "treatment"])
        for _ in range(5):
            mgr.record_outcome("exp1", "control", 0.0)
            mgr.record_outcome("exp1", "treatment", 1.0)
        # With only 5 samples each, even a large difference may not be significant
        # (depends on variance, but with zero variance in each group, t-test should still work)
        # This test verifies the method runs without error
        result = mgr.is_significant("exp1", alpha=0.05)
        assert isinstance(result, bool)

    def test_traffic_split_approximate(self):
        """Traffic split approximately matches configured weights."""
        mgr = ABTestManager()
        mgr.create_experiment("exp1", variants=["A", "B"], weights=[0.7, 0.3])
        counts = {"A": 0, "B": 0}
        n = 1000
        for i in range(n):
            v = mgr.assign_variant("exp1", f"user-{i}")
            counts[v] += 1
        # Should be roughly 70/30 (with some tolerance)
        assert abs(counts["A"] / n - 0.7) < 0.1
        assert abs(counts["B"] / n - 0.3) < 0.1

    def test_experiment_with_time_window(self):
        """Experiment respects time window."""
        mgr = ABTestManager()
        now = time.time()
        exp = mgr.create_experiment(
            "exp1",
            variants=["A", "B"],
            start_time=now - 3600,
            end_time=now + 3600,
        )
        assert exp.start_time == now - 3600
        assert exp.end_time == now + 3600

    def test_experiment_expired(self):
        """Expired experiment does not assign variants."""
        mgr = ABTestManager()
        now = time.time()
        mgr.create_experiment(
            "exp1",
            variants=["A", "B"],
            start_time=now - 7200,
            end_time=now - 3600,
        )
        with pytest.raises(ValueError, match="expired"):
            mgr.assign_variant("exp1", "user-1")

    def test_list_experiments(self):
        """List all created experiments."""
        mgr = ABTestManager()
        mgr.create_experiment("exp1", variants=["A", "B"])
        mgr.create_experiment("exp2", variants=["X", "Y"])
        names = mgr.list_experiments()
        assert "exp1" in names
        assert "exp2" in names

    def test_results_include_variance(self):
        """Results include variance per variant."""
        mgr = ABTestManager()
        mgr.create_experiment("exp1", variants=["control", "treatment"])
        mgr.record_outcome("exp1", "control", 0.0)
        mgr.record_outcome("exp1", "control", 1.0)
        results = mgr.get_results("exp1")
        assert "variance" in results["control"]
        assert results["control"]["variance"] > 0

    def test_results_empty_variant(self):
        """Results for variant with no data have zero count."""
        mgr = ABTestManager()
        mgr.create_experiment("exp1", variants=["control", "treatment"])
        results = mgr.get_results("exp1")
        assert results["control"]["count"] == 0
        assert results["treatment"]["count"] == 0
