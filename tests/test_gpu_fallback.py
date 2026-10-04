"""Tests for GPU fallback mechanism."""

import pytest

from src.optimization.gpu_fallback import GPUFallback


def test_gpu_success():
    """GPU solver succeeds and its result is returned."""
    fb = GPUFallback()
    result = fb.solve("prob", lambda p: "gpu_result", lambda p: "cpu_result")
    assert result == "gpu_result"


def test_gpu_failure_falls_back_to_cpu():
    """When GPU fails, CPU solver is used."""
    fb = GPUFallback()

    def gpu_fail(p):
        raise RuntimeError("CUDA OOM")

    result = fb.solve("prob", gpu_fail, lambda p: "cpu_result")
    assert result == "cpu_result"


def test_problem_passed_to_both():
    """The problem argument is passed through to whichever solver runs."""
    fb = GPUFallback()
    seen = []

    def gpu_fail(p):
        seen.append(p)
        raise RuntimeError("fail")

    def cpu_ok(p):
        seen.append(p)
        return "ok"

    fb.solve("my_problem", gpu_fail, cpu_ok)
    assert seen == ["my_problem", "my_problem"]


def test_both_fail_raises():
    """If both solvers fail, the CPU exception propagates."""
    fb = GPUFallback()

    def gpu_fail(p):
        raise RuntimeError("GPU error")

    def cpu_fail(p):
        raise ValueError("CPU error")

    with pytest.raises(ValueError, match="CPU error"):
        fb.solve("prob", gpu_fail, cpu_fail)


def test_cpu_result_returned_on_fallback():
    """CPU solver result is returned when GPU raises."""
    fb = GPUFallback()

    def gpu_fail(p):
        raise Exception("CUDA not available")

    def cpu_ok(p):
        return 42

    assert fb.solve("prob", gpu_fail, cpu_ok) == 42
