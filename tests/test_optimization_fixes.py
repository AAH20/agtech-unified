"""Tests for the three optimization fixes:
1. predict_yield batch validation handles partial batches
2. analyze_field uses highest-severity detection
3. GPUFallback only catches GPU-related exceptions
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.optimization.crop_vision import (
    CropVisionPipeline,
    DiseaseDetection,
    FieldHealthReport,
)
from src.optimization.gpu_fallback import GPUFallback


# --------------------------------------------------------------------------- #
# Fix 1: predict_yield batch validation
# --------------------------------------------------------------------------- #
class TestPredictYieldBatchValidation:
    """predict_yield should accept partial batches and reject invalid sizes."""

    def test_partial_batch_accepted(self):
        """A chunk smaller than batch_size should be accepted."""
        pipeline = CropVisionPipeline.__new__(CropVisionPipeline)
        pipeline.batch_size = 4
        pipeline._yield = MagicMock()

        # Simulate: 5 images, batch_size=4 → chunks of 4 and 1
        # The last chunk has 1 image, model returns 1 value
        def mock_predict(images, batch_size=1):
            n = len(images)
            return [MagicMock(outputs={"Y": np.array([5.0] * n)})]

        pipeline._yield.predict = mock_predict
        pipeline._primary_output = lambda res: res.outputs["Y"]

        images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(5)]
        result = pipeline.predict_yield(images)
        assert result.estimated_yield_t_per_ha == pytest.approx(5.0)
        assert len(result.per_image_yields) == 5

    def test_empty_output_rejected(self):
        """Model returning zero values should raise ValueError."""
        pipeline = CropVisionPipeline.__new__(CropVisionPipeline)
        pipeline.batch_size = 4
        pipeline._yield = MagicMock()

        def mock_predict(images, batch_size=1):
            return [MagicMock(outputs={"Y": np.array([])})]

        pipeline._yield.predict = mock_predict
        pipeline._primary_output = lambda res: res.outputs["Y"]

        images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
        with pytest.raises(ValueError, match="Yield model returned"):
            pipeline.predict_yield(images)

    def test_oversized_output_rejected(self):
        """Model returning more values than images should raise ValueError."""
        pipeline = CropVisionPipeline.__new__(CropVisionPipeline)
        pipeline.batch_size = 4
        pipeline._yield = MagicMock()

        def mock_predict(images, batch_size=1):
            # Return 10 values for 3 images — clearly wrong
            return [MagicMock(outputs={"Y": np.array([5.0] * 10)})]

        pipeline._yield.predict = mock_predict
        pipeline._primary_output = lambda res: res.outputs["Y"]

        images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
        with pytest.raises(ValueError, match="Yield model returned"):
            pipeline.predict_yield(images)


# --------------------------------------------------------------------------- #
# Fix 2: analyze_field uses highest-severity detection
# --------------------------------------------------------------------------- #
class TestAnalyzeFieldHighestSeverity:
    """analyze_field should return the detection with the highest severity."""

    def test_highest_severity_detection_selected(self):
        """When multiple images have different severities, the worst is reported."""
        pipeline = CropVisionPipeline.__new__(CropVisionPipeline)
        pipeline._disease = MagicMock()
        pipeline._weed = None
        pipeline._yield = None

        detections = [
            DiseaseDetection("healthy", 0.9, 0.1),
            DiseaseDetection("leaf_blight", 0.8, 0.7),
            DiseaseDetection("rust", 0.6, 0.4),
        ]
        pipeline.detect_disease = MagicMock(return_value=detections)

        images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
        report = pipeline.analyze_field(images)

        assert isinstance(report, FieldHealthReport)
        assert report.disease.label == "leaf_blight"
        assert report.disease.severity == pytest.approx(0.7)

    def test_single_detection_returned(self):
        """With a single image, that detection is returned directly."""
        pipeline = CropVisionPipeline.__new__(CropVisionPipeline)
        pipeline._disease = MagicMock()
        pipeline._weed = None
        pipeline._yield = None

        detections = [DiseaseDetection("healthy", 0.95, 0.05)]
        pipeline.detect_disease = MagicMock(return_value=detections)

        images = [np.zeros((8, 8, 3), dtype=np.uint8)]
        report = pipeline.analyze_field(images)

        assert report.disease.label == "healthy"
        assert report.disease.severity == pytest.approx(0.05)


# --------------------------------------------------------------------------- #
# Fix 3: GPUFallback exception handling
# --------------------------------------------------------------------------- #
class TestGPUFallbackExceptionHandling:
    """GPUFallback should only catch GPU-related exceptions."""

    def test_non_gpu_exception_not_caught(self):
        """ValueError from GPU solver should propagate, not trigger fallback."""
        fb = GPUFallback()

        def gpu_fail(p):
            raise ValueError("some non-GPU error")

        def cpu_ok(p):
            return "cpu_result"

        with pytest.raises(ValueError, match="some non-GPU error"):
            fb.solve("prob", gpu_fail, cpu_ok)

    def test_cuda_error_triggers_fallback(self):
        """RuntimeError with CUDA message should trigger CPU fallback."""
        fb = GPUFallback()

        def gpu_fail(p):
            raise RuntimeError("CUDA out of memory")

        def cpu_ok(p):
            return "cpu_result"

        result = fb.solve("prob", gpu_fail, cpu_ok)
        assert result == "cpu_result"

    def test_import_error_triggers_fallback(self):
        """ImportError (e.g., CUDA library missing) should trigger fallback."""
        fb = GPUFallback()

        def gpu_fail(p):
            raise ImportError("libcudart.so not found")

        def cpu_ok(p):
            return 42

        result = fb.solve("prob", gpu_fail, cpu_ok)
        assert result == 42
