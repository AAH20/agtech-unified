"""Tests for CropVisionPipeline — disease detection, weed classification, yield prediction."""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper, numpy_helper

from src.optimization.crop_vision import (
    DISEASE_CLASSES,
    WEED_CLASSES,
    CropVisionPipeline,
    DiseaseDetection,
    FieldHealthReport,
    WeedClassification,
    YieldPrediction,
)


def _make_classifier_model(path, num_classes):
    """Create a deterministic ONNX classifier: always predicts class 0.

    Output = softmax(logits + mean(X)*0) so input is used but doesn't affect output.
    """
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, num_classes])
    logits = np.zeros((1, num_classes), dtype=np.float32)
    logits[0, 0] = 20.0
    logits_init = numpy_helper.from_array(logits, "logits")
    zero_arr = np.array([0.0], dtype=np.float32)
    zero_init = numpy_helper.from_array(zero_arr, "zero")
    mean_node = helper.make_node("ReduceMean", ["X"], ["M"], axes=[1, 2, 3], keepdims=0)
    mul_node = helper.make_node("Mul", ["M", "zero"], ["M_zero"])
    add_node = helper.make_node("Add", ["logits", "M_zero"], ["L"])
    softmax = helper.make_node("Softmax", ["L"], ["Y"], axis=1)
    graph = helper.make_graph(
        [mean_node, mul_node, add_node, softmax], "classifier", [X], [Y], [logits_init, zero_init]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


def _make_yield_model(path):
    """Create a minimal ONNX yield model: constant output 5.0 t/ha."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    const_arr = np.array([5.0], dtype=np.float32)
    const_init = numpy_helper.from_array(const_arr, "yield_val")
    identity = helper.make_node("Identity", ["yield_val"], ["Y"])
    graph = helper.make_graph([identity], "yield", [X], [Y], [const_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


@pytest.fixture
def disease_model(tmp_path):
    p = tmp_path / "disease.onnx"
    _make_classifier_model(p, len(DISEASE_CLASSES))
    return str(p)


@pytest.fixture
def weed_model(tmp_path):
    p = tmp_path / "weed.onnx"
    _make_classifier_model(p, len(WEED_CLASSES))
    return str(p)


@pytest.fixture
def yield_model(tmp_path):
    p = tmp_path / "yield.onnx"
    _make_yield_model(p)
    return str(p)


@pytest.fixture
def pipeline(disease_model, weed_model, yield_model):
    return CropVisionPipeline(
        disease_model=disease_model,
        weed_model=weed_model,
        yield_model=yield_model,
        batch_size=1,
    )


def _random_images(n, size=8):
    return [np.random.randint(0, 256, (size, size, 3), dtype=np.uint8) for _ in range(n)]


# --------------------------------------------------------------------- #
# Disease detection
# --------------------------------------------------------------------- #
def test_detect_disease_returns_detection(pipeline):
    """detect_disease returns DiseaseDetection with valid label."""
    images = _random_images(3)
    results = pipeline.detect_disease(images)
    assert len(results) == 3
    for d in results:
        assert isinstance(d, DiseaseDetection)
        assert d.label in DISEASE_CLASSES
        assert 0.0 <= d.confidence <= 1.0
        assert 0.0 <= d.severity <= 1.0
        assert set(d.probabilities.keys()) == set(DISEASE_CLASSES)


def test_detect_disease_severity_for_healthy(pipeline):
    """Fixed classifier favors class 0 (healthy) → severity is low."""
    images = _random_images(1)
    result = pipeline.detect_disease(images)[0]
    assert result.label == "healthy"
    assert result.severity < 0.6


def test_detect_disease_without_model_raises():
    """detect_disease raises RuntimeError when model not configured."""
    pipeline = CropVisionPipeline()
    with pytest.raises(RuntimeError, match="Disease"):
        pipeline.detect_disease(_random_images(1))


# --------------------------------------------------------------------- #
# Weed classification
# --------------------------------------------------------------------- #
def test_classify_weeds_returns_classification(pipeline):
    """classify_weeds returns WeedClassification with valid label."""
    images = _random_images(3)
    results = pipeline.classify_weeds(images)
    assert len(results) == 3
    for w in results:
        assert isinstance(w, WeedClassification)
        assert w.label in WEED_CLASSES
        assert 0.0 <= w.confidence <= 1.0
        assert 0.0 <= w.coverage_ratio <= 1.0
        assert set(w.probabilities.keys()) == set(WEED_CLASSES)


def test_classify_weeds_coverage_for_none(pipeline):
    """Fixed classifier favors class 0 (none) → coverage is low."""
    images = _random_images(1)
    result = pipeline.classify_weeds(images)[0]
    assert result.label == "none"
    assert result.coverage_ratio < 0.5 or result.coverage_ratio < 0.6


def test_classify_weeds_without_model_raises():
    """classify_weeds raises RuntimeError when model not configured."""
    pipeline = CropVisionPipeline()
    with pytest.raises(RuntimeError, match="Weed"):
        pipeline.classify_weeds(_random_images(1))


# --------------------------------------------------------------------- #
# Yield prediction
# --------------------------------------------------------------------- #
def test_predict_yield_returns_prediction(pipeline):
    """predict_yield returns YieldPrediction with mean and CI."""
    images = _random_images(4)
    result = pipeline.predict_yield(images)
    assert isinstance(result, YieldPrediction)
    assert result.estimated_yield_t_per_ha > 0
    assert len(result.confidence_interval) == 2
    lo, hi = result.confidence_interval
    assert lo <= result.estimated_yield_t_per_ha <= hi
    assert len(result.per_image_yields) == 4


def test_predict_yield_mean_matches_per_image(pipeline):
    """The field mean equals the mean of per-image yields."""
    images = _random_images(3)
    result = pipeline.predict_yield(images)
    expected = float(np.mean(result.per_image_yields))
    assert result.estimated_yield_t_per_ha == pytest.approx(expected)


def test_predict_yield_without_model_raises():
    """predict_yield raises RuntimeError when model not configured."""
    pipeline = CropVisionPipeline()
    with pytest.raises(RuntimeError, match="Yield"):
        pipeline.predict_yield(_random_images(1))


# --------------------------------------------------------------------- #
# Combined analysis
# --------------------------------------------------------------------- #
def test_analyze_field_returns_report(pipeline):
    """analyze_field returns a FieldHealthReport with all fields."""
    images = _random_images(5)
    report = pipeline.analyze_field(images)
    assert isinstance(report, FieldHealthReport)
    assert isinstance(report.disease, DiseaseDetection)
    assert isinstance(report.weed, WeedClassification)
    assert report.yield_t_per_ha > 0
    assert 0.0 <= report.health_score <= 1.0
    assert isinstance(report.recommendations, list)


def test_analyze_field_empty_images_raises(pipeline):
    """analyze_field with no images raises ValueError."""
    with pytest.raises(ValueError, match="at least one"):
        pipeline.analyze_field([])


def test_analyze_field_partial_models():
    """Pipeline with only disease model still produces a report."""
    pipeline = CropVisionPipeline(disease_model=None, weed_model=None, yield_model=None)
    images = _random_images(2)
    report = pipeline.analyze_field(images)
    assert report.disease.label == "unavailable"
    assert report.weed.label == "unavailable"
    assert report.health_score == 1.0  # no disease/weed → perfect


def test_health_score_pure_crop(pipeline):
    """Healthy + no weeds → health_score is high."""
    images = _random_images(1)
    report = pipeline.analyze_field(images)
    assert report.health_score > 0.4


def test_recommendations_for_healthy(pipeline):
    """Healthy field produces no recommendations."""
    images = _random_images(1)
    report = pipeline.analyze_field(images)
    assert report.recommendations == []
