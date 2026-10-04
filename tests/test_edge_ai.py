"""Tests for EdgeAIInference — ONNX Runtime model loading, preprocessing, inference."""
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper, numpy_helper

from src.optimization.edge_ai import EdgeAIInference


def _make_add_model(path, input_shape=("N", 3, 8, 8), output_shape=("N", 3, 8, 8)):
    """Create a minimal ONNX model: output = input + 1 (dynamic batch)."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)
    add_node = helper.make_node("Add", ["X", "one"], ["Y"])
    one = numpy_helper.from_array(np.ones((3, 8, 8), dtype=np.float32), "one")
    graph = helper.make_graph([add_node], "test", [X], [Y], [one])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


def _make_identity_model(path, input_shape=(1, 3, 8, 8)):
    """Create a minimal ONNX model: output = input (identity)."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, input_shape)
    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph([node], "test", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


@pytest.fixture
def add_model(tmp_path):
    p = tmp_path / "add.onnx"
    _make_add_model(p)
    return str(p)


@pytest.fixture
def identity_model(tmp_path):
    p = tmp_path / "identity.onnx"
    _make_identity_model(p)
    return str(p)


# --------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------- #
def test_load_model_returns_info(add_model):
    """load_model returns ModelInfo with correct input/output names."""
    engine = EdgeAIInference()
    info = engine.load_model(add_model)
    assert info.input_names == ["X"]
    assert info.output_names == ["Y"]
    assert info.input_shapes["X"] == ["N", 3, 8, 8]
    assert info.output_shapes["Y"] == ["N", 3, 8, 8]
    assert len(info.providers) >= 1


def test_load_model_missing_file_raises(tmp_path):
    """Loading a non-existent model path raises FileNotFoundError."""
    engine = EdgeAIInference()
    with pytest.raises(FileNotFoundError):
        engine.load_model(str(tmp_path / "nope.onnx"))


def test_load_model_wrong_extension_raises(tmp_path):
    """Loading a non-.onnx file raises ValueError."""
    p = tmp_path / "model.pt"
    p.write_bytes(b"not onnx")
    engine = EdgeAIInference()
    with pytest.raises(ValueError, match="onnx"):
        engine.load_model(str(p))


def test_constructor_with_model_path(add_model):
    """Passing model_path to the constructor loads the model immediately."""
    engine = EdgeAIInference(add_model)
    assert engine.is_loaded
    assert engine.model_info.input_names == ["X"]


# --------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------- #
def test_preprocess_output_shape(identity_model):
    """preprocess returns NCHW float32 tensor with correct shape."""
    engine = EdgeAIInference(identity_model)
    images = [np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8) for _ in range(3)]
    tensor = engine.preprocess(images, input_shape=(3, 3, 8, 8))
    assert tensor.shape == (3, 3, 8, 8)
    assert tensor.dtype == np.float32


def test_preprocess_empty_raises(identity_model):
    """preprocess with no images raises ValueError."""
    engine = EdgeAIInference(identity_model)
    with pytest.raises(ValueError, match="empty"):
        engine.preprocess([])


def test_preprocess_grayscale_expands(identity_model):
    """Grayscale (2D) images are expanded to 3 channels."""
    engine = EdgeAIInference(identity_model)
    images = [np.random.randint(0, 256, (8, 8), dtype=np.uint8)]
    tensor = engine.preprocess(images, input_shape=(1, 3, 8, 8))
    assert tensor.shape == (1, 3, 8, 8)


# --------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------- #
def test_infer_add_model(add_model):
    """Inference on the add model returns input + 1."""
    engine = EdgeAIInference(add_model)
    tensor = np.ones((1, 3, 8, 8), dtype=np.float32)
    result = engine.infer(tensor)
    assert "Y" in result.outputs
    np.testing.assert_allclose(result.outputs["Y"], 2.0, atol=1e-5)
    assert result.latency_ms > 0


def test_infer_batch_chunks(add_model):
    """infer_batch splits tensors into chunks of batch_size."""
    engine = EdgeAIInference(add_model)
    tensors = [np.full((3, 8, 8), float(i), dtype=np.float32) for i in range(5)]
    results = engine.infer_batch(tensors, batch_size=2)
    # 5 tensors / batch_size 2 → 3 chunks
    assert len(results) == 3
    # First chunk: tensors 0,1 → outputs 1,2
    np.testing.assert_allclose(results[0].outputs["Y"][0], 1.0, atol=1e-5)
    np.testing.assert_allclose(results[0].outputs["Y"][1], 2.0, atol=1e-5)


def test_infer_batch_empty(add_model):
    """infer_batch with no tensors returns empty list."""
    engine = EdgeAIInference(add_model)
    assert engine.infer_batch([]) == []


def test_predict_end_to_end(identity_model):
    """predict preprocesses and infers in one call."""
    engine = EdgeAIInference(identity_model)
    images = [np.zeros((8, 8, 3), dtype=np.uint8)]
    results = engine.predict(images, batch_size=1)
    assert len(results) == 1
    # Identity model: output ≈ input (normalized zeros → -mean/std)
    assert results[0].outputs["Y"].shape == (1, 3, 8, 8)


def test_benchmark_returns_stats(identity_model):
    """benchmark runs warmup + iterations and returns BenchmarkStats."""
    engine = EdgeAIInference(identity_model)
    images = [np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)]
    stats = engine.benchmark(images, warmup=1, iterations=3, batch_size=1)
    assert stats.iterations == 3
    assert stats.total_ms > 0
    assert stats.mean_ms > 0
    assert stats.min_ms <= stats.p50_ms <= stats.max_ms
    assert stats.p95_ms >= stats.p50_ms


def test_infer_without_model_raises():
    """Calling infer without loading a model raises RuntimeError."""
    engine = EdgeAIInference()
    with pytest.raises(RuntimeError, match="load_model"):
        engine.infer(np.zeros((1, 3, 8, 8), dtype=np.float32))


def test_release_clears_session(add_model):
    """release() unloads the model; subsequent calls raise."""
    engine = EdgeAIInference(add_model)
    assert engine.is_loaded
    engine.release()
    assert not engine.is_loaded
    with pytest.raises(RuntimeError):
        engine.model_info
