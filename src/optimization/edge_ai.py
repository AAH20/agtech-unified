"""Edge AI inference engine backed by ONNX Runtime.

Provides model loading, image preprocessing, chunked batch inference,
and latency benchmarking for on-device agricultural vision workloads
(drone imagery tiles, crop/weed classification, yield regression).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:  # onnxruntime is an optional heavy dependency; import lazily-safe
    import onnxruntime as ort
except ImportError:  # pragma: no cover - exercised only without the package
    ort = None  # type: ignore[assignment]


@dataclass
class InferenceResult:
    """Output tensor bundle from one inference call."""

    outputs: Dict[str, np.ndarray]
    latency_ms: float


@dataclass
class BenchmarkStats:
    """Latency statistics from a benchmark run."""

    iterations: int
    total_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float


@dataclass
class ModelInfo:
    """Static description of a loaded ONNX model's interface."""

    input_names: List[str]
    output_names: List[str]
    input_shapes: Dict[str, List[Optional[int]]]
    output_shapes: Dict[str, List[Optional[int]]]
    providers: List[str]


class EdgeAIInference:
    """ONNX Runtime inference session wrapper for edge deployment.

    Usage::

        engine = EdgeAIInference(intra_op_threads=2)
        engine.load_model("model.onnx")
        result = engine.predict(images)          # preprocess + infer
        stats = engine.benchmark(images)         # latency profile
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        providers: Optional[Sequence[str]] = None,
        intra_op_threads: int = 1,
        inter_op_threads: int = 1,
        optimization_level: str = "all",
    ) -> None:
        if ort is None:
            raise RuntimeError(
                "onnxruntime is required for EdgeAIInference; "
                "install it with `pip install onnxruntime`"
            )
        assert ort is not None  # narrowed for type checkers
        self._session: Optional[ort.InferenceSession] = None
        self._model_path: Optional[str] = None
        self._providers = list(providers) if providers else None
        self._intra_op_threads = intra_op_threads
        self._inter_op_threads = inter_op_threads
        self._optimization_level = optimization_level
        self._input_name: Optional[str] = None
        self._output_names: List[str] = []
        if model_path is not None:
            self.load_model(model_path)

    # ------------------------------------------------------------------ #
    # Model management
    # ------------------------------------------------------------------ #
    def load_model(self, model_path: str) -> ModelInfo:
        """Load an ONNX model from *model_path* and validate its interface."""
        if ort is None:
            raise RuntimeError(
                "onnxruntime is required for EdgeAIInference; "
                "install it with `pip install onnxruntime`"
            )
        assert ort is not None  # narrowed for type checkers
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        if not path.suffix == ".onnx":
            raise ValueError(f"Model file must have .onnx extension: {path}")

        so = ort.SessionOptions()
        so.intra_op_num_threads = self._intra_op_threads
        so.inter_op_num_threads = self._inter_op_threads
        so.graph_optimization_level = getattr(
            ort.GraphOptimizationLevel, f"ORT_ENABLE_{self._optimization_level.upper()}"
        )
        providers = self._providers or ort.get_available_providers()

        self._session = ort.InferenceSession(str(path), sess_options=so, providers=providers)
        self._model_path = str(path)
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        logger.info("Loaded ONNX model %s (providers=%s)", path, self._session.get_providers())
        return self.model_info

    @property
    def model_info(self) -> ModelInfo:
        """Describe the loaded model's inputs/outputs."""
        self._require_session()
        assert self._session is not None
        return ModelInfo(
            input_names=[i.name for i in self._session.get_inputs()],
            output_names=[o.name for o in self._session.get_outputs()],
            input_shapes={i.name: list(i.shape) for i in self._session.get_inputs()},
            output_shapes={o.name: list(o.shape) for o in self._session.get_outputs()},
            providers=list(self._session.get_providers()),
        )

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def release(self) -> None:
        """Release the underlying session and free native resources."""
        self._session = None
        self._model_path = None
        self._input_name = None
        self._output_names = []

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    def preprocess(
        self,
        images: Sequence[np.ndarray],
        input_shape: Optional[Sequence[int]] = None,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> np.ndarray:
        """Convert HWC uint8/float images to a normalized NCHW float32 tensor.

        *input_shape* is ``(N, C, H, W)``; when given, images are center-cropped
        and resized (nearest-neighbour) to ``(H, W)``. Channel order is RGB.
        """
        if not images:
            raise ValueError("Cannot preprocess an empty image list")
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean and std must have 3 channels")

        target_hw: Optional[Tuple[int, int]] = None
        if input_shape is not None:
            if len(input_shape) != 4:
                raise ValueError("input_shape must be (N, C, H, W)")
            _, c, h, w = input_shape
            if c != 3:
                raise ValueError("Only 3-channel (RGB) models are supported")
            target_hw = (int(h), int(w))

        batch = []
        for img in images:
            arr = np.asarray(img)
            if arr.ndim == 2:  # grayscale -> RGB
                arr = np.stack([arr] * 3, axis=-1)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"Expected HWC image with 3 channels, got shape {arr.shape}")
            if target_hw is not None and (arr.shape[0], arr.shape[1]) != target_hw:
                arr = self._resize_nearest(arr, target_hw)
            arr = (
                arr.astype(np.float32) / 255.0 if arr.dtype == np.uint8 else arr.astype(np.float32)
            )
            arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
            batch.append(np.transpose(arr, (2, 0, 1)))  # HWC -> CHW
        return np.stack(batch, axis=0).astype(np.float32)

    @staticmethod
    def _resize_nearest(img: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        """Nearest-neighbour resize of an HWC image to ``(H, W)``."""
        h, w = target_hw
        src_h, src_w = img.shape[:2]
        row_idx = (np.arange(h) * src_h // h).clip(0, src_h - 1)
        col_idx = (np.arange(w) * src_w // w).clip(0, src_w - 1)
        return img[row_idx][:, col_idx]

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def infer(self, tensor: np.ndarray) -> InferenceResult:
        """Run one inference call on an NCHW float32 tensor."""
        self._require_session()
        assert self._session is not None and self._input_name is not None
        if tensor.ndim != 4:
            raise ValueError(f"Expected NCHW tensor, got shape {tensor.shape}")
        start = time.perf_counter()
        outputs = self._session.run(self._output_names, {self._input_name: tensor})
        latency_ms = (time.perf_counter() - start) * 1000.0
        return InferenceResult(
            outputs={name: np.asarray(v) for name, v in zip(self._output_names, outputs)},
            latency_ms=latency_ms,
        )

    def infer_batch(
        self, tensors: Sequence[np.ndarray], batch_size: int = 1
    ) -> List[InferenceResult]:
        """Run inference over many preprocessed tensors in chunks of *batch_size*."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not tensors:
            return []
        results: List[InferenceResult] = []
        for start in range(0, len(tensors), batch_size):
            chunk = np.stack(tensors[start : start + batch_size], axis=0)
            results.append(self.infer(chunk))
        return results

    def predict(
        self,
        images: Sequence[np.ndarray],
        batch_size: int = 1,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> List[InferenceResult]:
        """Full pipeline: preprocess images then run chunked batch inference."""
        self._require_session()
        assert self._session is not None
        tensor = self.preprocess(
            images, input_shape=self._session.get_inputs()[0].shape, mean=mean, std=std
        )
        return self.infer_batch(list(tensor), batch_size=batch_size)

    # ------------------------------------------------------------------ #
    # Benchmarking
    # ------------------------------------------------------------------ #
    def benchmark(
        self,
        images: Sequence[np.ndarray],
        warmup: int = 2,
        iterations: int = 10,
        batch_size: int = 1,
    ) -> BenchmarkStats:
        """Measure inference latency with warmup, returning percentile stats."""
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        self._require_session()
        for _ in range(max(0, warmup)):
            self.predict(images, batch_size=batch_size)
        latencies: List[float] = []
        for _ in range(iterations):
            results = self.predict(images, batch_size=batch_size)
            latencies.append(sum(r.latency_ms for r in results))
        arr = np.asarray(latencies, dtype=np.float64)
        return BenchmarkStats(
            iterations=iterations,
            total_ms=float(arr.sum()),
            mean_ms=float(arr.mean()),
            min_ms=float(arr.min()),
            max_ms=float(arr.max()),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _require_session(self) -> None:
        if self._session is None:
            raise RuntimeError("No model loaded; call load_model() first")

    def __enter__(self) -> "EdgeAIInference":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()
