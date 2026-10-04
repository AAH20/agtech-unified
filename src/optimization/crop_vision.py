"""Crop vision pipeline: disease detection, weed classification, yield prediction.
Orchestrates three ONNX models through :class:`EdgeAIInference`:

* **Disease detection** — multi-class classifier over crop leaf conditions.
* **Weed classification** — multi-class classifier over weed types.
* **Yield prediction** — regression model mapping canopy imagery to t/ha.

Each stage returns a typed result dataclass; :meth:`CropVisionPipeline.analyze_field`
runs all three and produces a combined field-health report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.optimization.edge_ai import EdgeAIInference, InferenceResult

logger = logging.getLogger(__name__)

DISEASE_CLASSES: Tuple[str, ...] = (
    "healthy",
    "leaf_blight",
    "leaf_spot",
    "powdery_mildew",
    "rust",
)

WEED_CLASSES: Tuple[str, ...] = (
    "none",
    "grass",
    "broadleaf",
    "sedge",
)


@dataclass
class DiseaseDetection:
    """Result of disease detection on a single image."""

    label: str
    confidence: float
    severity: float  # 0..1, fraction of non-healthy probability mass
    probabilities: Dict[str, float] = field(default_factory=dict)


@dataclass
class WeedClassification:
    """Result of weed classification on a single image."""

    label: str
    confidence: float
    coverage_ratio: float  # 0..1, fraction of probability mass on weed classes
    probabilities: Dict[str, float] = field(default_factory=dict)


@dataclass
class YieldPrediction:
    """Result of yield prediction over a set of images."""

    estimated_yield_t_per_ha: float
    confidence_interval: Tuple[float, float]
    per_image_yields: List[float] = field(default_factory=list)


@dataclass
class FieldHealthReport:
    """Combined analysis of a field from drone imagery."""

    disease: DiseaseDetection
    weed: WeedClassification
    yield_t_per_ha: float
    health_score: float  # 0..1 composite
    recommendations: List[str] = field(default_factory=list)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    x = np.asarray(logits, dtype=np.float64).ravel()
    x = x - x.max()
    exp = np.exp(x)
    return exp / exp.sum()


class CropVisionPipeline:
    """End-to-end crop vision pipeline over drone imagery.

    Parameters
    ----------
    disease_model, weed_model, yield_model:
        Paths to the three ONNX models. Any may be ``None``; the corresponding
        stage is then skipped (its result reports "unavailable").
    batch_size:
        Inference chunk size passed to the edge engine.
    """

    def __init__(
        self,
        disease_model: Optional[str] = None,
        weed_model: Optional[str] = None,
        yield_model: Optional[str] = None,
        batch_size: int = 4,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        self.batch_size = batch_size
        self._disease = (
            EdgeAIInference(disease_model, providers=providers) if disease_model else None
        )
        self._weed = EdgeAIInference(weed_model, providers=providers) if weed_model else None
        self._yield = EdgeAIInference(yield_model, providers=providers) if yield_model else None

    # ------------------------------------------------------------------ #
    # Stage 1: disease detection
    # ------------------------------------------------------------------ #
    def detect_disease(self, images: Sequence[np.ndarray]) -> List[DiseaseDetection]:
        """Classify each image into a crop disease category."""
        if self._disease is None:
            raise RuntimeError("Disease model not configured")
        if not images:
            raise ValueError("detect_disease requires at least one image")
        results = self._disease.predict(images, batch_size=self.batch_size)
        detections: List[DiseaseDetection] = []
        for res in results:
            logits = self._primary_output(res)  # (B, C) — one row per image
            if logits.ndim != 2 or logits.shape[1] != len(DISEASE_CLASSES):
                raise ValueError(
                    f"Disease model must output (B, {len(DISEASE_CLASSES)}), got {logits.shape}"
                )
            for row in logits:
                probs = _softmax(row)
                prob_map = dict(zip(DISEASE_CLASSES, probs.tolist()))
                idx = int(np.argmax(probs))
                severity = float(1.0 - prob_map["healthy"])
                detections.append(
                    DiseaseDetection(
                        label=DISEASE_CLASSES[idx],
                        confidence=float(probs[idx]),
                        severity=severity,
                        probabilities=prob_map,
                    )
                )
        return detections

    # ------------------------------------------------------------------ #
    # Stage 2: weed classification
    # ------------------------------------------------------------------ #
    def classify_weeds(self, images: Sequence[np.ndarray]) -> List[WeedClassification]:
        """Classify each image into a weed category."""
        if self._weed is None:
            raise RuntimeError("Weed model not configured")
        if not images:
            raise ValueError("classify_weeds requires at least one image")
        results = self._weed.predict(images, batch_size=self.batch_size)
        classifications: List[WeedClassification] = []
        for res in results:
            logits = self._primary_output(res)  # (B, C) — one row per image
            if logits.ndim != 2 or logits.shape[1] != len(WEED_CLASSES):
                raise ValueError(
                    f"Weed model must output (B, {len(WEED_CLASSES)}), got {logits.shape}"
                )
            for row in logits:
                probs = _softmax(row)
                prob_map = dict(zip(WEED_CLASSES, probs.tolist()))
                idx = int(np.argmax(probs))
                coverage = float(1.0 - prob_map["none"])
                classifications.append(
                    WeedClassification(
                        label=WEED_CLASSES[idx],
                        confidence=float(probs[idx]),
                        coverage_ratio=coverage,
                        probabilities=prob_map,
                    )
                )
        return classifications

    # ------------------------------------------------------------------ #
    # Stage 3: yield prediction
    # ------------------------------------------------------------------ #
    def predict_yield(self, images: Sequence[np.ndarray]) -> YieldPrediction:
        """Predict yield (t/ha) from canopy imagery.

        The model emits one value per image; the field estimate is the mean
        and the 95% confidence interval uses the standard error of the mean.
        """
        if self._yield is None:
            raise RuntimeError("Yield model not configured")
        if not images:
            raise ValueError("predict_yield requires at least one image")
        results = self._yield.predict(images, batch_size=self.batch_size)
        per_image: List[float] = []
        for res in results:
            out = self._primary_output(res).ravel()  # (B,) — one value per image
            # Validate against actual chunk size, not self.batch_size
            # The last chunk may be smaller than batch_size
            if out.size == 0 or out.size > len(images):
                raise ValueError(f"Yield model returned {out.size} values for {len(images)} images")
            per_image.extend(float(v) for v in out)
        arr = np.asarray(per_image, dtype=np.float64)
        mean = float(arr.mean())
        if arr.size > 1:
            sem = float(arr.std(ddof=1) / np.sqrt(arr.size))
        else:
            sem = 0.0
        return YieldPrediction(
            estimated_yield_t_per_ha=mean,
            confidence_interval=(mean - 1.96 * sem, mean + 1.96 * sem),
            per_image_yields=per_image,
        )

    # ------------------------------------------------------------------ #
    # Combined analysis
    # ------------------------------------------------------------------ #
    def analyze_field(self, images: Sequence[np.ndarray]) -> FieldHealthReport:
        """Run all configured stages and build a composite field report."""
        if not images:
            raise ValueError("analyze_field requires at least one image")

        if self._disease is not None:
            detections = self.detect_disease(images)
            disease = max(detections, key=lambda d: d.severity)
        else:
            disease = DiseaseDetection("unavailable", 0.0, 0.0)

        if self._weed is not None:
            weed = self._aggregate_weed(self.classify_weeds(images))
        else:
            weed = WeedClassification("unavailable", 0.0, 0.0)

        if self._yield is not None:
            yield_pred = self.predict_yield(images)
            yield_val = yield_pred.estimated_yield_t_per_ha
        else:
            yield_val = 0.0

        health_score = self._health_score(disease, weed)
        recommendations = self._recommendations(disease, weed, yield_val)
        return FieldHealthReport(
            disease=disease,
            weed=weed,
            yield_t_per_ha=yield_val,
            health_score=health_score,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _primary_output(result: InferenceResult) -> np.ndarray:
        """Return the first output tensor of an inference result."""
        return next(iter(result.outputs.values()))

    @staticmethod
    def _aggregate_disease(detections: List[DiseaseDetection]) -> DiseaseDetection:
        """Aggregate disease detections across all images using majority vote."""
        if not detections:
            return DiseaseDetection("unavailable", 0.0, 0.0)
        if len(detections) == 1:
            return detections[0]

        # Majority vote on label
        labels = [d.label for d in detections]
        label_counts: Dict[str, int] = {}
        for lbl in labels:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        best_label = max(label_counts, key=lambda k: label_counts[k])

        # Mean severity and confidence
        mean_severity = float(np.mean([d.severity for d in detections]))
        mean_confidence = float(np.mean([d.confidence for d in detections]))

        # Average probabilities
        avg_probs: Dict[str, float] = {}
        for cls in DISEASE_CLASSES:
            avg_probs[cls] = float(np.mean([d.probabilities.get(cls, 0.0) for d in detections]))

        return DiseaseDetection(
            label=best_label,
            confidence=mean_confidence,
            severity=mean_severity,
            probabilities=avg_probs,
        )

    @staticmethod
    def _aggregate_weed(classifications: List[WeedClassification]) -> WeedClassification:
        """Aggregate weed classifications across all images using majority vote."""
        if not classifications:
            return WeedClassification("unavailable", 0.0, 0.0)
        if len(classifications) == 1:
            return classifications[0]

        # Majority vote on label
        labels = [w.label for w in classifications]
        label_counts: Dict[str, int] = {}
        for lbl in labels:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        best_label = max(label_counts, key=lambda k: label_counts[k])

        # Mean coverage and confidence
        mean_coverage = float(np.mean([w.coverage_ratio for w in classifications]))
        mean_confidence = float(np.mean([w.confidence for w in classifications]))

        # Average probabilities
        avg_probs: Dict[str, float] = {}
        for cls in WEED_CLASSES:
            avg_probs[cls] = float(
                np.mean([w.probabilities.get(cls, 0.0) for w in classifications])
            )

        return WeedClassification(
            label=best_label,
            confidence=mean_confidence,
            coverage_ratio=mean_coverage,
            probabilities=avg_probs,
        )

    @staticmethod
    def _health_score(disease: DiseaseDetection, weed: WeedClassification) -> float:
        """Composite 0..1 health score: 1 = pristine crop."""
        return float(np.clip(1.0 - 0.6 * disease.severity - 0.4 * weed.coverage_ratio, 0.0, 1.0))

    @staticmethod
    def _recommendations(
        disease: DiseaseDetection, weed: WeedClassification, yield_t_per_ha: float
    ) -> List[str]:
        recs: List[str] = []
        if disease.label not in ("healthy", "unavailable") and disease.confidence > 0.5:
            recs.append(
                f"Apply targeted treatment for {disease.label} "
                f"(confidence {disease.confidence:.0%})"
            )
        if weed.label not in ("none", "unavailable") and weed.coverage_ratio > 0.2:
            recs.append(f"Spot-spray {weed.label} weeds (coverage {weed.coverage_ratio:.0%})")
        if yield_t_per_ha < 3.0:
            recs.append("Low yield forecast — review irrigation and fertilization")
        return recs

    def release(self) -> None:
        """Release all underlying inference sessions."""
        for engine in (self._disease, self._weed, self._yield):
            if engine is not None:
                engine.release()
