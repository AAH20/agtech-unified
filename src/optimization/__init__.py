"""Optimization module: TSP/VRP solvers, GPU acceleration, edge AI, and local search."""

from src.optimization.crop_vision import (
    DiseaseDetection,
    FieldHealthReport,
    WeedClassification,
    YieldPrediction,
)
from src.optimization.edge_ai import BenchmarkStats, EdgeAIInference, InferenceResult, ModelInfo
from src.optimization.gpu_fallback import GPUFallback
from src.optimization.gpu_tsp import GPUTSPResult, GPUTSPSolver
from src.optimization.gpu_vrp import GPUVRPResult, GPUVRPSolver
from src.optimization.local_search import TwoOpt
from src.optimization.tsp import TSPInstance, TSPResult, TSPSolver
from src.optimization.vrp import VRPInstance, VRPResult, VRPSolver

__all__ = [
    "TSPInstance",
    "TSPResult",
    "TSPSolver",
    "VRPInstance",
    "VRPResult",
    "VRPSolver",
    "GPUTSPResult",
    "GPUTSPSolver",
    "GPUVRPResult",
    "GPUVRPSolver",
    "TwoOpt",
    "GPUFallback",
    "EdgeAIInference",
    "InferenceResult",
    "BenchmarkStats",
    "ModelInfo",
    "DiseaseDetection",
    "WeedClassification",
    "YieldPrediction",
    "FieldHealthReport",
]
