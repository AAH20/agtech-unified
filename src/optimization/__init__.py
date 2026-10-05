"""Optimization module: TSP/VRP solvers, GPU acceleration, edge AI, and local search."""

from src.optimization.ant_colony import AntColonyTSP
from src.optimization.crop_vision import (
    DiseaseDetection,
    FieldHealthReport,
    WeedClassification,
    YieldPrediction,
)
from src.optimization.edge_ai import BenchmarkStats, EdgeAIInference, InferenceResult, ModelInfo
from src.optimization.genetic_algorithm import GeneticAlgorithmTSP
from src.optimization.gpu_fallback import GPUFallback

try:
    from src.optimization.gpu_tsp import GPUTSPResult, GPUTSPSolver
    from src.optimization.gpu_vrp import GPUVRPResult, GPUVRPSolver
except ImportError:
    GPUTSPResult = None
    GPUTSPSolver = None
    GPUVRPResult = None
    GPUVRPSolver = None
from src.optimization.local_search import TwoOpt
from src.optimization.simulated_annealing import SimulatedAnnealingTSP
from src.optimization.three_opt import ThreeOpt
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
    "ThreeOpt",
    "SimulatedAnnealingTSP",
    "GeneticAlgorithmTSP",
    "AntColonyTSP",
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
