"""Digital twin module: simulation engine, knowledge graph, ontology, and nutrient/water models."""

from src.digital_twin.disease_model import (
    DiseaseModel,
    DiseaseSimulationResult,
    DiseaseState,
)
from src.digital_twin.graph_algorithms import (
    CommunityDetectionResult,
    Graph,
    GraphAlgorithms,
    PageRankResult,
)
from src.digital_twin.knowledge_graph import AgriKnowledgeGraph, Entity, Relationship
from src.digital_twin.nutrient_cycling import (
    FertilizerApplication,
    NutrientBudget,
    NutrientCyclingModel,
    NutrientCyclingResult,
    NutrientState,
)
from src.digital_twin.ontology import AGROVOCOntology, Concept, SearchResult
from src.digital_twin.pest_model import PestModel, PestSimulationResult, PestState
from src.digital_twin.root_growth import (
    RootGrowthModel,
    RootGrowthResult,
    RootGrowthState,
    RootLayerState,
)
from src.digital_twin.simulator import DigitalTwin, SimulationResult, SimulationState
from src.digital_twin.sync import (
    HierarchicalSynchronizer,
    HierarchicalSyncResult,
    SyncResult,
    TwinLevel,
    TwinSynchronizer,
)
from src.digital_twin.water_balance import (
    IrrigationSchedule,
    SolarGeometry,
    WaterBalanceModel,
    WaterBalanceResult,
    WaterBalanceState,
)
from src.digital_twin.yield_prediction import (
    GrainQuality,
    YieldModel,
    YieldPrediction,
)

__all__ = [
    "SimulationState",
    "SimulationResult",
    "DigitalTwin",
    "Entity",
    "Relationship",
    "AgriKnowledgeGraph",
    "Concept",
    "SearchResult",
    "AGROVOCOntology",
    "NutrientState",
    "FertilizerApplication",
    "NutrientBudget",
    "NutrientCyclingResult",
    "NutrientCyclingModel",
    "WaterBalanceState",
    "IrrigationSchedule",
    "WaterBalanceResult",
    "WaterBalanceModel",
    "SolarGeometry",
    "PestState",
    "PestSimulationResult",
    "PestModel",
    "RootGrowthState",
    "RootLayerState",
    "RootGrowthResult",
    "RootGrowthModel",
    "GrainQuality",
    "YieldPrediction",
    "YieldModel",
    "DiseaseState",
    "DiseaseSimulationResult",
    "DiseaseModel",
    "SyncResult",
    "TwinSynchronizer",
    "HierarchicalSynchronizer",
    "HierarchicalSyncResult",
    "TwinLevel",
    "Graph",
    "PageRankResult",
    "CommunityDetectionResult",
    "GraphAlgorithms",
]
