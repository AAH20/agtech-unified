"""Digital twin module: simulation engine, knowledge graph, ontology, and nutrient/water models."""

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
from src.digital_twin.simulator import DigitalTwin, SimulationResult, SimulationState
from src.digital_twin.water_balance import (
    IrrigationSchedule,
    WaterBalanceModel,
    WaterBalanceResult,
    WaterBalanceState,
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
    "Graph",
    "PageRankResult",
    "CommunityDetectionResult",
    "GraphAlgorithms",
]
