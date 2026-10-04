"""Tests for genomics orchestrator: analysis → event_bus + knowledge_graph."""

from src.digital_twin.knowledge_graph import AgriKnowledgeGraph
from src.genomics.genomics_orchestrator import GenomicsOrchestrator
from src.integration.event_bus import EventBus, EventType


class TestGenomicsOrchestrator:
    """GenomicsOrchestrator publishes analysis results to event bus and knowledge graph."""

    def test_analyze_protein_publishes_event(self):
        """Protein analysis publishes an event to the bus."""
        bus = EventBus()
        orch = GenomicsOrchestrator(bus=bus)

        orch.analyze_protein("ACDEFGHIKLMNPQRSTVWY")

        events = bus.get_history()
        assert len(events) == 1
        assert events[0].source == "genomics.orchestrator"
        assert events[0].payload["length"] == 20

    def test_analyze_crispr_publishes_event(self):
        """CRISPR analysis publishes an event to the bus."""
        bus = EventBus()
        orch = GenomicsOrchestrator(bus=bus)

        orch.analyze_crispr("ATCGATCGATCGATCGATCGATCGGG")

        events = bus.get_history()
        assert len(events) == 1
        assert events[0].source == "genomics.orchestrator"

    def test_protein_result_added_to_knowledge_graph(self):
        """Protein analysis adds an entity to the knowledge graph."""
        bus = EventBus()
        kg = AgriKnowledgeGraph()
        orch = GenomicsOrchestrator(bus=bus, knowledge_graph=kg)

        orch.analyze_protein("ACDEFGHIKLMNPQRSTVWY")

        entities = kg.query(entity_type="protein")
        assert len(entities) == 1
        assert entities[0].properties["length"] == 20

    def test_crispr_result_added_to_knowledge_graph(self):
        """CRISPR analysis adds guide RNA entities to the knowledge graph."""
        bus = EventBus()
        kg = AgriKnowledgeGraph()
        orch = GenomicsOrchestrator(bus=bus, knowledge_graph=kg)

        orch.analyze_crispr("ATCGATCGATCGATCGATCGATCGGG")

        entities = kg.query(entity_type="guide_rna")
        assert len(entities) > 0

    def test_subscribe_to_events(self):
        """Handlers can subscribe to genomics events."""
        bus = EventBus()
        orch = GenomicsOrchestrator(bus=bus)
        received = []
        orch.subscribe(EventType.RECOMMENDATION_PRODUCED, received.append)

        orch.analyze_protein("ACDEFGHIKLMNPQRSTVWY")

        assert len(received) == 1
