"""Genomics orchestrator: publishes analysis results to event bus and knowledge graph.

Bridges CRISPR and protein analysis to the event bus and knowledge graph,
so genomic insights can inform crop recommendations and knowledge discovery.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.digital_twin.knowledge_graph import AgriKnowledgeGraph, Entity
from src.genomics.crispr import CRISPRDesigner
from src.genomics.protein import ProteinAnalyzer
from src.integration.event_bus import DomainEvent, EventBus, EventType

logger = logging.getLogger(__name__)


class GenomicsOrchestrator:
    """Orchestrates genomics analysis and publishes results.

    Runs CRISPR guide design and protein analysis, then publishes results
    to the event bus and adds entities to the knowledge graph.
    """

    def __init__(
        self,
        bus: EventBus,
        knowledge_graph: Optional[AgriKnowledgeGraph] = None,
    ) -> None:
        self._bus = bus
        self._kg = knowledge_graph
        self._crispr = CRISPRDesigner()
        self._protein = ProteinAnalyzer()

    @property
    def bus(self) -> EventBus:
        return self._bus

    @property
    def knowledge_graph(self) -> Optional[AgriKnowledgeGraph]:
        return self._kg

    def analyze_protein(self, sequence: str):
        """Analyze a protein sequence and publish results.

        Args:
            sequence: Amino acid sequence string.

        Returns:
            ProteinResult from the analyzer.
        """
        result = self._protein.analyze(sequence)

        if self._kg is not None:
            entity = Entity(
                id=f"protein_{hash(sequence) & 0xFFFFFF}",
                label=f"Protein ({result.length}aa)",
                entity_type="protein",
                properties={
                    "length": result.length,
                    "molecular_weight": result.molecular_weight,
                    "isoelectric_point": result.isoelectric_point,
                    "hydrophobicity": result.hydrophobicity,
                    "stability_score": result.stability_score,
                    "instability_index": result.instability_index,
                },
            )
            self._kg.add_entity(entity)

        event = DomainEvent(
            event_type=EventType.RECOMMENDATION_PRODUCED,
            source="genomics.orchestrator",
            payload={
                "analysis_type": "protein",
                "length": result.length,
                "molecular_weight": result.molecular_weight,
                "stability_score": result.stability_score,
            },
        )
        self._bus.publish(event)
        return result

    def analyze_crispr(self, dna_sequence: str):
        """Design CRISPR guides and publish results.

        Args:
            dna_sequence: DNA sequence to design guides for.

        Returns:
            List of GuideRNA objects.
        """
        guides = self._crispr.design_guide(dna_sequence)

        if self._kg is not None:
            for i, guide in enumerate(guides):
                entity = Entity(
                    id=f"guide_{hash(guide.sequence) & 0xFFFFFF}_{i}",
                    label=f"Guide RNA ({guide.pam})",
                    entity_type="guide_rna",
                    properties={
                        "sequence": guide.sequence,
                        "pam": guide.pam,
                        "gc_content": guide.gc_content,
                        "off_target_score": guide.off_target_score,
                        "efficiency_score": guide.efficiency_score,
                        "position": guide.position,
                    },
                )
                self._kg.add_entity(entity)

        event = DomainEvent(
            event_type=EventType.RECOMMENDATION_PRODUCED,
            source="genomics.orchestrator",
            payload={
                "analysis_type": "crispr",
                "num_guides": len(guides),
                "guides": [
                    {
                        "sequence": g.sequence,
                        "pam": g.pam,
                        "efficiency_score": g.efficiency_score,
                    }
                    for g in guides
                ],
            },
        )
        self._bus.publish(event)
        return guides

    def subscribe(self, event_type: str, handler) -> None:
        """Subscribe a handler to an event type on the bus."""
        self._bus.subscribe(event_type, handler)
