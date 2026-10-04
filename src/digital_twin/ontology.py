"""AGROVOC ontology integration for agricultural knowledge.

Provides concept management, semantic search, hierarchy navigation,
and RDF import/export using SKOS-compatible structures.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class Concept:
    """An AGROVOC concept with labels and hierarchy."""
    id: str
    pref_label: str
    alt_labels: List[str] = field(default_factory=list)
    broader: Optional[str] = None
    narrower: List[str] = field(default_factory=list)
    related: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """A semantic search result with relevance score."""
    concept_id: str
    score: float
    matched_label: str


class AGROVOCOntology:
    """AGROVOC ontology with concept mapping and semantic search."""

    def __init__(self):
        self._concepts: Dict[str, Concept] = {}
        self._narrower_map: Dict[str, Set[str]] = {}

    def add_concept(self, concept: Concept) -> None:
        """Add a concept to the ontology."""
        if concept.id in self._concepts:
            raise ValueError(f"Concept '{concept.id}' already exists")
        self._concepts[concept.id] = concept
        # Build narrower map from broader links
        if concept.broader:
            self._narrower_map.setdefault(concept.broader, set()).add(concept.id)

    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Retrieve a concept by ID."""
        return self._concepts.get(concept_id)

    def remove_concept(self, concept_id: str) -> None:
        """Remove a concept and clean up hierarchy links."""
        if concept_id not in self._concepts:
            return
        concept = self._concepts[concept_id]
        # Remove from parent's narrower set
        if concept.broader and concept.broader in self._narrower_map:
            self._narrower_map[concept.broader].discard(concept_id)
        # Clear broader reference from all children (via narrower map)
        children = self._narrower_map.pop(concept_id, set())
        for child_id in children:
            if child_id in self._concepts:
                self._concepts[child_id].broader = None
        # Also clear from concept.narrower list if present
        for child_id in concept.narrower:
            if child_id in self._concepts:
                self._concepts[child_id].broader = None
        del self._concepts[concept_id]

    def count(self) -> int:
        """Return the number of concepts."""
        return len(self._concepts)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Semantic search over concept labels.

        Scores by exact match (1.0), prefix match (0.8),
        substring match (0.5), and token overlap (0.3+).
        """
        if not query or not query.strip():
            return []
        query_lower = query.lower().strip()
        query_tokens = set(query_lower.split())
        results: List[SearchResult] = []

        for cid, concept in self._concepts.items():
            labels = [concept.pref_label] + concept.alt_labels
            best_score = 0.0
            best_label = concept.pref_label
            for label in labels:
                label_lower = label.lower()
                if label_lower == query_lower:
                    score = 1.0
                elif label_lower.startswith(query_lower):
                    score = 0.8
                elif query_lower in label_lower:
                    score = 0.5
                else:
                    label_tokens = set(label_lower.split())
                    if label_tokens and query_tokens:
                        overlap = len(query_tokens & label_tokens) / len(query_tokens)
                        if overlap > 0:
                            score = 0.3 * overlap
                        else:
                            score = 0.0
                    else:
                        score = 0.0
                if score > best_score:
                    best_score = score
                    best_label = label
            if best_score > 0:
                results.append(SearchResult(
                    concept_id=cid,
                    score=best_score,
                    matched_label=best_label,
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def map_text(self, text: str) -> Optional[str]:
        """Map free text to the best-matching concept ID."""
        results = self.search(text, top_k=1)
        return results[0].concept_id if results else None

    def get_narrower(self, concept_id: str) -> List[str]:
        """Get narrower (child) concept IDs."""
        return sorted(self._narrower_map.get(concept_id, set()))

    def get_broader(self, concept_id: str) -> Optional[str]:
        """Get broader (parent) concept ID."""
        concept = self._concepts.get(concept_id)
        return concept.broader if concept else None

    def get_related(self, concept_id: str) -> List[str]:
        """Get related concept IDs."""
        concept = self._concepts.get(concept_id)
        return list(concept.related) if concept else []

    def to_rdf(self):
        """Export ontology as an rdflib Graph with SKOS triples."""
        from rdflib import Graph, Namespace, Literal, URIRef
        from rdflib.namespace import RDF, RDFS

        SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
        AGR = Namespace("http://aims.fao.org/aos/agrovoc/")

        g = Graph()
        g.bind("skos", SKOS)
        g.bind("agrovoc", AGR)

        for cid, concept in self._concepts.items():
            uri = AGR[cid]
            g.add((uri, RDF.type, SKOS.Concept))
            g.add((uri, SKOS.prefLabel, Literal(concept.pref_label)))
            for alt in concept.alt_labels:
                g.add((uri, SKOS.altLabel, Literal(alt)))
            if concept.broader:
                g.add((uri, SKOS.broader, AGR[concept.broader]))
            for child in concept.narrower:
                g.add((uri, SKOS.narrower, AGR[child]))
            for rel in concept.related:
                g.add((uri, SKOS.related, AGR[rel]))

        return g

    @classmethod
    def from_rdf(cls, graph) -> "AGROVOCOntology":
        """Import ontology from an rdflib Graph."""
        from rdflib import Namespace, Literal
        from rdflib.namespace import RDF

        SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
        AGR = Namespace("http://aims.fao.org/aos/agrovoc/")

        ont = cls()
        for uri in graph.subjects(RDF.type, SKOS.Concept):
            # Extract concept ID from URI
            cid = str(uri).split("/")[-1]
            pref_labels = list(graph.objects(uri, SKOS.prefLabel))
            pref_label = str(pref_labels[0]) if pref_labels else cid
            alt_labels = [str(l) for l in graph.objects(uri, SKOS.altLabel)]
            broader_list = list(graph.objects(uri, SKOS.broader))
            broader = str(broader_list[0]).split("/")[-1] if broader_list else None
            narrower = [str(n).split("/")[-1] for n in graph.objects(uri, SKOS.narrower)]
            related = [str(r).split("/")[-1] for r in graph.objects(uri, SKOS.related)]

            ont.add_concept(Concept(
                id=cid,
                pref_label=pref_label,
                alt_labels=alt_labels,
                broader=broader,
                narrower=narrower,
                related=related,
            ))

        return ont
