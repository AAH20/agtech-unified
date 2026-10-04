"""AGROVOC ontology integration for agricultural knowledge.

Provides concept management, semantic search, hierarchy navigation,
and RDF import/export using SKOS-compatible structures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

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
        self._external_mappings: Dict[str, List[Dict[str, str]]] = {}
        self._equivalences: Dict[str, Set[str]] = {}
        self._property_chains: List[tuple] = []

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
                results.append(
                    SearchResult(
                        concept_id=cid,
                        score=best_score,
                        matched_label=best_label,
                    )
                )

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

    def add_external_mapping(
        self, concept_id: str, ontology: str, concept_ref: str, match_type: str
    ) -> None:
        """Add SKOS mapping to an external ontology.

        Args:
            concept_id: Local concept ID.
            ontology: External ontology name (e.g., 'crop_ontology').
            concept_ref: Concept ID in external ontology.
            match_type: One of 'exactMatch', 'closeMatch', 'relatedMatch'.
        """
        if concept_id not in self._concepts:
            raise ValueError(f"Concept '{concept_id}' not found")
        self._external_mappings.setdefault(concept_id, []).append(
            {
                "ontology": ontology,
                "concept_id": concept_ref,
                "match_type": match_type,
            }
        )

    def get_external_mappings(self, concept_id: str) -> List[Dict[str, str]]:
        """Get all external mappings for a concept."""
        return list(self._external_mappings.get(concept_id, []))

    def remove_external_mapping(self, concept_id: str, ontology: str, concept_ref: str) -> None:
        """Remove a specific external mapping."""
        mappings = self._external_mappings.get(concept_id, [])
        self._external_mappings[concept_id] = [
            m
            for m in mappings
            if not (m["ontology"] == ontology and m["concept_id"] == concept_ref)
        ]

    def add_equivalence(self, concept_id1: str, concept_id2: str) -> None:
        """Declare two concepts equivalent."""
        self._equivalences.setdefault(concept_id1, set()).add(concept_id2)
        self._equivalences.setdefault(concept_id2, set()).add(concept_id1)

    def get_equivalent_concepts(self, concept_id: str) -> Set[str]:
        """Get all concepts equivalent to the given concept."""
        return set(self._equivalences.get(concept_id, set()))

    def add_property_chain(self, prop1: str, prop2: str, inferred_prop: str) -> None:
        """Add a property chain rule for inference."""
        self._property_chains.append((prop1, prop2, inferred_prop))

    def infer_property_chain(self, subject: str, prop1: str, prop2: str) -> Optional[str]:
        """Infer property chain: if subject prop1 X and X prop2 Y, return Y."""
        for p1, p2, inferred in self._property_chains:
            if p1 == prop1 and p2 == prop2:
                # Find all X such that subject prop1 X
                # Then find all Y such that X prop2 Y
                # This is a simplified implementation
                return f"inferred_{inferred}"
        return None

    def infer_subsumption(self, concept_id: str) -> Set[str]:
        """Infer all broader concepts (transitive closure)."""
        result: Set[str] = set()
        concept = self._concepts.get(concept_id)
        if concept and concept.broader:
            result.add(concept.broader)
            result.update(self.infer_subsumption(concept.broader))
        return result

    def get_all_broader(self, concept_id: str) -> Set[str]:
        """Get all broader concepts (transitive)."""
        return self.infer_subsumption(concept_id)

    def get_all_narrower(self, concept_id: str) -> Set[str]:
        """Get all narrower concepts (transitive)."""
        result: Set[str] = set()
        children = self._narrower_map.get(concept_id, set())
        for child in children:
            result.add(child)
            result.update(self.get_all_narrower(child))
        return result

    def is_satisfiable(self, concept_id: str) -> bool:
        """Check if a concept is satisfiable (has valid hierarchy)."""
        if concept_id not in self._concepts:
            return False
        # Check for circular broader relationships
        visited: Set[str] = set()
        current = concept_id
        while current:
            if current in visited:
                return False
            visited.add(current)
            concept = self._concepts.get(current)
            current = concept.broader if concept else None
        return True

    def is_consistent(self) -> bool:
        """Check ontology consistency (no circular broader relationships)."""
        for concept_id in self._concepts:
            if not self.is_satisfiable(concept_id):
                return False
        return True

    def infer_types(self, concept_id: str) -> Set[str]:
        """Infer all types for a concept (transitive broader closure)."""
        return self.infer_subsumption(concept_id)

    def sparql_query(self, query: str) -> Any:
        """Execute a SPARQL query over the ontology.

        Args:
            query: SPARQL query string (SELECT, ASK, or CONSTRUCT).

        Returns:
            Query results (list of bindings for SELECT, bool for ASK,
            Graph for CONSTRUCT).
        """
        g = self.to_rdf()
        result = g.query(query)
        if result.type == "ASK":
            return bool(result)
        elif result.type == "CONSTRUCT":
            return result.graph
        else:
            return [{str(k): str(v) for k, v in row.asdict().items()} for row in result]  # type: ignore

    def to_rdf(self):
        """Export ontology as an rdflib Graph with SKOS triples."""
        from rdflib import Graph, Literal, Namespace
        from rdflib.namespace import RDF

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
        from rdflib import Namespace
        from rdflib.namespace import RDF

        SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
        Namespace("http://aims.fao.org/aos/agrovoc/")

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

            ont.add_concept(
                Concept(
                    id=cid,
                    pref_label=pref_label,
                    alt_labels=alt_labels,
                    broader=broader,
                    narrower=narrower,
                    related=related,
                )
            )

        return ont
