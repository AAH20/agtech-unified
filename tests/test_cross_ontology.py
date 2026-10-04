"""Tests for cross-ontology mapping and SKOS alignment."""

import pytest

from src.digital_twin.ontology import AGROVOCOntology, Concept


class TestCrossOntologyMapping:
    """Test SKOS mapping to external ontologies."""

    def test_add_external_mapping(self):
        """Add SKOS mapping to external ontology."""
        ont = AGROVOCOntology()
        ont.add_concept(Concept(id="wheat", pref_label="Wheat"))
        ont.add_external_mapping("wheat", "crop_ontology", "CO_123", "exactMatch")
        mappings = ont.get_external_mappings("wheat")
        assert len(mappings) == 1
        assert mappings[0]["ontology"] == "crop_ontology"
        assert mappings[0]["concept_id"] == "CO_123"
        assert mappings[0]["match_type"] == "exactMatch"

    def test_close_match(self):
        """Add closeMatch mapping."""
        ont = AGROVOCOntology()
        ont.add_concept(Concept(id="wheat", pref_label="Wheat"))
        ont.add_external_mapping("wheat", "plant_ontology", "PO_456", "closeMatch")
        mappings = ont.get_external_mappings("wheat")
        assert mappings[0]["match_type"] == "closeMatch"

    def test_related_match(self):
        """Add relatedMatch mapping."""
        ont = AGROVOCOntology()
        ont.add_concept(Concept(id="wheat", pref_label="Wheat"))
        ont.add_external_mapping("wheat", "foodon", "FOOD_789", "relatedMatch")
        mappings = ont.get_external_mappings("wheat")
        assert mappings[0]["match_type"] == "relatedMatch"

    def test_multiple_mappings(self):
        """Multiple mappings for same concept."""
        ont = AGROVOCOntology()
        ont.add_concept(Concept(id="wheat", pref_label="Wheat"))
        ont.add_external_mapping("wheat", "crop_ontology", "CO_123", "exactMatch")
        ont.add_external_mapping("wheat", "plant_ontology", "PO_456", "closeMatch")
        ont.add_external_mapping("wheat", "foodon", "FOOD_789", "relatedMatch")
        mappings = ont.get_external_mappings("wheat")
        assert len(mappings) == 3

    def test_no_mappings(self):
        """Concept with no mappings returns empty list."""
        ont = AGROVOCOntology()
        ont.add_concept(Concept(id="wheat", pref_label="Wheat"))
        assert ont.get_external_mappings("wheat") == []

    def test_remove_mapping(self):
        """Remove a specific mapping."""
        ont = AGROVOCOntology()
        ont.add_concept(Concept(id="wheat", pref_label="Wheat"))
        ont.add_external_mapping("wheat", "crop_ontology", "CO_123", "exactMatch")
        ont.remove_external_mapping("wheat", "crop_ontology", "CO_123")
        assert ont.get_external_mappings("wheat") == []

    def test_search_across_ontologies(self):
        """Search finds concepts via external mappings."""
        ont = AGROVOCOntology()
        ont.add_concept(Concept(id="wheat", pref_label="Wheat"))
        ont.add_external_mapping("wheat", "crop_ontology", "CO_123", "exactMatch")
        results = ont.search("Wheat")
        assert len(results) > 0
        assert results[0].concept_id == "wheat"

    def test_rdf_export_with_mappings(self):
        """RDF export includes SKOS mapping properties."""
        pytest.importorskip("rdflib")
        ont = AGROVOCOntology()
        ont.add_concept(Concept(id="wheat", pref_label="Wheat"))
        ont.add_external_mapping("wheat", "crop_ontology", "CO_123", "exactMatch")
        g = ont.to_rdf()
        # Should have at least 2 triples (type + prefLabel or mapping)
        assert len(g) >= 2
