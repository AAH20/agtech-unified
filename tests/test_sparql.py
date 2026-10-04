"""Tests for SPARQL query support on ontology."""

import pytest

rdflib = pytest.importorskip("rdflib")

from src.digital_twin.ontology import AGROVOCOntology, Concept


class TestSPARQLSupport:
    """Test SPARQL query interface."""

    def test_sparql_select(self):
        """Execute SPARQL SELECT query."""
        ont = AGROVOCOntology()
        ont.add_concept(Concept(id="wheat", pref_label="Wheat"))
        ont.add_concept(Concept(id="corn", pref_label="Corn"))
        ont.add_concept(Concept(id="rice", pref_label="Rice"))

        query = """
        SELECT ?concept ?label
        WHERE {
            ?concept <http://www.w3.org/2004/02/skos/core#prefLabel> ?label .
        }
        """
        results = ont.sparql_query(query)
        assert len(results) == 3

    def test_sparql_filter(self):
        """SPARQL query with FILTER."""
        ont = AGROVOCOntology()
        ont.add_concept(Concept(id="wheat", pref_label="Wheat"))
        ont.add_concept(Concept(id="corn", pref_label="Corn"))

        query = """
        SELECT ?label
        WHERE {
            ?concept <http://www.w3.org/2004/02/skos/core#prefLabel> ?label .
            FILTER(CONTAINS(?label, "Wheat"))
        }
        """
        results = ont.sparql_query(query)
        assert len(results) == 1
        assert "Wheat" in str(results[0])

    def test_sparql_ask(self):
        """SPARQL ASK query."""
        ont = AGROVOCOntology()
        ont.add_concept(Concept(id="wheat", pref_label="Wheat"))

        query = """
        ASK {
            ?concept <http://www.w3.org/2004/02/skos/core#prefLabel> "Wheat" .
        }
        """
        result = ont.sparql_query(query)
        assert result is True

    def test_sparql_construct(self):
        """SPARQL CONSTRUCT query."""
        ont = AGROVOCOntology()
        ont.add_concept(Concept(id="wheat", pref_label="Wheat", broader="crop"))

        query = """
        CONSTRUCT {
            ?concept <http://example.org/hasParent> ?parent .
        }
        WHERE {
            ?concept <http://www.w3.org/2004/02/skos/core#broader> ?parent .
        }
        """
        result = ont.sparql_query(query)
        assert result is not None

    def test_sparql_hierarchy(self):
        """SPARQL query for concept hierarchy."""
        ont = AGROVOCOntology()
        ont.add_concept(Concept(id="wheat", pref_label="Wheat", broader="crop"))
        ont.add_concept(Concept(id="corn", pref_label="Corn", broader="crop"))
        ont.add_concept(Concept(id="crop", pref_label="Crop"))

        query = """
        SELECT ?child ?parent
        WHERE {
            ?child <http://www.w3.org/2004/02/skos/core#broader> ?parent .
        }
        """
        results = ont.sparql_query(query)
        assert len(results) == 2

    def test_sparql_empty_ontology(self):
        """SPARQL query on empty ontology."""
        ont = AGROVOCOntology()
        query = """
        SELECT ?concept ?label
        WHERE {
            ?concept <http://www.w3.org/2004/02/skos/core#prefLabel> ?label .
        }
        """
        results = ont.sparql_query(query)
        assert len(results) == 0
