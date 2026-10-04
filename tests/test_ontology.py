"""Test AGROVOC ontology integration for agricultural knowledge."""

import pytest

from src.digital_twin.ontology import AGROVOCOntology, Concept


def test_add_and_get_concept():
    """Adding a concept makes it retrievable by ID."""
    ont = AGROVOCOntology()
    ont.add_concept(Concept(id="c_001", pref_label="Wheat"))
    c = ont.get_concept("c_001")
    assert c is not None
    assert c.pref_label == "Wheat"


def test_get_missing_concept_returns_none():
    """Unknown concept ID returns None."""
    ont = AGROVOCOntology()
    assert ont.get_concept("nope") is None


def test_add_concept_duplicate_id_raises():
    """Duplicate concept ID raises ValueError."""
    ont = AGROVOCOntology()
    ont.add_concept(Concept(id="c_001", pref_label="Wheat"))
    with pytest.raises(ValueError, match="already exists"):
        ont.add_concept(Concept(id="c_001", pref_label="Barley"))


def test_semantic_search_exact_match():
    """Exact label match returns the concept with highest score."""
    ont = AGROVOCOntology()
    ont.add_concept(Concept(id="c_001", pref_label="Wheat"))
    ont.add_concept(Concept(id="c_002", pref_label="Rice"))
    results = ont.search("Wheat")
    assert len(results) >= 1
    assert results[0].concept_id == "c_001"
    assert results[0].score == 1.0


def test_semantic_search_partial_match():
    """Partial label match returns relevant concepts."""
    ont = AGROVOCOntology()
    ont.add_concept(Concept(id="c_001", pref_label="Wheat"))
    ont.add_concept(Concept(id="c_002", pref_label="Rice"))
    results = ont.search("Whe")
    assert any(r.concept_id == "c_001" for r in results)


def test_semantic_search_alt_label_match():
    """Search matches alternative labels."""
    ont = AGROVOCOntology()
    ont.add_concept(Concept(id="c_001", pref_label="Wheat", alt_labels=["Triticum"]))
    results = ont.search("Triticum")
    assert any(r.concept_id == "c_001" for r in results)


def test_semantic_search_no_match():
    """No matching concepts returns empty list."""
    ont = AGROVOCOntology()
    ont.add_concept(Concept(id="c_001", pref_label="Wheat"))
    assert ont.search("xyzzy_nonexistent") == []


def test_broader_narrower_hierarchy():
    """Broader/narrower relationships form a hierarchy."""
    ont = AGROVOCOntology()
    ont.add_concept(Concept(id="c_001", pref_label="Crop"))
    ont.add_concept(Concept(id="c_002", pref_label="Wheat", broader="c_001"))
    ont.add_concept(Concept(id="c_003", pref_label="Rice", broader="c_001"))
    children = ont.get_narrower("c_001")
    assert set(children) == {"c_002", "c_003"}
    assert ont.get_broader("c_002") == "c_001"


def test_map_text_to_concept():
    """Free text maps to the best-matching concept."""
    ont = AGROVOCOntology()
    ont.add_concept(Concept(id="c_001", pref_label="Wheat"))
    ont.add_concept(Concept(id="c_002", pref_label="Rice"))
    cid = ont.map_text("Wheat")
    assert cid == "c_001"


def test_map_text_unknown_returns_none():
    """Unmappable text returns None."""
    ont = AGROVOCOntology()
    ont.add_concept(Concept(id="c_001", pref_label="Wheat"))
    assert ont.map_text("xyzzy_nonexistent") is None


def test_rdf_export_contains_concepts():
    """RDF export produces a valid graph with concept triples."""
    ont = AGROVOCOntology()
    ont.add_concept(Concept(id="c_001", pref_label="Wheat"))
    ont.add_concept(Concept(id="c_002", pref_label="Rice", broader="c_001"))
    g = ont.to_rdf()
    assert len(g) > 0
    # Check that prefLabel triple exists
    from rdflib import Namespace

    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
    AGR = Namespace("http://aims.fao.org/aos/agrovoc/")
    wheat = AGR["c_001"]
    labels = list(g.objects(wheat, SKOS.prefLabel))
    assert any(str(l) == "Wheat" for l in labels)


def test_rdf_import_roundtrip():
    """Export then import preserves concepts."""
    ont = AGROVOCOntology()
    ont.add_concept(Concept(id="c_001", pref_label="Wheat"))
    ont.add_concept(Concept(id="c_002", pref_label="Rice", broader="c_001"))
    g = ont.to_rdf()
    ont2 = AGROVOCOntology.from_rdf(g)
    assert ont2.get_concept("c_001") is not None
    assert ont2.get_concept("c_001").pref_label == "Wheat"
    assert ont2.get_concept("c_002").broader == "c_001"


def test_remove_concept():
    """Removing a concept makes it unretrievable."""
    ont = AGROVOCOntology()
    ont.add_concept(Concept(id="c_001", pref_label="Wheat"))
    ont.remove_concept("c_001")
    assert ont.get_concept("c_001") is None


def test_remove_concept_cleans_relationships():
    """Removing a concept removes its hierarchy links."""
    ont = AGROVOCOntology()
    ont.add_concept(Concept(id="c_001", pref_label="Crop"))
    ont.add_concept(Concept(id="c_002", pref_label="Wheat", broader="c_001"))
    ont.remove_concept("c_001")
    assert ont.get_broader("c_002") is None
    assert "c_002" not in ont.get_narrower("c_001")


def test_semantic_search_scored_and_sorted():
    """Search results are scored and sorted descending."""
    ont = AGROVOCOntology()
    ont.add_concept(Concept(id="c_001", pref_label="Wheat"))
    ont.add_concept(Concept(id="c_002", pref_label="Wheatgrass"))
    ont.add_concept(Concept(id="c_003", pref_label="Rice"))
    results = ont.search("Wheat")
    assert len(results) >= 2
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].concept_id == "c_001"


def test_concept_related():
    """Related concepts are stored and retrievable."""
    ont = AGROVOCOntology()
    ont.add_concept(Concept(id="c_001", pref_label="Wheat", related=["c_002"]))
    ont.add_concept(Concept(id="c_002", pref_label="Flour"))
    related = ont.get_related("c_001")
    assert "c_002" in related


def test_concept_count():
    """Concept count tracks additions and removals."""
    ont = AGROVOCOntology()
    assert ont.count() == 0
    ont.add_concept(Concept(id="c_001", pref_label="Wheat"))
    ont.add_concept(Concept(id="c_002", pref_label="Rice"))
    assert ont.count() == 2
    ont.remove_concept("c_001")
    assert ont.count() == 1
