"""Test agricultural knowledge graph with entity management and queries."""

import pytest

from src.digital_twin.knowledge_graph import AgriKnowledgeGraph, Entity, Relationship


def test_add_and_get_entity():
    """Adding an entity makes it retrievable by ID."""
    kg = AgriKnowledgeGraph()
    kg.add_entity(Entity(id="e_001", label="Field_A", entity_type="field"))
    e = kg.get_entity("e_001")
    assert e is not None
    assert e.label == "Field_A"
    assert e.entity_type == "field"


def test_get_missing_entity_returns_none():
    """Unknown entity ID returns None."""
    kg = AgriKnowledgeGraph()
    assert kg.get_entity("nope") is None


def test_add_entity_duplicate_id_raises():
    """Duplicate entity ID raises ValueError."""
    kg = AgriKnowledgeGraph()
    kg.add_entity(Entity(id="e_001", label="Field_A", entity_type="field"))
    with pytest.raises(ValueError, match="already exists"):
        kg.add_entity(Entity(id="e_001", label="Field_B", entity_type="field"))


def test_add_relationship():
    """Relationships connect entities."""
    kg = AgriKnowledgeGraph()
    kg.add_entity(Entity(id="e_001", label="Field_A", entity_type="field"))
    kg.add_entity(Entity(id="e_002", label="Wheat", entity_type="crop"))
    kg.add_relationship(Relationship(source="e_001", target="e_002", rel_type="grows"))
    rels = kg.get_relationships("e_001")
    assert len(rels) == 1
    assert rels[0].target == "e_002"
    assert rels[0].rel_type == "grows"


def test_add_relationship_missing_entity_raises():
    """Relationship with unknown entity raises ValueError."""
    kg = AgriKnowledgeGraph()
    kg.add_entity(Entity(id="e_001", label="Field_A", entity_type="field"))
    with pytest.raises(ValueError, match="not found"):
        kg.add_relationship(Relationship(source="e_001", target="e_missing", rel_type="grows"))


def test_query_by_type():
    """Query returns entities of the given type."""
    kg = AgriKnowledgeGraph()
    kg.add_entity(Entity(id="e_001", label="Field_A", entity_type="field"))
    kg.add_entity(Entity(id="e_002", label="Field_B", entity_type="field"))
    kg.add_entity(Entity(id="e_003", label="Wheat", entity_type="crop"))
    fields = kg.query(entity_type="field")
    assert len(fields) == 2
    assert all(e.entity_type == "field" for e in fields)


def test_query_by_label():
    """Query returns entities matching label."""
    kg = AgriKnowledgeGraph()
    kg.add_entity(Entity(id="e_001", label="Field_A", entity_type="field"))
    kg.add_entity(Entity(id="e_002", label="Field_B", entity_type="field"))
    results = kg.query(label="Field_A")
    assert len(results) == 1
    assert results[0].id == "e_001"


def test_query_no_match():
    """Query with no matches returns empty list."""
    kg = AgriKnowledgeGraph()
    kg.add_entity(Entity(id="e_001", label="Field_A", entity_type="field"))
    assert kg.query(entity_type="nonexistent") == []


def test_query_by_relationship():
    """Query returns entities connected by a relationship type."""
    kg = AgriKnowledgeGraph()
    kg.add_entity(Entity(id="e_001", label="Field_A", entity_type="field"))
    kg.add_entity(Entity(id="e_002", label="Wheat", entity_type="crop"))
    kg.add_entity(Entity(id="e_003", label="Rice", entity_type="crop"))
    kg.add_relationship(Relationship(source="e_001", target="e_002", rel_type="grows"))
    results = kg.query(relationship="grows")
    assert len(results) == 1
    assert results[0].id == "e_002"


def test_remove_entity():
    """Removing an entity makes it unretrievable."""
    kg = AgriKnowledgeGraph()
    kg.add_entity(Entity(id="e_001", label="Field_A", entity_type="field"))
    kg.remove_entity("e_001")
    assert kg.get_entity("e_001") is None


def test_remove_entity_cleans_relationships():
    """Removing an entity removes its relationships."""
    kg = AgriKnowledgeGraph()
    kg.add_entity(Entity(id="e_001", label="Field_A", entity_type="field"))
    kg.add_entity(Entity(id="e_002", label="Wheat", entity_type="crop"))
    kg.add_relationship(Relationship(source="e_001", target="e_002", rel_type="grows"))
    kg.remove_entity("e_001")
    assert kg.get_relationships("e_001") == []
    assert kg.get_relationships("e_002") == []


def test_entity_count():
    """Entity count tracks additions and removals."""
    kg = AgriKnowledgeGraph()
    assert kg.entity_count() == 0
    kg.add_entity(Entity(id="e_001", label="Field_A", entity_type="field"))
    kg.add_entity(Entity(id="e_002", label="Wheat", entity_type="crop"))
    assert kg.entity_count() == 2
    kg.remove_entity("e_001")
    assert kg.entity_count() == 1


def test_relationship_count():
    """Relationship count tracks additions."""
    kg = AgriKnowledgeGraph()
    kg.add_entity(Entity(id="e_001", label="Field_A", entity_type="field"))
    kg.add_entity(Entity(id="e_002", label="Wheat", entity_type="crop"))
    assert kg.relationship_count() == 0
    kg.add_relationship(Relationship(source="e_001", target="e_002", rel_type="grows"))
    assert kg.relationship_count() == 1


def test_get_neighbors():
    """Neighbors returns directly connected entities."""
    kg = AgriKnowledgeGraph()
    kg.add_entity(Entity(id="e_001", label="Field_A", entity_type="field"))
    kg.add_entity(Entity(id="e_002", label="Wheat", entity_type="crop"))
    kg.add_entity(Entity(id="e_003", label="Rice", entity_type="crop"))
    kg.add_relationship(Relationship(source="e_001", target="e_002", rel_type="grows"))
    kg.add_relationship(Relationship(source="e_001", target="e_003", rel_type="grows"))
    neighbors = kg.get_neighbors("e_001")
    assert set(neighbors) == {"e_002", "e_003"}


def test_path_exists():
    """Path exists between connected entities."""
    kg = AgriKnowledgeGraph()
    kg.add_entity(Entity(id="e_001", label="Field_A", entity_type="field"))
    kg.add_entity(Entity(id="e_002", label="Wheat", entity_type="crop"))
    kg.add_entity(Entity(id="e_003", label="Flour", entity_type="product"))
    kg.add_relationship(Relationship(source="e_001", target="e_002", rel_type="grows"))
    kg.add_relationship(Relationship(source="e_002", target="e_003", rel_type="produces"))
    assert kg.path_exists("e_001", "e_003") is True
    assert kg.path_exists("e_003", "e_001") is False


def test_path_exists_same_node():
    """Path from a node to itself always exists."""
    kg = AgriKnowledgeGraph()
    kg.add_entity(Entity(id="e_001", label="Field_A", entity_type="field"))
    assert kg.path_exists("e_001", "e_001") is True


def test_to_dict_serialization():
    """Graph serializes to a dict with entities and relationships."""
    kg = AgriKnowledgeGraph()
    kg.add_entity(Entity(id="e_001", label="Field_A", entity_type="field"))
    kg.add_entity(Entity(id="e_002", label="Wheat", entity_type="crop"))
    kg.add_relationship(Relationship(source="e_001", target="e_002", rel_type="grows"))
    d = kg.to_dict()
    assert "entities" in d
    assert "relationships" in d
    assert len(d["entities"]) == 2
    assert len(d["relationships"]) == 1


def test_from_dict_roundtrip():
    """Dict serialization roundtrips correctly."""
    kg = AgriKnowledgeGraph()
    kg.add_entity(Entity(id="e_001", label="Field_A", entity_type="field"))
    kg.add_entity(Entity(id="e_002", label="Wheat", entity_type="crop"))
    kg.add_relationship(Relationship(source="e_001", target="e_002", rel_type="grows"))
    d = kg.to_dict()
    kg2 = AgriKnowledgeGraph.from_dict(d)
    assert kg2.entity_count() == 2
    assert kg2.relationship_count() == 1
    assert kg2.get_entity("e_001").label == "Field_A"


def test_entity_properties():
    """Entity properties are stored and retrievable."""
    kg = AgriKnowledgeGraph()
    kg.add_entity(
        Entity(
            id="e_001",
            label="Field_A",
            entity_type="field",
            properties={"area_ha": 10.0, "soil_type": "loam"},
        )
    )
    e = kg.get_entity("e_001")
    assert e.properties["area_ha"] == 10.0
    assert e.properties["soil_type"] == "loam"


def test_update_entity_properties():
    """Entity properties can be updated."""
    kg = AgriKnowledgeGraph()
    kg.add_entity(
        Entity(id="e_001", label="Field_A", entity_type="field", properties={"area_ha": 10.0})
    )
    kg.update_entity_properties("e_001", {"area_ha": 15.0})
    e = kg.get_entity("e_001")
    assert e.properties["area_ha"] == 15.0


def test_update_entity_properties_missing_raises():
    """Updating properties of unknown entity raises ValueError."""
    kg = AgriKnowledgeGraph()
    with pytest.raises(ValueError, match="not found"):
        kg.update_entity_properties("e_missing", {"x": 1})


def test_query_with_filters():
    """Query supports combined type and property filters."""
    kg = AgriKnowledgeGraph()
    kg.add_entity(
        Entity(id="e_001", label="Field_A", entity_type="field", properties={"soil_type": "loam"})
    )
    kg.add_entity(
        Entity(id="e_002", label="Field_B", entity_type="field", properties={"soil_type": "clay"})
    )
    kg.add_entity(
        Entity(id="e_003", label="Wheat", entity_type="crop", properties={"soil_type": "loam"})
    )
    results = kg.query(entity_type="field", properties={"soil_type": "loam"})
    assert len(results) == 1
    assert results[0].id == "e_001"


def test_clear_graph():
    """Clearing removes all entities and relationships."""
    kg = AgriKnowledgeGraph()
    kg.add_entity(Entity(id="e_001", label="Field_A", entity_type="field"))
    kg.add_entity(Entity(id="e_002", label="Wheat", entity_type="crop"))
    kg.add_relationship(Relationship(source="e_001", target="e_002", rel_type="grows"))
    kg.clear()
    assert kg.entity_count() == 0
    assert kg.relationship_count() == 0
