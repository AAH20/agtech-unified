"""Tests for graph adapter between AgriKnowledgeGraph and Graph."""

from src.digital_twin.graph_algorithms import Graph, GraphAlgorithms
from src.digital_twin.knowledge_graph import AgriKnowledgeGraph, Entity, Relationship


class TestGraphAdapter:
    """Test adapter from AgriKnowledgeGraph to Graph."""

    def test_convert_entities_to_nodes(self):
        """Entities become graph nodes."""
        kg = AgriKnowledgeGraph()
        kg.add_entity(Entity(id="f1", label="Field 1", entity_type="field"))
        kg.add_entity(Entity(id="c1", label="Crop 1", entity_type="crop"))

        g = Graph.from_knowledge_graph(kg)
        assert "f1" in g.nodes
        assert "c1" in g.nodes

    def test_convert_relationships_to_edges(self):
        """Relationships become weighted edges."""
        kg = AgriKnowledgeGraph()
        kg.add_entity(Entity(id="f1", label="Field 1", entity_type="field"))
        kg.add_entity(Entity(id="c1", label="Crop 1", entity_type="crop"))
        kg.add_relationship(Relationship(source="f1", target="c1", rel_type="grows"))

        g = Graph.from_knowledge_graph(kg)
        assert g.edge_weight("f1", "c1") > 0.0

    def test_pagerank_on_knowledge_graph(self):
        """PageRank works on converted knowledge graph."""
        kg = AgriKnowledgeGraph()
        kg.add_entity(Entity(id="f1", label="Field 1", entity_type="field"))
        kg.add_entity(Entity(id="c1", label="Crop 1", entity_type="crop"))
        kg.add_entity(Entity(id="s1", label="Sensor 1", entity_type="sensor"))
        kg.add_relationship(Relationship(source="f1", target="c1", rel_type="grows"))
        kg.add_relationship(Relationship(source="s1", target="f1", rel_type="located_in"))

        g = Graph.from_knowledge_graph(kg)
        result = GraphAlgorithms.pagerank(g)
        assert len(result.scores) == 3
        assert all(0.0 <= v <= 1.0 for v in result.scores.values())

    def test_community_detection_on_knowledge_graph(self):
        """Community detection works on converted graph."""
        kg = AgriKnowledgeGraph()
        kg.add_entity(Entity(id="f1", label="Field 1", entity_type="field"))
        kg.add_entity(Entity(id="f2", label="Field 2", entity_type="field"))
        kg.add_entity(Entity(id="c1", label="Crop 1", entity_type="crop"))
        kg.add_entity(Entity(id="c2", label="Crop 2", entity_type="crop"))
        kg.add_relationship(Relationship(source="f1", target="c1", rel_type="grows"))
        kg.add_relationship(Relationship(source="f2", target="c2", rel_type="grows"))

        g = Graph.from_knowledge_graph(kg)
        result = GraphAlgorithms.detect_communities(g)
        assert len(result.communities) == 4

    def test_edge_weights_from_properties(self):
        """Edge weights derived from relationship properties."""
        kg = AgriKnowledgeGraph()
        kg.add_entity(Entity(id="f1", label="Field 1", entity_type="field"))
        kg.add_entity(Entity(id="c1", label="Crop 1", entity_type="crop"))
        kg.add_relationship(
            Relationship(
                source="f1",
                target="c1",
                rel_type="grows",
                properties={"strength": 0.8},
            )
        )

        g = Graph.from_knowledge_graph(kg)
        assert g.edge_weight("f1", "c1") == 0.8

    def test_empty_graph_conversion(self):
        """Empty knowledge graph converts to empty graph."""
        kg = AgriKnowledgeGraph()
        g = Graph.from_knowledge_graph(kg)
        assert len(g.nodes) == 0

    def test_bidirectional_sync(self):
        """Changes in KG reflect in Graph and vice versa."""
        kg = AgriKnowledgeGraph()
        kg.add_entity(Entity(id="f1", label="Field 1", entity_type="field"))
        kg.add_entity(Entity(id="c1", label="Crop 1", entity_type="crop"))
        kg.add_relationship(Relationship(source="f1", target="c1", rel_type="grows"))

        g = Graph.from_knowledge_graph(kg)
        assert "f1" in g.nodes

        # Add to KG
        kg.add_entity(Entity(id="s1", label="Sensor 1", entity_type="sensor"))
        # Graph should be updatable
        g2 = Graph.from_knowledge_graph(kg)
        assert "s1" in g2.nodes
