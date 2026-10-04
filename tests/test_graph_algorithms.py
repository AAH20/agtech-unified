"""Test graph algorithms for agricultural network analysis."""

from src.digital_twin.graph_algorithms import (
    CommunityDetectionResult,
    Graph,
    GraphAlgorithms,
    PageRankResult,
)


def test_graph_add_node_and_edge():
    """Nodes and edges are correctly added to the graph."""
    g = Graph()
    g.add_node("field_a")
    g.add_node("field_b")
    g.add_edge("field_a", "field_b", weight=2.0)
    assert "field_a" in g.nodes
    assert "field_b" in g.nodes
    assert g.edge_weight("field_a", "field_b") == 2.0


def test_graph_neighbors():
    """Outgoing and incoming neighbors are correctly identified."""
    g = Graph()
    g.add_edge("a", "b")
    g.add_edge("c", "a")
    assert g.get_neighbors("a") == ["b"]
    assert g.get_in_neighbors("a") == ["c"]
    assert g.out_degree("a") == 1
    assert g.in_degree("a") == 1


def test_graph_remove_edge():
    """Edge removal updates the graph correctly."""
    g = Graph()
    g.add_edge("a", "b", weight=1.0)
    g.remove_edge("a", "b")
    assert g.edge_weight("a", "b") == 0.0
    assert g.get_neighbors("a") == []


def test_pagerank_simple_chain():
    """PageRank on a simple chain gives higher rank to later nodes (sinks)."""
    g = Graph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    result = GraphAlgorithms.pagerank(g)
    assert isinstance(result, PageRankResult)
    assert result.converged is True
    assert result.scores["c"] > result.scores["a"]


def test_pagerank_uniform_graph():
    """PageRank on a complete graph gives uniform scores."""
    g = Graph()
    nodes = ["a", "b", "c", "d"]
    for n1 in nodes:
        for n2 in nodes:
            if n1 != n2:
                g.add_edge(n1, n2, weight=1.0)
    result = GraphAlgorithms.pagerank(g)
    scores = list(result.scores.values())
    assert all(abs(s - scores[0]) < 1e-6 for s in scores)


def test_pagerank_empty_graph():
    """PageRank on empty graph returns empty scores."""
    g = Graph()
    result = GraphAlgorithms.pagerank(g)
    assert result.scores == {}
    assert result.converged is True


def test_pagerank_dangling_nodes():
    """PageRank handles dangling nodes (no out-edges)."""
    g = Graph()
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    # b and c have no out-edges
    result = GraphAlgorithms.pagerank(g)
    assert result.converged is True
    assert all(score > 0 for score in result.scores.values())


def test_detect_communities_two_clusters():
    """Community detection finds two clear clusters."""
    g = Graph()
    # Cluster 1: a-b-c
    g.add_edge("a", "b", weight=1.0)
    g.add_edge("b", "c", weight=1.0)
    g.add_edge("a", "c", weight=1.0)
    # Cluster 2: d-e-f
    g.add_edge("d", "e", weight=1.0)
    g.add_edge("e", "f", weight=1.0)
    g.add_edge("d", "f", weight=1.0)
    # Weak bridge
    g.add_edge("c", "d", weight=0.1)
    result = GraphAlgorithms.detect_communities(g, seed=42)
    assert isinstance(result, CommunityDetectionResult)
    assert result.num_communities == 2
    # Nodes in same cluster should have same community
    assert result.communities["a"] == result.communities["b"]
    assert result.communities["d"] == result.communities["e"]


def test_detect_communities_empty_graph():
    """Community detection on empty graph returns empty result."""
    g = Graph()
    result = GraphAlgorithms.detect_communities(g)
    assert result.communities == {}
    assert result.num_communities == 0


def test_detect_communities_modularity():
    """Modularity is non-negative for valid partitions."""
    g = Graph()
    g.add_edge("a", "b", weight=1.0)
    g.add_edge("b", "c", weight=1.0)
    g.add_edge("c", "a", weight=1.0)
    g.add_edge("d", "e", weight=1.0)
    g.add_edge("e", "f", weight=1.0)
    g.add_edge("f", "d", weight=1.0)
    result = GraphAlgorithms.detect_communities(g, seed=42)
    assert result.modularity >= 0.0


def test_betweenness_centrality():
    """Betweenness centrality identifies bridge nodes."""
    g = Graph()
    # Path graph: middle nodes have higher betweenness
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "d")
    g.add_edge("d", "e")
    result = GraphAlgorithms.betweenness_centrality(g)
    assert result["c"] > result["a"]
    assert result["c"] > result["e"]


def test_betweenness_centrality_empty():
    """Betweenness centrality on empty graph returns empty dict."""
    g = Graph()
    result = GraphAlgorithms.betweenness_centrality(g)
    assert result == {}


def test_adjacency_matrix():
    """Adjacency matrix is correctly constructed."""
    g = Graph()
    g.add_edge("a", "b", weight=3.0)
    g.add_edge("b", "c", weight=2.0)
    node_list, matrix = g.to_adjacency_matrix()
    assert node_list == ["a", "b", "c"]
    assert matrix[0][1] == 3.0  # a->b
    assert matrix[1][2] == 2.0  # b->c
    assert matrix[0][2] == 0.0  # a->c (no edge)
