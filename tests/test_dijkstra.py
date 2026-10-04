"""Tests for Dijkstra shortest path in graph algorithms."""

from src.digital_twin.graph_algorithms import Graph, GraphAlgorithms


class TestDijkstra:
    """Test Dijkstra shortest path algorithm."""

    def test_shortest_path_basic(self):
        """Find shortest path in a simple graph."""
        g = Graph()
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)
        g.add_edge("A", "C", weight=4.0)

        dist, path = GraphAlgorithms.dijkstra(g, "A", "C")
        assert dist == 3.0
        assert path == ["A", "B", "C"]

    def test_shortest_path_direct(self):
        """Direct path is shortest when it exists."""
        g = Graph()
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("A", "C", weight=1.5)
        g.add_edge("B", "C", weight=1.0)

        dist, path = GraphAlgorithms.dijkstra(g, "A", "C")
        assert dist == 1.5
        assert path == ["A", "C"]

    def test_no_path(self):
        """No path returns infinity and empty path."""
        g = Graph()
        g.add_edge("A", "B", weight=1.0)
        g.add_node("C")

        dist, path = GraphAlgorithms.dijkstra(g, "A", "C")
        assert dist == float("inf")
        assert path == []

    def test_same_node(self):
        """Path from node to itself is zero."""
        g = Graph()
        g.add_node("A")

        dist, path = GraphAlgorithms.dijkstra(g, "A", "A")
        assert dist == 0.0
        assert path == ["A"]

    def test_complex_graph(self):
        """Shortest path in a complex graph."""
        g = Graph()
        g.add_edge("A", "B", weight=4.0)
        g.add_edge("A", "C", weight=2.0)
        g.add_edge("B", "C", weight=1.0)
        g.add_edge("B", "D", weight=5.0)
        g.add_edge("C", "D", weight=8.0)
        g.add_edge("C", "E", weight=10.0)
        g.add_edge("D", "E", weight=2.0)
        g.add_edge("D", "F", weight=6.0)
        g.add_edge("E", "F", weight=3.0)

        dist, path = GraphAlgorithms.dijkstra(g, "A", "F")
        assert dist == 14.0
        assert path == ["A", "B", "D", "E", "F"]

    def test_all_pairs(self):
        """Compute shortest paths from source to all nodes."""
        g = Graph()
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)
        g.add_edge("A", "C", weight=4.0)

        distances = GraphAlgorithms.dijkstra_all(g, "A")
        assert distances["A"] == 0.0
        assert distances["B"] == 1.0
        assert distances["C"] == 3.0
