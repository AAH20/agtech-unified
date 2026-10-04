"""Graph algorithms for agricultural network analysis.

Provides PageRank for importance ranking and community detection
for identifying clusters in agricultural knowledge graphs,
supply chains, and sensor networks.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Graph:
    """Directed weighted graph for agricultural networks."""

    nodes: Set[str] = field(default_factory=set)
    edges: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    _reverse_edges: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))

    def add_node(self, node_id: str) -> None:
        """Add a node to the graph."""
        self.nodes.add(node_id)

    def add_edge(self, source: str, target: str, weight: float = 1.0) -> None:
        """Add a directed edge to the graph."""
        if source not in self.nodes:
            self.add_node(source)
        if target not in self.nodes:
            self.add_node(target)
        self.edges[source][target] = weight
        self._reverse_edges[target][source] = weight

    def remove_edge(self, source: str, target: str) -> None:
        """Remove an edge from the graph."""
        self.edges[source].pop(target, None)
        self._reverse_edges[target].pop(source, None)

    def get_neighbors(self, node_id: str) -> List[str]:
        """Get outgoing neighbors of a node."""
        return list(self.edges.get(node_id, {}).keys())

    def get_in_neighbors(self, node_id: str) -> List[str]:
        """Get incoming neighbors of a node."""
        return list(self._reverse_edges.get(node_id, {}).keys())

    def out_degree(self, node_id: str) -> int:
        """Get out-degree of a node."""
        return len(self.edges.get(node_id, {}))

    def in_degree(self, node_id: str) -> int:
        """Get in-degree of a node."""
        return len(self._reverse_edges.get(node_id, {}))

    def edge_weight(self, source: str, target: str) -> float:
        """Get weight of an edge, 0 if not present."""
        return self.edges.get(source, {}).get(target, 0.0)

    def to_adjacency_matrix(self) -> Tuple[List[str], List[List[float]]]:
        """Convert to adjacency matrix representation.

        Returns:
            Tuple of (node_list, matrix) where matrix[i][j] is the
            weight from node i to node j.
        """
        node_list = sorted(self.nodes)
        n = len(node_list)
        node_idx = {node: i for i, node in enumerate(node_list)}
        matrix = [[0.0] * n for _ in range(n)]
        for src, targets in self.edges.items():
            for tgt, w in targets.items():
                matrix[node_idx[src]][node_idx[tgt]] = w
        return node_list, matrix


@dataclass
class PageRankResult:
    """Result of PageRank computation."""

    scores: Dict[str, float]
    iterations: int
    converged: bool
    damping_factor: float


@dataclass
class CommunityDetectionResult:
    """Result of community detection."""

    communities: Dict[str, int]  # node_id -> community_id
    num_communities: int
    modularity: float


class GraphAlgorithms:
    """Graph algorithms for agricultural network analysis."""

    @staticmethod
    def pagerank(
        graph: Graph,
        damping_factor: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> PageRankResult:
        """Compute PageRank scores for all nodes.

        Uses the standard PageRank algorithm with damping factor.
        Handles dangling nodes (nodes with no out-edges) by
        redistributing their rank uniformly.

        Args:
            graph: The graph to analyze.
            damping_factor: Probability of following a link (0-1).
            max_iterations: Maximum iterations.
            tolerance: Convergence threshold.

        Returns:
            PageRankResult with scores and convergence info.
        """
        if not graph.nodes:
            return PageRankResult(
                scores={}, iterations=0, converged=True, damping_factor=damping_factor
            )

        nodes = list(graph.nodes)
        n = len(nodes)
        {node: i for i, node in enumerate(nodes)}

        # Initialize scores uniformly
        scores = {node: 1.0 / n for node in nodes}

        # Precompute out-degree weights
        out_weights: Dict[str, float] = {}
        for node in nodes:
            total_w = sum(graph.edges.get(node, {}).values())
            out_weights[node] = total_w if total_w > 0 else 0.0

        converged = False
        iteration = 0

        for iteration in range(1, max_iterations + 1):
            new_scores: Dict[str, float] = {}
            dangling_sum = 0.0

            # Calculate dangling node contribution
            for node in nodes:
                if out_weights[node] == 0.0:
                    dangling_sum += scores[node]

            # Distribute dangling mass uniformly
            dangling_contrib = dangling_sum / n

            for node in nodes:
                # Base score from damping
                rank = (1.0 - damping_factor) / n + damping_factor * dangling_contrib

                # Add contributions from incoming edges
                for in_node in graph.get_in_neighbors(node):
                    w = graph.edge_weight(in_node, node)
                    if out_weights[in_node] > 0:
                        rank += damping_factor * scores[in_node] * w / out_weights[in_node]

                new_scores[node] = rank

            # Check convergence
            diff = sum(abs(new_scores[node] - scores[node]) for node in nodes)
            scores = new_scores

            if diff < tolerance:
                converged = True
                break

        return PageRankResult(
            scores=scores,
            iterations=iteration,
            converged=converged,
            damping_factor=damping_factor,
        )

    @staticmethod
    def detect_communities(
        graph: Graph,
        resolution: float = 1.0,
        max_iterations: int = 100,
        seed: Optional[int] = None,
    ) -> CommunityDetectionResult:
        """Detect communities using the Louvain method.

        Optimizes modularity through greedy node movement and
        graph aggregation. Uses a simplified single-level approach
        suitable for small to medium agricultural networks.

        Args:
            graph: The graph to analyze.
            resolution: Resolution parameter (higher = more communities).
            max_iterations: Maximum iterations.
            seed: Random seed for reproducibility.

        Returns:
            CommunityDetectionResult with community assignments.
        """
        if not graph.nodes:
            return CommunityDetectionResult(communities={}, num_communities=0, modularity=0.0)

        if seed is not None:
            random.seed(seed)

        nodes = list(graph.nodes)
        n = len(nodes)

        # Initialize: each node in its own community
        communities: Dict[str, int] = {node: i for i, node in enumerate(nodes)}

        # Precompute node weights (total edge weight per node)
        node_weights: Dict[str, float] = {}
        total_weight = 0.0
        for node in nodes:
            w = sum(graph.edges.get(node, {}).values()) + sum(
                graph._reverse_edges.get(node, {}).values()
            )
            node_weights[node] = w
            total_weight += w

        if total_weight == 0:
            # No edges: each node is its own community
            return CommunityDetectionResult(
                communities=communities,
                num_communities=n,
                modularity=0.0,
            )

        # Greedy modularity optimization
        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # Randomize node order for fairness
            order = nodes[:]
            random.shuffle(order)

            for node in order:
                current_comm = communities[node]

                # Calculate gain for moving to each neighboring community
                comm_gains: Dict[int, float] = defaultdict(float)

                for neighbor in graph.get_neighbors(node):
                    w = graph.edge_weight(node, neighbor)
                    comm_gains[communities[neighbor]] += w

                for neighbor in graph.get_in_neighbors(node):
                    w = graph.edge_weight(neighbor, node)
                    comm_gains[communities[neighbor]] += w

                # Find best community
                best_comm = current_comm
                best_gain = 0.0

                for comm, weight_to_comm in comm_gains.items():
                    if comm == current_comm:
                        continue
                    # Simplified modularity gain
                    gain = weight_to_comm - resolution * node_weights[node] * sum(
                        node_weights[n2] for n2 in nodes if communities[n2] == comm
                    ) / (2.0 * total_weight)
                    if gain > best_gain:
                        best_gain = gain
                        best_comm = comm

                if best_comm != current_comm:
                    communities[node] = best_comm
                    improved = True

        # Renumber communities to 0..k-1
        unique_comms = sorted(set(communities.values()))
        comm_map = {old: new for new, old in enumerate(unique_comms)}
        communities = {node: comm_map[comm] for node, comm in communities.items()}

        num_communities = len(unique_comms)
        modularity = GraphAlgorithms._modularity(graph, communities, total_weight)

        return CommunityDetectionResult(
            communities=communities,
            num_communities=num_communities,
            modularity=modularity,
        )

    @staticmethod
    def _modularity(
        graph: Graph,
        communities: Dict[str, int],
        total_weight: float,
    ) -> float:
        """Calculate modularity of a community partition.

        Q = (1/2m) * Σ [A_ij - k_i*k_j/2m] * δ(c_i, c_j)
        """
        if total_weight == 0:
            return 0.0

        q = 0.0
        nodes = list(graph.nodes)

        for i in nodes:
            for j in nodes:
                a_ij = graph.edge_weight(i, j)
                k_i = sum(graph.edges.get(i, {}).values()) + sum(
                    graph._reverse_edges.get(i, {}).values()
                )
                k_j = sum(graph.edges.get(j, {}).values()) + sum(
                    graph._reverse_edges.get(j, {}).values()
                )
                if communities[i] == communities[j]:
                    q += a_ij - (k_i * k_j) / (2.0 * total_weight)

        return q / (2.0 * total_weight)

    @staticmethod
    def betweenness_centrality(graph: Graph, normalized: bool = True) -> Dict[str, float]:
        """Calculate betweenness centrality using Brandes' algorithm.

        Args:
            graph: The graph to analyze.
            normalized: Whether to normalize scores.

        Returns:
            Dict mapping node IDs to betweenness centrality scores.
        """
        centrality: Dict[str, float] = {node: 0.0 for node in graph.nodes}

        for source in graph.nodes:
            # BFS
            queue = [source]
            visited = {source}
            distance: Dict[str, int] = {source: 0}
            predecessors: Dict[str, List[str]] = defaultdict(list)
            sigma: Dict[str, float] = {source: 1.0}

            while queue:
                current = queue.pop(0)
                for neighbor in graph.get_neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distance[neighbor] = distance[current] + 1
                        queue.append(neighbor)
                    if distance.get(neighbor) == distance[current] + 1:
                        sigma[neighbor] = sigma.get(neighbor, 0.0) + sigma[current]
                        predecessors[neighbor].append(current)

            # Dependency accumulation
            delta: Dict[str, float] = {node: 0.0 for node in graph.nodes}
            nodes_by_dist = sorted(visited, key=lambda n: distance[n], reverse=True)
            for node in nodes_by_dist:
                for pred in predecessors[node]:
                    delta[pred] += (sigma[pred] / sigma[node]) * (1.0 + delta[node])
                if node != source:
                    centrality[node] += delta[node]

        if normalized and len(graph.nodes) > 2:
            scale = 1.0 / ((len(graph.nodes) - 1) * (len(graph.nodes) - 2))
            for node in centrality:
                centrality[node] *= scale

        return centrality
