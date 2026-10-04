"""Gene co-expression network analysis for agricultural biotechnology.

Builds gene networks from expression data, identifies hub genes,
detects communities, and computes network statistics.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

logger = logging.getLogger(__name__)


@dataclass
class NetworkEdge:
    """An edge in the gene co-expression network."""

    gene1: str
    gene2: str
    weight: float


@dataclass
class NetworkResult:
    """Result of gene network analysis."""

    edges: List[NetworkEdge]
    num_nodes: int
    num_edges: int
    hub_genes: List[str]
    modularity: float
    clustering_coefficient: float


class GeneNetworkAnalyzer:
    """Analyzes gene co-expression networks from expression data."""

    def __init__(self, correlation_threshold: float = 0.7):
        self.correlation_threshold = correlation_threshold

    def analyze(self, expression_data: Dict[str, Sequence[float]]) -> NetworkResult:
        """Build and analyze a gene co-expression network.

        Args:
            expression_data: Dict mapping gene names to expression vectors.

        Returns:
            NetworkResult with edges, statistics, and hub genes.
        """
        if not expression_data:
            raise ValueError("Expression data cannot be empty")

        for gene, values in expression_data.items():
            if not values:
                raise ValueError("Expression values cannot be empty")

        genes = list(expression_data.keys())
        edges = self._build_edges(genes, expression_data)
        adjacency = self._build_adjacency(genes, edges)

        hub_genes = self._find_hubs(genes, adjacency)
        modularity = self._compute_modularity(genes, edges, adjacency)
        clustering = self._compute_clustering_coefficient(genes, adjacency)

        return NetworkResult(
            edges=edges,
            num_nodes=len(genes),
            num_edges=len(edges),
            hub_genes=hub_genes,
            modularity=modularity,
            clustering_coefficient=clustering,
        )

    def _build_edges(
        self,
        genes: List[str],
        expression_data: Dict[str, Sequence[float]],
    ) -> List[NetworkEdge]:
        """Build edges between correlated genes."""
        edges = []
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                g1, g2 = genes[i], genes[j]
                corr = self._pearson_correlation(expression_data[g1], expression_data[g2])
                if abs(corr) >= self.correlation_threshold:
                    edges.append(NetworkEdge(gene1=g1, gene2=g2, weight=corr))
        return edges

    def _build_adjacency(
        self,
        genes: List[str],
        edges: List[NetworkEdge],
    ) -> Dict[str, List[str]]:
        """Build adjacency list from edges."""
        adjacency: Dict[str, List[str]] = {g: [] for g in genes}
        for edge in edges:
            adjacency[edge.gene1].append(edge.gene2)
            adjacency[edge.gene2].append(edge.gene1)
        return adjacency

    def _find_hubs(
        self,
        genes: List[str],
        adjacency: Dict[str, List[str]],
    ) -> List[str]:
        """Identify hub genes (top 25% by degree, min degree 2)."""
        if not genes:
            return []

        degrees = {g: len(adjacency[g]) for g in genes}
        max_degree = max(degrees.values()) if degrees else 0

        if max_degree < 2:
            return []

        # Hubs are genes with degree >= 50% of max degree
        threshold = max(2, max_degree * 0.5)
        hubs = [g for g in genes if degrees[g] >= threshold]
        return sorted(hubs)

    def _compute_modularity(
        self,
        genes: List[str],
        edges: List[NetworkEdge],
        adjacency: Dict[str, List[str]],
    ) -> float:
        """Compute network modularity using a simple greedy approach.

        For simplicity, we use a label propagation approach to detect
        communities and then compute modularity.
        """
        if not edges or not genes:
            return 0.0

        # Simple community detection: connected components
        communities = self._connected_components(genes, adjacency)

        if len(communities) <= 1:
            return 0.0

        # Compute modularity
        m = len(edges)
        if m == 0:
            return 0.0

        degree = {g: len(adjacency[g]) for g in genes}

        q = 0.0
        for community in communities:
            community_set = set(community)
            # Edges within community
            lc = sum(1 for e in edges if e.gene1 in community_set and e.gene2 in community_set)
            # Sum of degrees in community
            dc = sum(degree[g] for g in community)

            q += lc / m - (dc / (2 * m)) ** 2

        return max(0.0, min(1.0, q))

    def _connected_components(
        self,
        genes: List[str],
        adjacency: Dict[str, List[str]],
    ) -> List[List[str]]:
        """Find connected components in the network."""
        visited = set()
        components = []

        for gene in genes:
            if gene in visited:
                continue
            component = []
            stack = [gene]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            components.append(component)

        return components

    def _compute_clustering_coefficient(
        self,
        genes: List[str],
        adjacency: Dict[str, List[str]],
    ) -> float:
        """Compute average clustering coefficient of the network."""
        if not genes:
            return 0.0

        coefficients = []
        for gene in genes:
            neighbors = adjacency[gene]
            k = len(neighbors)
            if k < 2:
                coefficients.append(0.0)
                continue

            # Count edges between neighbors
            neighbor_set = set(neighbors)
            edges_between = 0
            for n1 in neighbors:
                for n2 in adjacency[n1]:
                    if n2 in neighbor_set:
                        edges_between += 1
            # Each edge counted twice
            edges_between //= 2

            possible = k * (k - 1) / 2
            coefficients.append(edges_between / possible if possible > 0 else 0.0)

        return sum(coefficients) / len(coefficients) if coefficients else 0.0

    @staticmethod
    def _pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
        """Compute Pearson correlation coefficient between two vectors."""
        n = len(x)
        if n == 0 or n != len(y):
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)

        if var_x == 0 or var_y == 0:
            return 0.0

        return cov / math.sqrt(var_x * var_y)


def analyze_gene_network(
    expression_data: Dict[str, Sequence[float]],
    correlation_threshold: float = 0.7,
) -> NetworkResult:
    """Convenience function for gene network analysis.

    Args:
        expression_data: Dict mapping gene names to expression vectors.
        correlation_threshold: Minimum absolute correlation for an edge.

    Returns:
        NetworkResult with edges and statistics.
    """
    analyzer = GeneNetworkAnalyzer(correlation_threshold=correlation_threshold)
    return analyzer.analyze(expression_data)
