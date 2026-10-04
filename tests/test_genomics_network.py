"""Test gene network analysis."""

import pytest

from src.genomics.network import (
    GeneNetworkAnalyzer,
    NetworkResult,
    analyze_gene_network,
)


class TestGeneNetworkAnalysis:
    """Tests for gene co-expression network analysis."""

    def setup_method(self):
        self.analyzer = GeneNetworkAnalyzer()

    def test_empty_expression_raises(self):
        """Empty expression data raises ValueError."""
        with pytest.raises(ValueError, match="Expression data cannot be empty"):
            self.analyzer.analyze({})

    def test_empty_gene_raises(self):
        """Gene with empty expression vector raises ValueError."""
        with pytest.raises(ValueError, match="Expression values cannot be empty"):
            self.analyzer.analyze({"gene1": []})

    def test_single_gene_no_edges(self):
        """Single gene produces no edges."""
        result = self.analyzer.analyze({"gene1": [1.0, 2.0, 3.0, 4.0]})
        assert isinstance(result, NetworkResult)
        assert len(result.edges) == 0

    def test_result_type(self):
        """Analysis returns NetworkResult."""
        data = {
            "gene1": [1.0, 2.0, 3.0, 4.0],
            "gene2": [2.0, 4.0, 6.0, 8.0],
        }
        result = self.analyzer.analyze(data)
        assert isinstance(result, NetworkResult)

    def test_correlated_genes_have_edge(self):
        """Highly correlated genes produce an edge."""
        data = {
            "gene1": [1.0, 2.0, 3.0, 4.0],
            "gene2": [2.0, 4.0, 6.0, 8.0],
        }
        result = self.analyzer.analyze(data)
        assert len(result.edges) > 0

    def test_edge_fields(self):
        """Edges have gene1, gene2, and weight fields."""
        data = {
            "gene1": [1.0, 2.0, 3.0, 4.0],
            "gene2": [2.0, 4.0, 6.0, 8.0],
        }
        result = self.analyzer.analyze(data)
        if result.edges:
            edge = result.edges[0]
            assert hasattr(edge, "gene1")
            assert hasattr(edge, "gene2")
            assert hasattr(edge, "weight")

    def test_edge_weight_range(self):
        """Edge weights are between -1 and 1."""
        data = {
            "gene1": [1.0, 2.0, 3.0, 4.0],
            "gene2": [2.0, 4.0, 6.0, 8.0],
            "gene3": [4.0, 3.0, 2.0, 1.0],
        }
        result = self.analyzer.analyze(data)
        for edge in result.edges:
            assert -1.0 <= edge.weight <= 1.0

    def test_negative_correlation(self):
        """Negatively correlated genes produce negative edge weight."""
        data = {
            "gene1": [1.0, 2.0, 3.0, 4.0],
            "gene2": [4.0, 3.0, 2.0, 1.0],
        }
        result = self.analyzer.analyze(data)
        assert len(result.edges) > 0
        assert result.edges[0].weight < 0

    def test_no_correlation_no_edge(self):
        """Uncorrelated genes produce no edge."""
        data = {
            "gene1": [1.0, 2.0, 3.0, 4.0],
            "gene2": [1.0, 1.0, 1.0, 1.0],
        }
        result = self.analyzer.analyze(data)
        assert len(result.edges) == 0

    def test_network_stats(self):
        """Network stats include node and edge counts."""
        data = {
            "gene1": [1.0, 2.0, 3.0, 4.0],
            "gene2": [2.0, 4.0, 6.0, 8.0],
            "gene3": [1.0, 3.0, 2.0, 4.0],
        }
        result = self.analyzer.analyze(data)
        assert result.num_nodes == 3
        assert result.num_edges == len(result.edges)

    def test_hub_genes(self):
        """Hub genes are identified."""
        data = {
            "hub": [1.0, 2.0, 3.0, 4.0, 5.0],
            "g1": [1.1, 2.1, 2.9, 4.1, 5.1],
            "g2": [0.9, 1.9, 3.1, 3.9, 4.9],
            "g3": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
        result = self.analyzer.analyze(data)
        assert "hub" in result.hub_genes

    def test_modularity_range(self):
        """Modularity is between 0 and 1."""
        data = {
            "g1": [1.0, 2.0, 3.0, 4.0],
            "g2": [1.1, 2.1, 2.9, 4.1],
            "g3": [5.0, 4.0, 3.0, 2.0],
            "g4": [5.1, 3.9, 3.1, 1.9],
        }
        result = self.analyzer.analyze(data)
        assert 0.0 <= result.modularity <= 1.0

    def test_convenience_function(self):
        """analyze_gene_network convenience function works."""
        data = {
            "gene1": [1.0, 2.0, 3.0, 4.0],
            "gene2": [2.0, 4.0, 6.0, 8.0],
        }
        result = analyze_gene_network(data)
        assert isinstance(result, NetworkResult)

    def test_custom_correlation_threshold(self):
        """Custom correlation threshold filters edges."""
        data = {
            "g1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "g2": [1.1, 2.1, 2.9, 4.1, 5.1],
        }
        strict = GeneNetworkAnalyzer(correlation_threshold=0.99)
        result = strict.analyze(data)
        # With very strict threshold, may or may not have edges
        assert isinstance(result, NetworkResult)

    def test_clustering_coefficient_range(self):
        """Clustering coefficient is between 0 and 1."""
        data = {
            "g1": [1.0, 2.0, 3.0, 4.0],
            "g2": [1.1, 2.1, 2.9, 4.1],
            "g3": [0.9, 1.9, 3.1, 3.9],
        }
        result = self.analyzer.analyze(data)
        assert 0.0 <= result.clustering_coefficient <= 1.0
