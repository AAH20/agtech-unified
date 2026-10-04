"""Tests for GO enrichment analysis (hypergeometric over-representation)."""

import pytest

from src.genomics.go_enrichment import (
    GOEnrichmentAnalyzer,
    GOEnrichmentResult,
    benjamini_hochberg,
    bonferroni,
    hypergeometric_pvalue,
)


class TestHypergeometricPvalue:
    """Tests for the hypergeometric survival function."""

    def test_overrepresentation_significant(self):
        """Strong over-representation gives small p-value."""
        # 8 of 10 query genes in term; 10 of 100 background genes in term
        p = hypergeometric_pvalue(k=8, K=10, n=10, N=100)
        assert p < 0.001

    def test_no_overrepresentation(self):
        """Proportional representation gives p-value near 1."""
        # 5 of 10 query genes in term; 50 of 100 background genes in term
        p = hypergeometric_pvalue(k=5, K=50, n=10, N=100)
        assert p > 0.5

    def test_extreme_overrepresentation(self):
        """All query genes in a rare term is highly significant."""
        p = hypergeometric_pvalue(k=10, K=10, n=10, N=1000)
        assert p < 1e-10

    def test_pvalue_range(self):
        """P-value is always in [0, 1]."""
        for k in range(0, 11):
            p = hypergeometric_pvalue(k=k, K=20, n=10, N=100)
            assert 0.0 <= p <= 1.0

    def test_k_greater_than_n_raises(self):
        """k cannot exceed n (query set size)."""
        with pytest.raises(ValueError):
            hypergeometric_pvalue(k=15, K=20, n=10, N=100)

    def test_background_exceeds_total_raises(self):
        """K cannot exceed N (background size)."""
        with pytest.raises(ValueError):
            hypergeometric_pvalue(k=5, K=200, n=10, N=100)

    def test_zero_k(self):
        """k=0 gives p-value of 1.0 (no over-representation)."""
        p = hypergeometric_pvalue(k=0, K=50, n=10, N=100)
        assert p == 1.0

    def test_underrepresentation_not_significant(self):
        """Under-representation gives large p-value (over-representation test)."""
        p = hypergeometric_pvalue(k=1, K=50, n=10, N=100)
        assert p > 0.9


class TestMultipleTestingCorrection:
    """Tests for multiple testing corrections."""

    def test_bonferroni(self):
        """Bonferroni multiplies by number of tests, capped at 1."""
        pvals = [0.01, 0.02, 0.03]
        adjusted = bonferroni(pvals)
        assert adjusted[0] == pytest.approx(0.03)
        assert adjusted[1] == pytest.approx(0.06)
        assert adjusted[2] == pytest.approx(0.09)

    def test_bonferroni_caps_at_one(self):
        """Bonferroni adjusted p-values never exceed 1."""
        pvals = [0.5, 0.6, 0.7]
        adjusted = bonferroni(pvals)
        assert all(p <= 1.0 for p in adjusted)

    def test_benjamini_hochberg(self):
        """BH correction produces monotonically non-decreasing adjusted p-values."""
        pvals = [0.005, 0.01, 0.02, 0.03, 0.04]
        adjusted = benjamini_hochberg(pvals)
        for i in range(1, len(adjusted)):
            assert adjusted[i] >= adjusted[i - 1]

    def test_bh_less_conservative_than_bonferroni(self):
        """BH adjusted p-values are <= Bonferroni adjusted p-values."""
        pvals = [0.01, 0.02, 0.03, 0.04, 0.05]
        bh = benjamini_hochberg(pvals)
        bonf = bonferroni(pvals)
        for b, f in zip(bh, bonf):
            assert b <= f

    def test_empty_input(self):
        """Empty input returns empty list."""
        assert bonferroni([]) == []
        assert benjamini_hochberg([]) == []


class TestGOEnrichmentAnalyzer:
    """Tests for the GO enrichment analyzer."""

    def setup_method(self):
        """Set up test data: background and query gene sets with GO annotations."""
        # Background: 100 genes, various GO annotations
        self.background = {}
        for i in range(100):
            # 20 genes annotated to GO:0008150 (biological_process)
            if i < 20:
                self.background[f"gene_{i}"] = {"GO:0008150", "GO:0005575"}
            # 10 genes annotated to GO:0003674 (molecular_function)
            elif i < 30:
                self.background[f"gene_{i}"] = {"GO:0003674"}
            else:
                self.background[f"gene_{i}"] = {"GO:0005575"}

        # Query: 10 genes, 8 of which are in GO:0008150
        self.query_genes = [f"gene_{i}" for i in range(8)] + ["gene_50", "gene_51"]

    def test_result_type(self):
        """Analysis returns list of GOEnrichmentResult."""
        analyzer = GOEnrichmentAnalyzer(self.background)
        results = analyzer.enrich(self.query_genes)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, GOEnrichmentResult)

    def test_enriched_term_detected(self):
        """Over-represented GO term is detected."""
        analyzer = GOEnrichmentAnalyzer(self.background)
        results = analyzer.enrich(self.query_genes)
        go_terms = {r.go_id for r in results}
        assert "GO:0008150" in go_terms

    def test_pvalue_significant_for_enriched(self):
        """Enriched term has significant p-value."""
        analyzer = GOEnrichmentAnalyzer(self.background)
        results = analyzer.enrich(self.query_genes)
        for r in results:
            if r.go_id == "GO:0008150":
                assert r.p_value < 0.01

    def test_enrichment_ratio(self):
        """Enrichment ratio is computed correctly."""
        analyzer = GOEnrichmentAnalyzer(self.background)
        results = analyzer.enrich(self.query_genes)
        for r in results:
            if r.go_id == "GO:0008150":
                # (8/10) / (20/100) = 0.8 / 0.2 = 4.0
                assert r.enrichment_ratio == pytest.approx(4.0)

    def test_gene_count(self):
        """Gene count matches query genes annotated to term."""
        analyzer = GOEnrichmentAnalyzer(self.background)
        results = analyzer.enrich(self.query_genes)
        for r in results:
            if r.go_id == "GO:0008150":
                assert r.query_gene_count == 8

    def test_background_gene_count(self):
        """Background gene count matches background genes annotated to term."""
        analyzer = GOEnrichmentAnalyzer(self.background)
        results = analyzer.enrich(self.query_genes)
        for r in results:
            if r.go_id == "GO:0008150":
                assert r.background_gene_count == 20

    def test_empty_query_raises(self):
        """Empty query gene list raises ValueError."""
        analyzer = GOEnrichmentAnalyzer(self.background)
        with pytest.raises(ValueError, match="Query gene list cannot be empty"):
            analyzer.enrich([])

    def test_empty_background_raises(self):
        """Empty background raises ValueError."""
        with pytest.raises(ValueError, match="Background cannot be empty"):
            GOEnrichmentAnalyzer({})

    def test_query_genes_not_in_background(self):
        """Query genes not in background are ignored."""
        analyzer = GOEnrichmentAnalyzer(self.background)
        query = [f"gene_{i}" for i in range(8)] + ["unknown_1", "unknown_2"]
        results = analyzer.enrich(query)
        for r in results:
            if r.go_id == "GO:0008150":
                assert r.query_gene_count == 8

    def test_min_gene_filter(self):
        """Terms with fewer than min_genes query genes are excluded."""
        analyzer = GOEnrichmentAnalyzer(self.background)
        results = analyzer.enrich(self.query_genes, min_genes=9)
        go_terms = {r.go_id for r in results}
        assert "GO:0008150" not in go_terms

    def test_significance_threshold(self):
        """Only terms below threshold are returned as significant."""
        analyzer = GOEnrichmentAnalyzer(self.background)
        results = analyzer.enrich(self.query_genes, p_threshold=0.001)
        for r in results:
            assert r.p_value < 0.001

    def test_multiple_testing_correction_applied(self):
        """Adjusted p-values are computed."""
        analyzer = GOEnrichmentAnalyzer(self.background)
        results = analyzer.enrich(self.query_genes, correction="bonferroni")
        for r in results:
            assert hasattr(r, "adjusted_p_value")
            assert r.adjusted_p_value >= r.p_value

    def test_bh_correction(self):
        """BH correction is applied when specified."""
        analyzer = GOEnrichmentAnalyzer(self.background)
        results = analyzer.enrich(self.query_genes, correction="bh")
        for r in results:
            assert hasattr(r, "adjusted_p_value")

    def test_no_correction(self):
        """No correction leaves adjusted_p_value equal to p_value."""
        analyzer = GOEnrichmentAnalyzer(self.background)
        results = analyzer.enrich(self.query_genes, correction=None)
        for r in results:
            assert r.adjusted_p_value == r.p_value

    def test_results_sorted_by_pvalue(self):
        """Results are sorted by p-value ascending."""
        analyzer = GOEnrichmentAnalyzer(self.background)
        results = analyzer.enrich(self.query_genes)
        for i in range(1, len(results)):
            assert results[i].p_value >= results[i - 1].p_value

    def test_all_query_genes_in_background(self):
        """All query genes in background are handled."""
        background = {f"g{i}": {"GO:001"} for i in range(50)}
        for i in range(50, 100):
            background[f"g{i}"] = {"GO:002"}
        query = [f"g{i}" for i in range(10)]
        analyzer = GOEnrichmentAnalyzer(background)
        results = analyzer.enrich(query)
        assert len(results) > 0

    def test_enrichment_ratio_greater_than_one_for_enriched(self):
        """Enriched terms have enrichment ratio > 1."""
        analyzer = GOEnrichmentAnalyzer(self.background)
        results = analyzer.enrich(self.query_genes)
        for r in results:
            if r.p_value < 0.05:
                assert r.enrichment_ratio > 1.0
