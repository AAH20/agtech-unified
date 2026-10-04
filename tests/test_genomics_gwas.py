"""Test GWAS (Genome-Wide Association Study) support."""

import pytest

from src.genomics.gwas import (
    GWASAnalyzer,
    GWASResult,
    run_gwas,
)


class TestGWAS:
    """Tests for GWAS analysis."""

    def setup_method(self):
        self.analyzer = GWASAnalyzer()

    def test_empty_genotypes_raises(self):
        """Empty genotype data raises ValueError."""
        with pytest.raises(ValueError, match="Genotype data cannot be empty"):
            self.analyzer.analyze([], [])

    def test_empty_phenotypes_raises(self):
        """Empty phenotype data raises ValueError."""
        with pytest.raises(ValueError, match="Phenotype data cannot be empty"):
            self.analyzer.analyze(["A/A", "A/T", "T/T"], [])

    def test_mismatched_lengths_raises(self):
        """Mismatched genotype/phenotype lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            self.analyzer.analyze(["A/A", "A/T"], [1.0])

    def test_result_type(self):
        """Analysis returns GWASResult."""
        genotypes = ["A/A"] * 20 + ["A/T"] * 20 + ["T/T"] * 20
        phenotypes = [1.0] * 20 + [2.0] * 20 + [3.0] * 20
        result = self.analyzer.analyze(genotypes, phenotypes)
        assert isinstance(result, GWASResult)

    def test_snp_fields(self):
        """SNP result has chrom, pos, ref, alt, p_value fields."""
        genotypes = ["A/A"] * 20 + ["A/T"] * 20 + ["T/T"] * 20
        phenotypes = [1.0] * 20 + [2.0] * 20 + [3.0] * 20
        result = self.analyzer.analyze(genotypes, phenotypes)
        if result.significant_snps:
            snp = result.significant_snps[0]
            assert hasattr(snp, "chrom")
            assert hasattr(snp, "pos")
            assert hasattr(snp, "ref")
            assert hasattr(snp, "alt")
            assert hasattr(snp, "p_value")

    def test_p_value_range(self):
        """P-values are between 0 and 1."""
        genotypes = ["A/A"] * 20 + ["A/T"] * 20 + ["T/T"] * 20
        phenotypes = [1.0] * 20 + [2.0] * 20 + [3.0] * 20
        result = self.analyzer.analyze(genotypes, phenotypes)
        for snp in result.significant_snps:
            assert 0.0 <= snp.p_value <= 1.0

    def test_significant_snps_detected(self):
        """Strong association produces significant SNPs."""
        genotypes = ["A/A"] * 30 + ["A/T"] * 30 + ["T/T"] * 30
        phenotypes = [1.0] * 30 + [2.0] * 30 + [3.0] * 30
        result = self.analyzer.analyze(genotypes, phenotypes)
        assert len(result.significant_snps) > 0

    def test_no_association(self):
        """Random phenotypes produce no significant SNPs."""
        import random

        random.seed(42)
        genotypes = ["A/A"] * 30 + ["A/T"] * 30 + ["T/T"] * 30
        phenotypes = [random.gauss(0, 1) for _ in range(90)]
        result = self.analyzer.analyze(genotypes, phenotypes)
        # With random data, should have no significant SNPs
        assert len(result.significant_snps) == 0

    def test_effect_size_range(self):
        """Effect sizes are finite numbers."""
        genotypes = ["A/A"] * 20 + ["A/T"] * 20 + ["T/T"] * 20
        phenotypes = [1.0] * 20 + [2.0] * 20 + [3.0] * 20
        result = self.analyzer.analyze(genotypes, phenotypes)
        for snp in result.significant_snps:
            assert -100.0 <= snp.effect_size <= 100.0

    def test_confidence_interval(self):
        """SNP has confidence interval for effect size."""
        genotypes = ["A/A"] * 20 + ["A/T"] * 20 + ["T/T"] * 20
        phenotypes = [1.0] * 20 + [2.0] * 20 + [3.0] * 20
        result = self.analyzer.analyze(genotypes, phenotypes)
        for snp in result.significant_snps:
            assert hasattr(snp, "ci_lower")
            assert hasattr(snp, "ci_upper")
            assert snp.ci_lower <= snp.ci_upper

    def test_manhattan_data(self):
        """Manhattan plot data is generated."""
        genotypes = ["A/A"] * 20 + ["A/T"] * 20 + ["T/T"] * 20
        phenotypes = [1.0] * 20 + [2.0] * 20 + [3.0] * 20
        result = self.analyzer.analyze(genotypes, phenotypes)
        assert hasattr(result, "manhattan_data")
        assert len(result.manhattan_data) > 0

    def test_convenience_function(self):
        """run_gwas convenience function works."""
        genotypes = ["A/A"] * 20 + ["A/T"] * 20 + ["T/T"] * 20
        phenotypes = [1.0] * 20 + [2.0] * 20 + [3.0] * 20
        result = run_gwas(genotypes, phenotypes)
        assert isinstance(result, GWASResult)

    def test_custom_significance_threshold(self):
        """Custom significance threshold works."""
        genotypes = ["A/A"] * 30 + ["A/T"] * 30 + ["T/T"] * 30
        phenotypes = [1.0] * 30 + [2.0] * 30 + [3.0] * 30
        strict = GWASAnalyzer(significance_threshold=1e-10)
        result = strict.analyze(genotypes, phenotypes)
        assert isinstance(result, GWASResult)

    def test_genotype_count(self):
        """Genotype count matches input."""
        genotypes = ["A/A"] * 10 + ["A/T"] * 10 + ["T/T"] * 10
        phenotypes = [1.0] * 10 + [2.0] * 10 + [3.0] * 10
        result = self.analyzer.analyze(genotypes, phenotypes)
        assert result.num_samples == 30

    def test_odds_ratio_positive(self):
        """Odds ratio is positive."""
        genotypes = ["A/A"] * 20 + ["A/T"] * 20 + ["T/T"] * 20
        phenotypes = [1.0] * 20 + [2.0] * 20 + [3.0] * 20
        result = self.analyzer.analyze(genotypes, phenotypes)
        for snp in result.significant_snps:
            assert snp.odds_ratio > 0
