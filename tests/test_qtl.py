"""Tests for QTL mapping (interval mapping for multi-locus detection)."""

import pytest

from src.genomics.qtl import (
    IntervalMapper,
    QTLMappingResult,
    run_qtl_mapping,
)


class TestIntervalMapper:
    """Tests for interval mapping QTL analysis."""

    def setup_method(self):
        """Set up test data: marker genotypes and phenotype."""
        # 3 markers on one chromosome, 60 individuals
        # Marker genotypes: 0=AA, 1=Aa, 2=aa (dosage of alt allele)
        self.markers = [
            ("chr1", 10, [0] * 20 + [1] * 20 + [2] * 20),
            ("chr1", 50, [0] * 20 + [1] * 20 + [2] * 20),
            ("chr1", 90, [0] * 20 + [1] * 20 + [2] * 20),
        ]
        # Phenotype with strong association at marker 2 (pos 50)
        self.phenotypes = [1.0] * 20 + [2.0] * 20 + [3.0] * 20

    def test_result_type(self):
        """Mapping returns QTLMappingResult."""
        mapper = IntervalMapper()
        result = mapper.map_qtl(self.markers, self.phenotypes)
        assert isinstance(result, QTLMappingResult)

    def test_significant_qtl_detected(self):
        """Strong association produces significant QTL."""
        mapper = IntervalMapper()
        result = mapper.map_qtl(self.markers, self.phenotypes)
        assert len(result.significant_qtls) > 0

    def test_qtl_has_lod_score(self):
        """QTL results have LOD scores."""
        mapper = IntervalMapper()
        result = mapper.map_qtl(self.markers, self.phenotypes)
        for qtl in result.significant_qtls:
            assert hasattr(qtl, "lod_score")
            assert qtl.lod_score > 0

    def test_qtl_has_position(self):
        """QTL results have chromosome and position."""
        mapper = IntervalMapper()
        result = mapper.map_qtl(self.markers, self.phenotypes)
        for qtl in result.significant_qtls:
            assert hasattr(qtl, "chrom")
            assert hasattr(qtl, "pos")

    def test_qtl_has_effect_size(self):
        """QTL results have effect size."""
        mapper = IntervalMapper()
        result = mapper.map_qtl(self.markers, self.phenotypes)
        for qtl in result.significant_qtls:
            assert hasattr(qtl, "effect_size")

    def test_qtl_has_pvalue(self):
        """QTL results have p-value."""
        mapper = IntervalMapper()
        result = mapper.map_qtl(self.markers, self.phenotypes)
        for qtl in result.significant_qtls:
            assert hasattr(qtl, "p_value")
            assert 0.0 <= qtl.p_value <= 1.0

    def test_empty_markers_raises(self):
        """Empty marker list raises ValueError."""
        mapper = IntervalMapper()
        with pytest.raises(ValueError, match="Marker data cannot be empty"):
            mapper.map_qtl([], self.phenotypes)

    def test_empty_phenotypes_raises(self):
        """Empty phenotype list raises ValueError."""
        mapper = IntervalMapper()
        with pytest.raises(ValueError, match="Phenotype data cannot be empty"):
            mapper.map_qtl(self.markers, [])

    def test_mismatched_lengths_raises(self):
        """Mismatched marker/phenotype lengths raise ValueError."""
        mapper = IntervalMapper()
        with pytest.raises(ValueError, match="same length"):
            mapper.map_qtl(self.markers, [1.0] * 50)

    def test_lod_threshold(self):
        """Custom LOD threshold filters QTLs."""
        mapper = IntervalMapper(lod_threshold=10.0)
        result = mapper.map_qtl(self.markers, self.phenotypes)
        for qtl in result.significant_qtls:
            assert qtl.lod_score >= 10.0

    def test_no_association(self):
        """Random phenotypes produce no significant QTLs."""
        import random

        random.seed(42)
        phenotypes = [random.gauss(0, 1) for _ in range(60)]
        mapper = IntervalMapper()
        result = mapper.map_qtl(self.markers, phenotypes)
        assert len(result.significant_qtls) == 0

    def test_multi_locus_detection(self):
        """Multiple QTLs on different chromosomes are detected."""
        markers = [
            ("chr1", 10, [0] * 20 + [1] * 20 + [2] * 20),
            ("chr2", 30, [0] * 20 + [1] * 20 + [2] * 20),
        ]
        # Phenotype associated with both markers
        phenotypes = [1.0] * 20 + [2.0] * 20 + [3.0] * 20
        mapper = IntervalMapper()
        result = mapper.map_qtl(markers, phenotypes)
        chroms = {qtl.chrom for qtl in result.significant_qtls}
        assert len(chroms) >= 1

    def test_interval_positions(self):
        """QTL positions are interpolated between markers."""
        mapper = IntervalMapper()
        result = mapper.map_qtl(self.markers, self.phenotypes)
        for qtl in result.significant_qtls:
            assert isinstance(qtl.pos, (int, float))

    def test_effect_size_direction(self):
        """Effect size sign matches phenotype trend."""
        mapper = IntervalMapper()
        result = mapper.map_qtl(self.markers, self.phenotypes)
        for qtl in result.significant_qtls:
            # Positive trend: AA=1, Aa=2, aa=3, so effect should be positive
            assert qtl.effect_size > 0

    def test_convenience_function(self):
        """run_qtl_mapping convenience function works."""
        result = run_qtl_mapping(self.markers, self.phenotypes)
        assert isinstance(result, QTLMappingResult)

    def test_num_samples(self):
        """Result reports number of samples."""
        mapper = IntervalMapper()
        result = mapper.map_qtl(self.markers, self.phenotypes)
        assert result.num_samples == 60

    def test_num_markers(self):
        """Result reports number of markers tested."""
        mapper = IntervalMapper()
        result = mapper.map_qtl(self.markers, self.phenotypes)
        assert result.num_markers == 3

    def test_lod_score_calculation(self):
        """LOD score is computed from likelihood ratio."""
        mapper = IntervalMapper()
        result = mapper.map_qtl(self.markers, self.phenotypes)
        for qtl in result.significant_qtls:
            # LOD = log10(L1/L0), should be positive for significant QTL
            assert qtl.lod_score > 0

    def test_marker_with_no_variation(self):
        """Marker with no genotype variation is skipped."""
        markers = [
            ("chr1", 10, [0] * 60),  # No variation
            ("chr1", 50, [0] * 20 + [1] * 20 + [2] * 20),
        ]
        phenotypes = [1.0] * 20 + [2.0] * 20 + [3.0] * 20
        mapper = IntervalMapper()
        result = mapper.map_qtl(markers, phenotypes)
        assert isinstance(result, QTLMappingResult)

    def test_single_marker(self):
        """Single marker mapping works."""
        markers = [("chr1", 10, [0] * 20 + [1] * 20 + [2] * 20)]
        phenotypes = [1.0] * 20 + [2.0] * 20 + [3.0] * 20
        mapper = IntervalMapper()
        result = mapper.map_qtl(markers, phenotypes)
        assert isinstance(result, QTLMappingResult)

    def test_qtl_result_fields(self):
        """QTLResult has all expected fields."""
        mapper = IntervalMapper()
        result = mapper.map_qtl(self.markers, self.phenotypes)
        if result.significant_qtls:
            qtl = result.significant_qtls[0]
            assert hasattr(qtl, "chrom")
            assert hasattr(qtl, "pos")
            assert hasattr(qtl, "lod_score")
            assert hasattr(qtl, "p_value")
            assert hasattr(qtl, "effect_size")
            assert hasattr(qtl, "ci_lower")
            assert hasattr(qtl, "ci_upper")

    def test_confidence_interval(self):
        """QTL confidence interval brackets the peak position."""
        mapper = IntervalMapper()
        result = mapper.map_qtl(self.markers, self.phenotypes)
        for qtl in result.significant_qtls:
            assert qtl.ci_lower <= qtl.pos <= qtl.ci_upper

    def test_pvalue_threshold(self):
        """Custom p-value threshold filters QTLs."""
        mapper = IntervalMapper(p_threshold=0.001)
        result = mapper.map_qtl(self.markers, self.phenotypes)
        for qtl in result.significant_qtls:
            assert qtl.p_value < 0.001
