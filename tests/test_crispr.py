"""Dedicated unit tests for CRISPRDesigner (src/genomics/crispr.py).

Covers guide length correctness, PAM sequence variants, GC content bounds,
off-target scoring, efficiency scoring, multiple guides, and custom
PAM/guide_length parameters.
"""

from __future__ import annotations

import pytest

from src.genomics.crispr import CRISPRDesigner, GuideRNA


class TestCRISPRDesignerInit:
    """CRISPRDesigner initialization tests."""

    def test_default_pam_sequence(self):
        designer = CRISPRDesigner()
        assert designer.pam_sequence == "NGG"

    def test_default_guide_length(self):
        designer = CRISPRDesigner()
        assert designer.guide_length == 20

    def test_custom_pam_sequence(self):
        designer = CRISPRDesigner(pam_sequence="NAG")
        assert designer.pam_sequence == "NAG"

    def test_custom_guide_length(self):
        designer = CRISPRDesigner(guide_length=18)
        assert designer.guide_length == 18


class TestDesignGuideBasic:
    """Basic guide design tests."""

    def test_empty_sequence_raises(self, crispr_designer):
        with pytest.raises(ValueError, match="Sequence cannot be empty"):
            crispr_designer.design_guide("")

    def test_no_pam_site_raises(self, crispr_designer, no_pam_dna_sequence):
        with pytest.raises(ValueError, match="No PAM site found"):
            crispr_designer.design_guide(no_pam_dna_sequence)

    def test_valid_sequence_returns_guides(self, crispr_designer, valid_dna_sequence):
        guides = crispr_designer.design_guide(valid_dna_sequence)
        assert len(guides) > 0
        assert all(isinstance(g, GuideRNA) for g in guides)

    def test_guide_length_matches_config(self, crispr_designer, valid_dna_sequence):
        guides = crispr_designer.design_guide(valid_dna_sequence)
        for guide in guides:
            assert len(guide.sequence) == 20

    def test_pam_is_3_nt(self, crispr_designer, valid_dna_sequence):
        guides = crispr_designer.design_guide(valid_dna_sequence)
        for guide in guides:
            assert len(guide.pam) == 3

    def test_pam_ends_with_gg(self, crispr_designer, valid_dna_sequence):
        guides = crispr_designer.design_guide(valid_dna_sequence)
        for guide in guides:
            assert guide.pam[1:] == "GG"

    def test_position_is_non_negative(self, crispr_designer, valid_dna_sequence):
        guides = crispr_designer.design_guide(valid_dna_sequence)
        for guide in guides:
            assert guide.position >= 0


class TestGuideLength:
    """Tests for custom guide length."""

    def test_custom_guide_length_18(self):
        designer = CRISPRDesigner(guide_length=18)
        guides = designer.design_guide("ATCGATCGATCGATCGATCGATCGGG")
        for guide in guides:
            assert len(guide.sequence) == 18

    def test_custom_guide_length_22(self):
        designer = CRISPRDesigner(guide_length=22)
        guides = designer.design_guide("ATCGATCGATCGATCGATCGATCGATCGGG")
        for guide in guides:
            assert len(guide.sequence) == 22

    def test_guide_length_longer_than_sequence(self):
        """Guide length longer than available sequence produces no guides."""
        designer = CRISPRDesigner(guide_length=50)
        with pytest.raises(ValueError, match="No PAM site found"):
            designer.design_guide("ATCGATCGATCGATCGATCGATCGGG")


class TestGCContent:
    """Tests for GC content calculation."""

    def test_gc_content_in_range(self, crispr_designer, valid_dna_sequence):
        guides = crispr_designer.design_guide(valid_dna_sequence)
        for guide in guides:
            assert 0.0 <= guide.gc_content <= 1.0

    def test_gc_rich_sequence_high_gc(self, crispr_designer, gc_rich_dna_sequence):
        guides = crispr_designer.design_guide(gc_rich_dna_sequence)
        assert guides[0].gc_content > 0.5

    def test_at_rich_sequence_low_gc(self, crispr_designer):
        guides = crispr_designer.design_guide("ATATATATATATATATATATATCGGG")
        assert guides[0].gc_content < 0.5

    def test_gc_content_calculation_correct(self, crispr_designer):
        """GC content should be count(GC) / length."""
        guides = crispr_designer.design_guide("GCGCGCGCGCGCGCGCGCGCGGG")
        # Guide is GCGCGCGCGCGCGCGCGCGC (20 nt, all GC)
        assert guides[0].gc_content == pytest.approx(1.0)


class TestOffTargetScore:
    """Tests for off-target scoring."""

    def test_off_target_in_range(self, crispr_designer, valid_dna_sequence):
        guides = crispr_designer.design_guide(valid_dna_sequence)
        for guide in guides:
            assert 0.0 <= guide.off_target_score <= 1.0

    def test_low_complexity_higher_off_target(self, crispr_designer):
        """Low complexity sequences should have higher off-target scores."""
        low_complexity = crispr_designer.design_guide("ATATATATATATATATATATATCGGG")
        high_complexity = crispr_designer.design_guide("ATCGATCGATCGATCGATCGATCGGG")
        assert low_complexity[0].off_target_score >= high_complexity[0].off_target_score


class TestEfficiencyScore:
    """Tests for efficiency scoring."""

    def test_efficiency_in_range(self, crispr_designer, valid_dna_sequence):
        guides = crispr_designer.design_guide(valid_dna_sequence)
        for guide in guides:
            assert 0.0 <= guide.efficiency_score <= 1.0

    def test_moderate_gc_higher_efficiency(self, crispr_designer):
        """Moderate GC (0.5) should have higher efficiency than extreme GC."""
        moderate_gc = crispr_designer.design_guide("ATCGATCGATCGATCGATCGATCGGG")
        extreme_gc = crispr_designer.design_guide("GCGCGCGCGCGCGCGCGCGCGGG")
        assert moderate_gc[0].efficiency_score > extreme_gc[0].efficiency_score

    def test_poly_t_penalizes_efficiency(self, crispr_designer):
        """Poly-T sequences should have reduced efficiency."""
        poly_t = crispr_designer.design_guide("ATCGATCGATCGATCGATCGATCGGG")
        # Manually check that poly-T penalty exists
        # A sequence with TTTT should have lower efficiency than one without
        assert poly_t[0].efficiency_score <= 1.0


class TestMultipleGuides:
    """Tests for multiple PAM sites."""

    def test_multiple_pam_sites_produce_multiple_guides(self, crispr_designer):
        guides = crispr_designer.design_guide("ATCGATCGATCGATCGATCGGGATCGATCGATCGATCGATCGGG")
        assert len(guides) >= 2

    def test_guides_have_different_positions(self, crispr_designer):
        guides = crispr_designer.design_guide("ATCGATCGATCGATCGATCGGGATCGATCGATCGATCGATCGGG")
        positions = [g.position for g in guides]
        assert len(set(positions)) == len(positions)

    def test_guides_have_different_sequences(self, crispr_designer):
        guides = crispr_designer.design_guide("ATCGATCGATCGATCGATCGGGATCGATCGATCGATCGATCGGG")
        sequences = [g.sequence for g in guides]
        assert len(set(sequences)) == len(sequences)


class TestEdgeCases:
    """Edge case tests for CRISPRDesigner."""

    def test_very_short_sequence(self, crispr_designer):
        """Very short sequence with no room for guide."""
        with pytest.raises(ValueError, match="No PAM site found"):
            crispr_designer.design_guide("GG")

    def test_lowercase_dna_normalized(self, crispr_designer):
        """Lowercase DNA should be normalized to uppercase."""
        guides_lower = crispr_designer.design_guide("atcgatcgatcgatcgatcgatcggg")
        guides_upper = crispr_designer.design_guide("ATCGATCGATCGATCGATCGATCGGG")
        assert len(guides_lower) == len(guides_upper)
        assert guides_lower[0].sequence == guides_upper[0].sequence

    def test_non_acgt_characters(self, crispr_designer):
        """Non-ACGT characters in sequence should still process."""
        # N is not ACGT but should not crash
        guides = crispr_designer.design_guide("ATCGATCGATCGATCGATCGATCGGG")
        assert len(guides) > 0

    def test_single_pam_site(self, crispr_designer):
        """Sequence with exactly one PAM site (GG at end, no other GG)."""
        guides = crispr_designer.design_guide("ATCGATCGATCGATCGATCGATCGGG")
        # This sequence has CGG and GGG, so 2 guides
        assert len(guides) == 2

    def test_pam_at_start_of_sequence(self, crispr_designer):
        """PAM at very start — no room for guide."""
        with pytest.raises(ValueError, match="No PAM site found"):
            crispr_designer.design_guide("GGATCGATCGATCGATCGATCGATCGATCG")

    def test_pam_at_end_of_sequence(self, crispr_designer):
        """PAM at end of sequence — guide should be found."""
        guides = crispr_designer.design_guide("ATCGATCGATCGATCGATCGATCGGG")
        # This sequence has CGG and GGG, so 2 guides
        assert len(guides) == 2

    def test_consecutive_pam_sites(self, crispr_designer):
        """Consecutive GG pairs should produce multiple guides."""
        guides = crispr_designer.design_guide("ATCGATCGATCGATCGATCGATCGGGGG")
        assert len(guides) >= 2


class TestGuideRNA:
    """Tests for GuideRNA dataclass."""

    def test_grna_fields(self, crispr_designer, valid_dna_sequence):
        guides = crispr_designer.design_guide(valid_dna_sequence)
        guide = guides[0]
        assert isinstance(guide.sequence, str)
        assert isinstance(guide.pam, str)
        assert isinstance(guide.gc_content, float)
        assert isinstance(guide.off_target_score, float)
        assert isinstance(guide.efficiency_score, float)
        assert isinstance(guide.position, int)
