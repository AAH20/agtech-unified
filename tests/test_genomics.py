"""Test genomics module: CRISPR guide RNA design and protein sequence analysis."""

import pytest

from src.genomics.crispr import CRISPRDesigner
from src.genomics.protein import ProteinAnalyzer

# === CRISPR Guide RNA Tests ===


def test_grna_empty_sequence():
    """BUG-011: Empty DNA sequence raises ValueError."""
    designer = CRISPRDesigner()
    with pytest.raises(ValueError, match="Sequence cannot be empty"):
        designer.design_guide("")


def test_grna_no_pam_site():
    """Sequence without PAM site (NGG) raises ValueError."""
    designer = CRISPRDesigner()
    with pytest.raises(ValueError, match="No PAM site found"):
        designer.design_guide("ATATATATATATATATATAT")


def test_grna_valid_pam():
    """Valid sequence with PAM site returns guide RNA."""
    designer = CRISPRDesigner()
    guides = designer.design_guide("ATCGATCGATCGATCGATCGATCGGG")
    assert len(guides) > 0
    assert len(guides[0].sequence) == 20  # 20nt guide
    assert guides[0].pam.endswith("GG")  # PAM ends with GG


def test_grna_pam_extraction():
    """PAM site is correctly extracted from sequence."""
    designer = CRISPRDesigner()
    guides = designer.design_guide("ATCGATCGATCGATCGATCGATCGGG")
    assert len(guides) > 0
    assert len(guides[0].pam) == 3  # PAM is 3nt (NGG)
    assert guides[0].pam[1:] == "GG"  # Last 2 bases are GG


def test_grna_gc_content():
    """Guide RNA GC content is calculated correctly."""
    designer = CRISPRDesigner()
    guides = designer.design_guide("GCGCGCGCGCGCGCGCGCGCGGG")
    assert len(guides) > 0
    # GC content should be high for GC-rich guide
    assert guides[0].gc_content > 0.5


def test_grna_off_target_score():
    """Off-target score is between 0 and 1."""
    designer = CRISPRDesigner()
    guides = designer.design_guide("ATCGATCGATCGATCGATCGATCGGG")
    assert len(guides) > 0
    assert 0.0 <= guides[0].off_target_score <= 1.0


def test_grna_multiple_pam_sites():
    """Multiple PAM sites produce multiple guides."""
    designer = CRISPRDesigner()
    guides = designer.design_guide("ATCGATCGATCGATCGATCGGGATCGATCGATCGATCGATCGGG")
    assert len(guides) >= 2


def test_grna_efficiency_score():
    """Efficiency score is between 0 and 1."""
    designer = CRISPRDesigner()
    guides = designer.design_guide("ATCGATCGATCGATCGATCGATCGGG")
    assert len(guides) > 0
    assert 0.0 <= guides[0].efficiency_score <= 1.0


# === Protein Analysis Tests ===


def test_protein_empty_sequence():
    """BUG-012: Empty protein sequence raises ValueError."""
    analyzer = ProteinAnalyzer()
    with pytest.raises(ValueError, match="Sequence cannot be empty"):
        analyzer.analyze("")


def test_protein_molecular_weight():
    """Molecular weight is calculated for valid sequence."""
    analyzer = ProteinAnalyzer()
    result = analyzer.analyze("ACDEFGHIKLMNPQRSTVWY")
    assert result.molecular_weight > 0


def test_protein_length():
    """Protein length matches sequence length."""
    analyzer = ProteinAnalyzer()
    result = analyzer.analyze("ACDEFGHIKLMNPQRSTVWY")
    assert result.length == 20


def test_protein_isoelectric_point():
    """Isoelectric point is between 0 and 14."""
    analyzer = ProteinAnalyzer()
    result = analyzer.analyze("ACDEFGHIKLMNPQRSTVWY")
    assert 0.0 <= result.isoelectric_point <= 14.0


def test_protein_hydrophobicity():
    """Hydrophobicity score is calculated."""
    analyzer = ProteinAnalyzer()
    result = analyzer.analyze("ACDEFGHIKLMNPQRSTVWY")
    assert result.hydrophobicity != 0.0


def test_protein_amino_acid_composition():
    """Amino acid composition sums to 1.0."""
    analyzer = ProteinAnalyzer()
    result = analyzer.analyze("ACDEFGHIKLMNPQRSTVWY")
    assert sum(result.composition.values()) == pytest.approx(1.0)


def test_protein_stability_score():
    """Stability score is between 0 and 1."""
    analyzer = ProteinAnalyzer()
    result = analyzer.analyze("ACDEFGHIKLMNPQRSTVWY")
    assert 0.0 <= result.stability_score <= 1.0


def test_protein_invalid_amino_acid():
    """BUG-012: Invalid amino acid raises ValueError."""
    analyzer = ProteinAnalyzer()
    with pytest.raises(ValueError, match="Invalid amino acid"):
        analyzer.analyze("ACDEFGHIKLMNPQRSTVWYZ")


def test_protein_instability_index():
    """Instability index is calculated (non-zero for unstable sequences)."""
    analyzer = ProteinAnalyzer()
    # Sequence with unstable dipeptides (DP, PS, TP, AS, SD)
    result = analyzer.analyze("ACDEFGHIKLMNPQRSTVWYDP")
    assert result.instability_index > 0
