"""Test enhanced genomics features: codon optimization, variant effect, allergenicity."""

import pytest

from src.genomics.crispr import codon_optimize
from src.genomics.protein import predict_variant_effect, screen_allergenicity

# === Codon Optimization Tests ===


def test_codon_optimize_human():
    """Codon optimization replaces rare codons with human-preferred ones."""
    # TTA (Leu) -> CTG (Leu) for human
    result = codon_optimize("TTATTA", "human")
    assert result == "CTGCTG"


def test_codon_optimize_ecoli():
    """Codon optimization uses E. coli codon preferences."""
    # AGA (Arg) -> CGT (Arg) for E. coli
    result = codon_optimize("AGA", "e_coli")
    assert result == "CGT"


def test_codon_optimize_invalid_length():
    """Sequence length not a multiple of 3 raises ValueError."""
    with pytest.raises(ValueError, match="multiple of 3"):
        codon_optimize("AT")


def test_codon_optimize_invalid_organism():
    """Unsupported organism raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported organism"):
        codon_optimize("ATG", "invalid_species")


# === Variant Effect Prediction Tests ===


def test_variant_effect_benign():
    """Conservative substitution (similar properties) is benign."""
    # Serine (hydro=-0.8) -> Threonine (hydro=-0.7), diff=0.1, no charge change
    result = predict_variant_effect("ACDEFGHIKLMNPQRSTVWY", 15, "T")
    assert result == "benign"


def test_variant_effect_probably_damaging():
    """Charge reversal is probably damaging."""
    # Lysine (K, +1) -> Glutamic acid (E, -1)
    result = predict_variant_effect("ACDEFGHIKLMNPQRSTVWY", 10, "E")
    assert result == "probably_damaging"


def test_variant_effect_possibly_damaging():
    """Moderate hydrophobicity change is possibly damaging."""
    # Alanine (hydro=1.8) -> Isoleucine (hydro=4.5), diff=2.7
    result = predict_variant_effect("ACDEFGHIKLMNPQRSTVWY", 0, "I")
    assert result == "possibly_damaging"


# === Allergenicity Screening Tests ===


def test_allergenicity_with_match():
    """Sequence with allergen motifs returns positive score."""
    # Contains "CGLR" motif
    result = screen_allergenicity("ACDEFGHIKLMNPQRSTVWYCGLR")
    assert result > 0.0
    assert result <= 1.0
