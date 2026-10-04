"""Test protein secondary structure prediction."""

import pytest

from src.genomics.structure import (
    SecondaryStructurePredictor,
    StructureResult,
    predict_secondary_structure,
)


class TestSecondaryStructurePrediction:
    """Tests for Chou-Fasman style secondary structure prediction."""

    def setup_method(self):
        self.predictor = SecondaryStructurePredictor()

    def test_empty_sequence_raises(self):
        """Empty sequence raises ValueError."""
        with pytest.raises(ValueError, match="Sequence cannot be empty"):
            self.predictor.predict("")

    def test_invalid_amino_acid_raises(self):
        """Invalid amino acid raises ValueError."""
        with pytest.raises(ValueError, match="Invalid amino acid"):
            self.predictor.predict("ACDEFGHIKLMNPQRSTVWYZ")

    def test_result_type(self):
        """Prediction returns StructureResult."""
        result = self.predictor.predict("ACDEFGHIKLMNPQRSTVWY")
        assert isinstance(result, StructureResult)

    def test_per_residue_length_matches(self):
        """Per-residue prediction length equals sequence length."""
        seq = "ACDEFGHIKLMNPQRSTVWY"
        result = self.predictor.predict(seq)
        assert len(result.per_residue) == len(seq)

    def test_per_residue_valid_labels(self):
        """All per-residue labels are H, E, or C."""
        result = self.predictor.predict("ACDEFGHIKLMNPQRSTVWY")
        for label in result.per_residue:
            assert label in ("H", "E", "C")

    def test_helix_fractions_sum_to_one(self):
        """Helix, sheet, coil fractions sum to 1.0."""
        result = self.predictor.predict("ACDEFGHIKLMNPQRSTVWY")
        total = result.helix_fraction + result.sheet_fraction + result.coil_fraction
        assert total == pytest.approx(1.0)

    def test_fractions_in_range(self):
        """All structure fractions are between 0 and 1."""
        result = self.predictor.predict("ACDEFGHIKLMNPQRSTVWY")
        assert 0.0 <= result.helix_fraction <= 1.0
        assert 0.0 <= result.sheet_fraction <= 1.0
        assert 0.0 <= result.coil_fraction <= 1.0

    def test_helix_rich_sequence(self):
        """Alanine-rich sequence has significant helix content."""
        # Ala is a strong helix former
        result = self.predictor.predict("AAAAAKAAAAAKAAAAAKAAAA")
        assert result.helix_fraction > 0.3

    def test_sheet_rich_sequence(self):
        """Valine-rich sequence has significant sheet content."""
        # Val is a strong sheet former
        result = self.predictor.predict("VVVVVTVVVVVTVVVVVTVVV")
        assert result.sheet_fraction > 0.2

    def test_coil_rich_sequence(self):
        """Proline-rich sequence has significant coil content."""
        # Pro is a helix breaker, favors coil
        result = self.predictor.predict("PPPPGPPPPGPPPPGPPPPGP")
        assert result.coil_fraction > 0.3

    def test_single_amino_acid(self):
        """Single amino acid returns valid result."""
        result = self.predictor.predict("A")
        assert len(result.per_residue) == 1
        assert result.per_residue[0] in ("H", "E", "C")

    def test_lowercase_sequence(self):
        """Lowercase sequence is handled."""
        result = self.predictor.predict("acdefghiklmnpqrstvwy")
        assert len(result.per_residue) == 20

    def test_convenience_function(self):
        """predict_secondary_structure convenience function works."""
        result = predict_secondary_structure("ACDEFGHIKLMNPQRSTVWY")
        assert isinstance(result, StructureResult)
        assert len(result.per_residue) == 20

    def test_dominant_structure_label(self):
        """Dominant structure is the most frequent label."""
        result = self.predictor.predict("AAAAAKAAAAAKAAAAAKAAAA")
        assert result.dominant_structure == "H"

    def test_confidence_score_range(self):
        """Confidence score is between 0 and 1."""
        result = self.predictor.predict("ACDEFGHIKLMNPQRSTVWY")
        assert 0.0 <= result.confidence <= 1.0
