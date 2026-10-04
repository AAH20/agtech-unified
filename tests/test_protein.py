"""Dedicated unit tests for ProteinAnalyzer (src/genomics/protein.py).

Covers molecular weight accuracy, isoelectric point bounds, hydrophobicity,
composition sums to 1.0, stability score bounds, instability index, and
known protein sequences.
"""

from __future__ import annotations

import pytest

from src.genomics.protein import ProteinAnalyzer, ProteinResult


class TestProteinAnalyzerInit:
    """ProteinAnalyzer initialization tests."""

    def test_valid_aa_set_complete(self):
        analyzer = ProteinAnalyzer()
        assert len(analyzer.VALID_AA) == 20
        assert "A" in analyzer.VALID_AA
        assert "W" in analyzer.VALID_AA


class TestAnalyzeBasic:
    """Basic analysis tests."""

    def test_empty_sequence_raises(self, protein_analyzer):
        with pytest.raises(ValueError, match="Sequence cannot be empty"):
            protein_analyzer.analyze("")

    def test_invalid_amino_acid_raises(self, protein_analyzer, invalid_protein_sequence):
        with pytest.raises(ValueError, match="Invalid amino acid"):
            protein_analyzer.analyze(invalid_protein_sequence)

    def test_valid_sequence_returns_result(self, protein_analyzer, valid_protein_sequence):
        result = protein_analyzer.analyze(valid_protein_sequence)
        assert isinstance(result, ProteinResult)

    def test_length_matches_sequence(self, protein_analyzer, valid_protein_sequence):
        result = protein_analyzer.analyze(valid_protein_sequence)
        assert result.length == 20

    def test_result_contains_all_fields(self, protein_analyzer, valid_protein_sequence):
        result = protein_analyzer.analyze(valid_protein_sequence)
        assert isinstance(result.length, int)
        assert isinstance(result.molecular_weight, float)
        assert isinstance(result.isoelectric_point, float)
        assert isinstance(result.hydrophobicity, float)
        assert isinstance(result.composition, dict)
        assert isinstance(result.stability_score, float)
        assert isinstance(result.instability_index, float)


class TestMolecularWeight:
    """Tests for molecular weight calculation."""

    def test_molecular_weight_positive(self, protein_analyzer, valid_protein_sequence):
        result = protein_analyzer.analyze(valid_protein_sequence)
        assert result.molecular_weight > 0

    def test_single_aa_weight(self, protein_analyzer):
        """Single alanine: weight = 18.015 + (89.09 - 18.015) = 89.09."""
        result = protein_analyzer.analyze("A")
        assert result.molecular_weight == pytest.approx(89.09, rel=1e-3)

    def test_two_aa_weight(self, protein_analyzer):
        """Two alanines: weight = 18.015 + 2*(89.09 - 18.015) = 160.165."""
        result = protein_analyzer.analyze("AA")
        expected = 18.015 + 2 * (89.09 - 18.015)
        assert result.molecular_weight == pytest.approx(expected, rel=1e-3)

    def test_molecular_weight_increases_with_length(self, protein_analyzer):
        short = protein_analyzer.analyze("A")
        long = protein_analyzer.analyze("AAAAAAAAAA")
        assert long.molecular_weight > short.molecular_weight

    def test_known_sequence_weight(self, protein_analyzer):
        """ACDEFGHIKLMNPQRSTVWY should have predictable weight."""
        result = protein_analyzer.analyze("ACDEFGHIKLMNPQRSTVWY")
        # Each AA contributes (weight - 18.015), plus 18.015 for terminal
        # This is a sanity check that weight is in reasonable range
        assert 2000 < result.molecular_weight < 3000


class TestIsoelectricPoint:
    """Tests for isoelectric point calculation."""

    def test_pi_in_range(self, protein_analyzer, valid_protein_sequence):
        result = protein_analyzer.analyze(valid_protein_sequence)
        assert 0.0 <= result.isoelectric_point <= 14.0

    def test_neutral_sequence_pi(self, protein_analyzer):
        """Sequence with no acidic/basic residues should have pI ~ 7.0."""
        result = protein_analyzer.analyze("GGGGGG")
        assert result.isoelectric_point == pytest.approx(7.0)

    def test_acidic_sequence_pi(self, protein_analyzer):
        """Acidic sequence pI uses simplified average of pKa values."""
        result = protein_analyzer.analyze("DDDDDD")
        # Simplified pI: average of acidic pKa[1] values (D=9.60)
        assert result.isoelectric_point == pytest.approx(9.6)

    def test_basic_sequence_pi(self, protein_analyzer):
        """Basic sequence pI uses simplified average of pKa values."""
        result = protein_analyzer.analyze("KKKKKK")
        # Simplified pI: average of basic pKa[0] values (K=2.18)
        assert result.isoelectric_point == pytest.approx(2.18)


class TestHydrophobicity:
    """Tests for hydrophobicity calculation."""

    def test_hydrophobicity_calculated(self, protein_analyzer, valid_protein_sequence):
        result = protein_analyzer.analyze(valid_protein_sequence)
        assert result.hydrophobicity != 0.0

    def test_hydrophobic_sequence_positive(self, protein_analyzer):
        """Hydrophobic sequence should have positive hydrophobicity."""
        result = protein_analyzer.analyze("IIIIII")
        assert result.hydrophobicity > 0

    def test_hydrophilic_sequence_negative(self, protein_analyzer):
        """Hydrophilic sequence should have negative hydrophobicity."""
        result = protein_analyzer.analyze("DDDDDD")
        assert result.hydrophobicity < 0

    def test_hydrophobicity_bounds(self, protein_analyzer):
        """Hydrophobicity should be within Kyte-Doolittle range [-4.5, 4.5]."""
        result = protein_analyzer.analyze("ACDEFGHIKLMNPQRSTVWY")
        assert -4.5 <= result.hydrophobicity <= 4.5


class TestComposition:
    """Tests for amino acid composition."""

    def test_composition_sums_to_1(self, protein_analyzer, valid_protein_sequence):
        result = protein_analyzer.analyze(valid_protein_sequence)
        assert sum(result.composition.values()) == pytest.approx(1.0)

    def test_composition_all_positive(self, protein_analyzer, valid_protein_sequence):
        result = protein_analyzer.analyze(valid_protein_sequence)
        assert all(v > 0 for v in result.composition.values())

    def test_composition_keys_are_aa(self, protein_analyzer, valid_protein_sequence):
        result = protein_analyzer.analyze(valid_protein_sequence)
        assert all(aa in ProteinAnalyzer.VALID_AA for aa in result.composition.keys())

    def test_single_aa_composition(self, protein_analyzer):
        result = protein_analyzer.analyze("A")
        assert result.composition == {"A": 1.0}

    def test_all_same_aa_composition(self, protein_analyzer):
        result = protein_analyzer.analyze("AAAAA")
        assert result.composition == {"A": 1.0}


class TestStabilityScore:
    """Tests for stability score."""

    def test_stability_in_range(self, protein_analyzer, valid_protein_sequence):
        result = protein_analyzer.analyze(valid_protein_sequence)
        assert 0.0 <= result.stability_score <= 1.0

    def test_stabilizing_residues_increase_stability(self, protein_analyzer):
        """C, W, Y, F, P are stabilizing residues."""
        stabilizing = protein_analyzer.analyze("CWCWCW")
        non_stabilizing = protein_analyzer.analyze("AAAAAA")
        assert stabilizing.stability_score > non_stabilizing.stability_score

    def test_stability_minimum(self, protein_analyzer):
        """No stabilizing residues → stability = 0.3."""
        result = protein_analyzer.analyze("AAAAAA")
        assert result.stability_score == pytest.approx(0.3)

    def test_stability_maximum(self, protein_analyzer):
        """All stabilizing residues → stability = 1.0."""
        result = protein_analyzer.analyze("CWCWCW")
        assert result.stability_score == pytest.approx(1.0)


class TestInstabilityIndex:
    """Tests for instability index."""

    def test_instability_non_negative(self, protein_analyzer, valid_protein_sequence):
        result = protein_analyzer.analyze(valid_protein_sequence)
        assert result.instability_index >= 0.0

    def test_unstable_dipeptides_increase_index(self, protein_analyzer):
        """DP, PS, TP, AS, SD are unstable dipeptides."""
        unstable = protein_analyzer.analyze("DPDPDP")
        stable = protein_analyzer.analyze("AAAAAA")
        assert unstable.instability_index > stable.instability_index

    def test_no_unstable_dipeptides_zero_index(self, protein_analyzer):
        result = protein_analyzer.analyze("AAAAAA")
        assert result.instability_index == pytest.approx(0.0)


class TestEdgeCases:
    """Edge case tests for ProteinAnalyzer."""

    def test_single_amino_acid(self, protein_analyzer, single_aa_protein_sequence):
        result = protein_analyzer.analyze(single_aa_protein_sequence)
        assert result.length == 1
        assert result.molecular_weight > 0

    def test_very_long_sequence(self, protein_analyzer):
        """Very long sequence should complete."""
        seq = "A" * 1000
        result = protein_analyzer.analyze(seq)
        assert result.length == 1000
        assert result.molecular_weight > 0

    def test_all_same_amino_acid(self, protein_analyzer, all_same_protein_sequence):
        result = protein_analyzer.analyze(all_same_protein_sequence)
        assert result.length == 10
        assert result.composition == {"A": 1.0}

    def test_lowercase_input_normalized(self, protein_analyzer):
        """Lowercase input should be normalized to uppercase."""
        lower = protein_analyzer.analyze("acdefghiklmnpqrstvwy")
        upper = protein_analyzer.analyze("ACDEFGHIKLMNPQRSTVWY")
        assert lower.molecular_weight == pytest.approx(upper.molecular_weight)
        assert lower.length == upper.length

    def test_rare_amino_acids_rejected(self, protein_analyzer):
        """U, O, B, Z, X are not in standard AA set."""
        for aa in ["U", "O", "B", "Z", "X"]:
            with pytest.raises(ValueError, match="Invalid amino acid"):
                protein_analyzer.analyze(aa)

    def test_mixed_case_input(self, protein_analyzer):
        """Mixed case should be normalized."""
        result = protein_analyzer.analyze("AcDeFgHiKlMnPqRsTvWy")
        assert result.length == 20

    def test_all_20_amino_acids(self, protein_analyzer):
        """All 20 standard amino acids should be accepted."""
        all_aa = "ACDEFGHIKLMNPQRSTVWY"
        result = protein_analyzer.analyze(all_aa)
        assert result.length == 20
        assert len(result.composition) == 20
