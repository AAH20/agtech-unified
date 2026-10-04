"""Tests for protein solubility and subcellular localization prediction."""

import pytest

from src.genomics.localization import SubcellularLocalizationPredictor
from src.genomics.solubility import SolubilityPredictor


class TestSolubilityPredictor:
    """Protein solubility prediction."""

    def test_soluble_protein(self):
        predictor = SolubilityPredictor()
        result = predictor.predict("MKDEKDEKDE")
        assert result.is_soluble is True
        assert result.score > 0.5

    def test_insoluble_protein(self):
        predictor = SolubilityPredictor()
        # Highly hydrophobic sequence
        result = predictor.predict("WWWWWWWWWWWW")
        assert result.is_soluble is False
        assert result.score < 0.5

    def test_score_range(self):
        predictor = SolubilityPredictor()
        result = predictor.predict("MKVLAAALLA")
        assert 0.0 <= result.score <= 1.0

    def test_empty_sequence_raises(self):
        predictor = SolubilityPredictor()
        with pytest.raises(ValueError, match="empty"):
            predictor.predict("")

    def test_hydrophobicity_factor(self):
        predictor = SolubilityPredictor()
        hydrophobic = "AAAAAAAAAAAA"
        hydrophilic = "DDDDDDDDDDDD"
        hydro_result = predictor.predict(hydrophobic)
        philic_result = predictor.predict(hydrophilic)
        assert hydro_result.score < philic_result.score

    def test_charge_factor(self):
        predictor = SolubilityPredictor()
        charged = "KKKKKKKKKKKK"
        neutral = "GGGGGGGGGGGG"
        charged_result = predictor.predict(charged)
        neutral_result = predictor.predict(neutral)
        assert charged_result.score > neutral_result.score

    def test_aromaticity_factor(self):
        predictor = SolubilityPredictor()
        aromatic = "FFFFFFFFFFFF"
        non_aromatic = "GGGGGGGGGGGG"
        aromatic_result = predictor.predict(aromatic)
        non_aromatic_result = predictor.predict(non_aromatic)
        assert aromatic_result.score < non_aromatic_result.score

    def test_length_factor(self):
        predictor = SolubilityPredictor()
        short = "MKVL"
        long_seq = "MKVLAAALLA" * 10
        short_result = predictor.predict(short)
        long_result = predictor.predict(long_seq)
        # Longer proteins tend to be less soluble
        assert long_result.score <= short_result.score + 0.1


class TestSubcellularLocalizationPredictor:
    """Subcellular localization prediction."""

    def test_cytoplasmic_default(self):
        predictor = SubcellularLocalizationPredictor()
        result = predictor.predict("MKVLDDALLA")
        assert result.localization == "cytoplasm"
        assert result.confidence > 0.0

    def test_nuclear_localization(self):
        predictor = SubcellularLocalizationPredictor()
        # NLS motif (SV40 large T antigen)
        result = predictor.predict("MKVLDDALLAKKKRKV")
        assert result.localization == "nucleus"

    def test_mitochondrial_localization(self):
        predictor = SubcellularLocalizationPredictor()
        # Mitochondrial targeting signal
        result = predictor.predict("MLSRNSIRFFKSTAAAAAK")
        assert result.localization == "mitochondria"

    def test_chloroplast_localization(self):
        predictor = SubcellularLocalizationPredictor()
        # Chloroplast transit peptide
        result = predictor.predict("MAASSMLSSAAAVSAAAAAAAAAAK")
        assert result.localization == "chloroplast"

    def test_secreted_protein(self):
        predictor = SubcellularLocalizationPredictor()
        # Signal peptide (long hydrophobic region)
        result = predictor.predict("MKKTAIAVALAAAFATVAQA")
        assert result.localization == "secreted"

    def test_empty_sequence_raises(self):
        predictor = SubcellularLocalizationPredictor()
        with pytest.raises(ValueError, match="empty"):
            predictor.predict("")

    def test_confidence_range(self):
        predictor = SubcellularLocalizationPredictor()
        result = predictor.predict("MKVLAAALLA")
        assert 0.0 <= result.confidence <= 1.0

    def test_signal_peptide_detection(self):
        predictor = SubcellularLocalizationPredictor()
        result = predictor.predict("MKKTAIAVALAAAFATVAQA")
        assert result.has_signal_peptide is True

    def test_nls_motif_detection(self):
        predictor = SubcellularLocalizationPredictor()
        result = predictor.predict("MKVLAAALLAKKKRKV")
        assert result.has_nls is True

    def test_transit_peptide_detection(self):
        predictor = SubcellularLocalizationPredictor()
        result = predictor.predict("MAASSMLSSAAAVSAAAAAAAAAAK")
        assert result.has_transit_peptide is True

    def test_multiple_localizations(self):
        predictor = SubcellularLocalizationPredictor()
        result = predictor.predict("MKVLAAALLA")
        assert isinstance(result.all_scores, dict)
        assert len(result.all_scores) > 0
