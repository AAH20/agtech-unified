"""Tests for allergenicity and toxicity prediction."""

import pytest

from src.genomics.allergenicity import AllergenicityPredictor, ToxicityPredictor


class TestAllergenicityPredictor:
    """Allergenicity screening using FAO/WHO decision tree."""

    def test_safe_protein(self):
        predictor = AllergenicityPredictor()
        result = predictor.predict("MKVLAAALLA")
        assert result.is_allergenic is False
        assert result.score < 0.5

    def test_known_allergen_detected(self):
        predictor = AllergenicityPredictor()
        # Use a known allergen fragment (Ara h 1 peanut allergen fragment)
        result = predictor.predict("RQQEQQFKRELRNLPQQCGLR")
        assert result.is_allergenic is True
        assert result.score > 0.5

    def test_exact_match_detection(self):
        predictor = AllergenicityPredictor()
        # Exact 8-mer match to known allergen
        result = predictor.predict("QQEQQFKR")
        assert result.is_allergenic is True

    def test_similarity_threshold(self):
        predictor = AllergenicityPredictor()
        result = predictor.predict("MKVLAAALLA")
        assert 0.0 <= result.score <= 1.0

    def test_empty_sequence_raises(self):
        predictor = AllergenicityPredictor()
        with pytest.raises(ValueError, match="empty"):
            predictor.predict("")

    def test_short_sequence_handled(self):
        predictor = AllergenicityPredictor()
        result = predictor.predict("MKVL")
        assert result.is_allergenic is False

    def test_decision_tree_steps(self):
        predictor = AllergenicityPredictor()
        result = predictor.predict("MKVLAAALLA")
        assert hasattr(result, "exact_match")
        assert hasattr(result, "similarity_score")
        assert hasattr(result, "decision")

    def test_custom_allergen_database(self):
        custom_db = ["CUSTOMALLERGENSEQ"]
        predictor = AllergenicityPredictor(allergen_db=custom_db)
        result = predictor.predict("CUSTOMALLERGENSEQ")
        assert result.is_allergenic is True

    def test_sliding_window_size(self):
        predictor = AllergenicityPredictor()
        assert predictor.window_size == 8

    def test_similarity_threshold_configurable(self):
        predictor = AllergenicityPredictor(similarity_threshold=0.9)
        assert predictor.similarity_threshold == 0.9


class TestToxicityPredictor:
    """Toxicity prediction for hemolytic/cytotoxic peptides."""

    def test_non_toxic_protein(self):
        predictor = ToxicityPredictor()
        result = predictor.predict("MKVLAAALLA")
        assert result.is_toxic is False
        assert result.score < 0.5

    def test_hemolytic_peptide_detected(self):
        predictor = ToxicityPredictor()
        # Melittin-like hemolytic peptide fragment
        result = predictor.predict("GIGAVLKVLTTGLPALISWIKRKRQQ")
        assert result.is_toxic is True
        assert result.score > 0.5

    def test_score_range(self):
        predictor = ToxicityPredictor()
        result = predictor.predict("MKVLAAALLA")
        assert 0.0 <= result.score <= 1.0

    def test_empty_sequence_raises(self):
        predictor = ToxicityPredictor()
        with pytest.raises(ValueError, match="empty"):
            predictor.predict("")

    def test_hydrophobicity_factor(self):
        predictor = ToxicityPredictor()
        # Highly hydrophobic sequence should score higher
        hydrophobic = "AAAAAAAAAAAA"
        hydrophilic = "DDDDDDDDDDDD"
        hydro_result = predictor.predict(hydrophobic)
        philic_result = predictor.predict(hydrophilic)
        assert hydro_result.score >= philic_result.score

    def test_charge_factor(self):
        predictor = ToxicityPredictor()
        # Highly charged sequence should score higher than neutral
        charged = "KKKKKKKKKKKK"
        neutral = "GGGGGGGGGGGG"
        charged_result = predictor.predict(charged)
        neutral_result = predictor.predict(neutral)
        assert charged_result.score >= neutral_result.score

    def test_length_factor(self):
        predictor = ToxicityPredictor()
        # Longer peptides have more potential toxic sites
        short = "MKVL"
        long_seq = "MKVLAAALLA" * 5
        short_result = predictor.predict(short)
        long_result = predictor.predict(long_seq)
        assert long_result.score >= short_result.score
