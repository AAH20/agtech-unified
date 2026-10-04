"""Tests for variant effect prediction."""

import pytest

from src.genomics.variant import Variant, VariantEffect, VariantEffectPredictor


class TestVariant:
    """Variant dataclass construction."""

    def test_snv_construction(self):
        v = Variant(chrom="chr1", pos=100, ref="A", alt="G")
        assert v.chrom == "chr1"
        assert v.pos == 100
        assert v.ref == "A"
        assert v.alt == "G"

    def test_indel_construction(self):
        v = Variant(chrom="chr1", pos=100, ref="AT", alt="A")
        assert len(v.ref) == 2
        assert len(v.alt) == 1

    def test_variant_is_snv(self):
        v = Variant(chrom="chr1", pos=100, ref="A", alt="G")
        assert v.is_snv() is True

    def test_variant_is_indel(self):
        v = Variant(chrom="chr1", pos=100, ref="AT", alt="A")
        assert v.is_snv() is False

    def test_variant_is_insertion(self):
        v = Variant(chrom="chr1", pos=100, ref="A", alt="AT")
        assert v.is_insertion() is True

    def test_variant_is_deletion(self):
        v = Variant(chrom="chr1", pos=100, ref="AT", alt="A")
        assert v.is_deletion() is True


class TestVariantEffect:
    """VariantEffect classification."""

    def test_synonymous(self):
        effect = VariantEffect(
            variant=Variant("chr1", 100, "A", "G"),
            effect_type="synonymous",
            impact="low",
            codon_change="AAA→AAG",
            aa_change="K→K",
        )
        assert effect.effect_type == "synonymous"
        assert effect.impact == "low"

    def test_missense(self):
        effect = VariantEffect(
            variant=Variant("chr1", 100, "A", "G"),
            effect_type="missense",
            impact="moderate",
            codon_change="AAA→GAA",
            aa_change="K→E",
        )
        assert effect.effect_type == "missense"
        assert effect.impact == "moderate"

    def test_nonsense(self):
        effect = VariantEffect(
            variant=Variant("chr1", 100, "A", "G"),
            effect_type="nonsense",
            impact="high",
            codon_change="AAA→TAA",
            aa_change="K→*",
        )
        assert effect.effect_type == "nonsense"
        assert effect.impact == "high"


class TestVariantEffectPredictor:
    """Variant effect prediction."""

    def test_predict_synonymous(self):
        predictor = VariantEffectPredictor()
        # Use reference sequence: AAA (Lys) → AAG (Lys) = synonymous
        ref_seq = "AAAGGG"
        effect = predictor.predict(
            Variant("chr1", 3, "A", "G"),
            reference_sequence=ref_seq,
        )
        assert effect.effect_type == "synonymous"
        assert effect.impact == "low"

    def test_predict_missense(self):
        predictor = VariantEffectPredictor()
        # Use reference sequence: AAA (Lys) → GAA (Glu) = missense
        ref_seq = "AAAGGG"
        effect = predictor.predict(
            Variant("chr1", 1, "A", "G"),
            reference_sequence=ref_seq,
        )
        assert effect.effect_type == "missense"
        assert effect.impact == "moderate"

    def test_predict_nonsense(self):
        predictor = VariantEffectPredictor()
        # Use reference sequence: AAA (Lys) → TAA (Stop) = nonsense
        ref_seq = "AAAGGG"
        effect = predictor.predict(
            Variant("chr1", 1, "A", "T"),
            reference_sequence=ref_seq,
        )
        assert effect.effect_type == "nonsense"
        assert effect.impact == "high"

    def test_predict_frameshift_insertion(self):
        predictor = VariantEffectPredictor()
        # Insertion of 1 base = frameshift
        effect = predictor.predict(Variant("chr1", 100, "A", "AT"))
        assert effect.effect_type == "frameshift"
        assert effect.impact == "high"

    def test_predict_frameshift_deletion(self):
        predictor = VariantEffectPredictor()
        # Deletion of 1 base = frameshift
        effect = predictor.predict(Variant("chr1", 100, "AT", "A"))
        assert effect.effect_type == "frameshift"
        assert effect.impact == "high"

    def test_predict_in_frame_indel(self):
        predictor = VariantEffectPredictor()
        # Deletion of 3 bases = in-frame deletion
        effect = predictor.predict(Variant("chr1", 100, "ATGC", "A"))
        assert effect.effect_type == "inframe_deletion"
        assert effect.impact == "moderate"

    def test_predict_start_lost(self):
        predictor = VariantEffectPredictor()
        # ATG (Met) → GTG (Val) = start lost
        ref_seq = "ATGGGG"
        effect = predictor.predict(
            Variant("chr1", 1, "A", "G"),
            reference_sequence=ref_seq,
        )
        assert effect.effect_type == "start_lost"
        assert effect.impact == "high"

    def test_predict_stop_lost(self):
        predictor = VariantEffectPredictor()
        # TAA (Stop) → CAA (Gln) = stop lost
        ref_seq = "TAAGGG"
        effect = predictor.predict(
            Variant("chr1", 1, "T", "C"),
            reference_sequence=ref_seq,
        )
        assert effect.effect_type == "stop_lost"
        assert effect.impact == "high"

    def test_predict_stop_retained(self):
        predictor = VariantEffectPredictor()
        # TAA (Stop) → TAG (Stop) = stop retained (synonymous)
        ref_seq = "TAAGGG"
        effect = predictor.predict(
            Variant("chr1", 2, "A", "G"),
            reference_sequence=ref_seq,
        )
        assert effect.effect_type == "synonymous"
        assert effect.impact == "low"

    def test_predict_with_reference_sequence(self):
        predictor = VariantEffectPredictor()
        ref_seq = "ATGCGTAAACCCGGGTTT"
        # Position 4 (1-indexed): C→A changes CGT (Arg) to AGT (Ser) = missense
        effect = predictor.predict(
            Variant("chr1", 4, "C", "A"),
            reference_sequence=ref_seq,
        )
        assert effect.effect_type == "missense"

    def test_predict_batch(self):
        predictor = VariantEffectPredictor()
        ref_seq = "AAAGGGAAAGGGAAAGGG"
        variants = [
            Variant("chr1", 3, "A", "G"),  # AAA→AAG synonymous
            Variant("chr1", 1, "A", "G"),  # AAA→GAA missense
            Variant("chr1", 1, "A", "T"),  # AAA→TAA nonsense
        ]
        effects = predictor.predict_batch(variants, reference_sequence=ref_seq)
        assert len(effects) == 3
        assert effects[0].effect_type == "synonymous"
        assert effects[1].effect_type == "missense"
        assert effects[2].effect_type == "nonsense"

    def test_predict_batch_without_reference(self):
        predictor = VariantEffectPredictor()
        # Use single-character ref/alt for SNV prediction without reference
        variants = [
            Variant("chr1", 100, "A", "G"),  # SNV
            Variant("chr1", 200, "A", "T"),  # SNV
            Variant("chr1", 300, "A", "C"),  # SNV
        ]
        effects = predictor.predict_batch(variants)
        assert len(effects) == 3
        # Without reference, all are classified as unknown
        assert all(e.effect_type == "unknown" for e in effects)

    def test_predict_empty_ref_raises(self):
        predictor = VariantEffectPredictor()
        with pytest.raises(ValueError, match="empty"):
            predictor.predict(Variant("chr1", 100, "A", "G"), reference_sequence="")

    def test_impact_score_range(self):
        predictor = VariantEffectPredictor()
        effect = predictor.predict(Variant("chr1", 100, "AAA", "GAA"))
        assert 0.0 <= effect.impact_score <= 1.0

    def test_sift_score_range(self):
        predictor = VariantEffectPredictor()
        effect = predictor.predict(Variant("chr1", 100, "AAA", "GAA"))
        assert 0.0 <= effect.sift_score <= 1.0

    def test_polyphen_score_range(self):
        predictor = VariantEffectPredictor()
        effect = predictor.predict(Variant("chr1", 100, "AAA", "GAA"))
        assert 0.0 <= effect.polyphen_score <= 1.0
