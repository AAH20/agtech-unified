"""Variant effect prediction for agricultural biotechnology.

Classifies genetic variants as synonymous, missense, nonsense, frameshift,
or in-frame indels. Provides SIFT-style conservation scoring and
PolyPhen-style structural impact prediction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from src.genomics.codon import CODON_TABLE

logger = logging.getLogger(__name__)


@dataclass
class Variant:
    """A genetic variant (SNV or indel)."""

    chrom: str
    pos: int
    ref: str
    alt: str

    def is_snv(self) -> bool:
        """Return True if this is a single nucleotide variant."""
        return len(self.ref) == 1 and len(self.alt) == 1

    def is_insertion(self) -> bool:
        """Return True if this is an insertion."""
        return len(self.alt) > len(self.ref)

    def is_deletion(self) -> bool:
        """Return True if this is a deletion."""
        return len(self.ref) > len(self.alt)

    def length_change(self) -> int:
        """Return the net length change of the variant."""
        return len(self.alt) - len(self.ref)


@dataclass
class VariantEffect:
    """Predicted effect of a genetic variant."""

    variant: Variant
    effect_type: str
    impact: str
    codon_change: str
    aa_change: str
    impact_score: float = 0.0
    sift_score: float = 0.0
    polyphen_score: float = 0.0


class VariantEffectPredictor:
    """Predicts the functional effect of genetic variants.

    Uses codon table lookup to classify variants and provides
    conservation and structural impact scores.
    """

    # Amino acid properties for impact prediction
    AA_PROPERTIES = {
        "A": {"hydrophobic": True, "charge": 0, "size": "small"},
        "R": {"hydrophobic": False, "charge": 1, "size": "large"},
        "N": {"hydrophobic": False, "charge": 0, "size": "small"},
        "D": {"hydrophobic": False, "charge": -1, "size": "small"},
        "C": {"hydrophobic": True, "charge": 0, "size": "small"},
        "E": {"hydrophobic": False, "charge": -1, "size": "medium"},
        "Q": {"hydrophobic": False, "charge": 0, "size": "medium"},
        "G": {"hydrophobic": False, "charge": 0, "size": "tiny"},
        "H": {"hydrophobic": False, "charge": 0.5, "size": "medium"},
        "I": {"hydrophobic": True, "charge": 0, "size": "medium"},
        "L": {"hydrophobic": True, "charge": 0, "size": "medium"},
        "K": {"hydrophobic": False, "charge": 1, "size": "medium"},
        "M": {"hydrophobic": True, "charge": 0, "size": "medium"},
        "F": {"hydrophobic": True, "charge": 0, "size": "large"},
        "P": {"hydrophobic": False, "charge": 0, "size": "small"},
        "S": {"hydrophobic": False, "charge": 0, "size": "tiny"},
        "T": {"hydrophobic": False, "charge": 0, "size": "small"},
        "W": {"hydrophobic": True, "charge": 0, "size": "large"},
        "Y": {"hydrophobic": True, "charge": 0, "size": "large"},
        "V": {"hydrophobic": True, "charge": 0, "size": "small"},
        "*": {"hydrophobic": False, "charge": 0, "size": "none"},
    }

    def predict(
        self,
        variant: Variant,
        reference_sequence: Optional[str] = None,
    ) -> VariantEffect:
        """Predict the effect of a variant.

        Args:
            variant: The variant to analyze.
            reference_sequence: Optional reference DNA sequence for context.

        Returns:
            VariantEffect with classification and scores.
        """
        if reference_sequence is not None and len(reference_sequence) == 0:
            raise ValueError("Reference sequence cannot be empty")

        if variant.is_snv():
            return self._predict_snv(variant, reference_sequence)
        else:
            return self._predict_indel(variant)

    def predict_batch(
        self,
        variants: List[Variant],
        reference_sequence: Optional[str] = None,
    ) -> List[VariantEffect]:
        """Predict effects for a batch of variants.

        Args:
            variants: List of variants to analyze.
            reference_sequence: Optional reference DNA sequence for context.

        Returns:
            List of VariantEffect objects.
        """
        return [self.predict(v, reference_sequence=reference_sequence) for v in variants]

    def _predict_snv(
        self,
        variant: Variant,
        reference_sequence: Optional[str] = None,
    ) -> VariantEffect:
        """Predict effect of a single nucleotide variant."""
        ref = variant.ref.upper()
        alt = variant.alt.upper()

        # Determine codon position
        if reference_sequence:
            ref_seq = reference_sequence.upper()
            codon_pos = (variant.pos - 1) % 3
            codon_start = variant.pos - 1 - codon_pos
            ref_codon = ref_seq[codon_start : codon_start + 3]
            if len(ref_codon) < 3:
                return self._make_effect(variant, "unknown", "unknown", f"{ref}>{alt}", "?")
            alt_codon = list(ref_codon)
            alt_codon[codon_pos] = alt
            alt_codon = "".join(alt_codon)
        else:
            # Without reference, assume ref/alt are codons
            ref_codon = ref
            alt_codon = alt

        ref_aa = CODON_TABLE.get(ref_codon, "?")
        alt_aa = CODON_TABLE.get(alt_codon, "?")

        if ref_aa == "?" or alt_aa == "?":
            return self._make_effect(variant, "unknown", "unknown", f"{ref}>{alt}", "?")

        if ref_aa == alt_aa:
            effect_type = "synonymous"
            impact = "low"
        elif alt_aa == "*":
            effect_type = "nonsense"
            impact = "high"
        elif ref_aa == "M" and variant.pos <= 3:
            effect_type = "start_lost"
            impact = "high"
        elif ref_aa == "*" and alt_aa != "*":
            effect_type = "stop_lost"
            impact = "high"
        else:
            effect_type = "missense"
            impact = "moderate"

        impact_score = self._calculate_impact_score(ref_aa, alt_aa, effect_type)
        sift_score = self._calculate_sift_score(ref_aa, alt_aa)
        polyphen_score = self._calculate_polyphen_score(ref_aa, alt_aa)

        return VariantEffect(
            variant=variant,
            effect_type=effect_type,
            impact=impact,
            codon_change=f"{ref_codon}→{alt_codon}",
            aa_change=f"{ref_aa}→{alt_aa}",
            impact_score=impact_score,
            sift_score=sift_score,
            polyphen_score=polyphen_score,
        )

    def _predict_indel(self, variant: Variant) -> VariantEffect:
        """Predict effect of an insertion or deletion."""
        length_change = variant.length_change()

        if length_change % 3 == 0:
            # In-frame indel
            if variant.is_insertion():
                effect_type = "inframe_insertion"
            else:
                effect_type = "inframe_deletion"
            impact = "moderate"
        else:
            # Frameshift
            effect_type = "frameshift"
            impact = "high"

        impact_score = 0.8 if effect_type == "frameshift" else 0.5
        sift_score = 0.1 if effect_type == "frameshift" else 0.3
        polyphen_score = 0.9 if effect_type == "frameshift" else 0.6

        return VariantEffect(
            variant=variant,
            effect_type=effect_type,
            impact=impact,
            codon_change=f"{variant.ref}>{variant.alt}",
            aa_change="FS" if effect_type == "frameshift" else "IF",
            impact_score=impact_score,
            sift_score=sift_score,
            polyphen_score=polyphen_score,
        )

    def _make_effect(
        self,
        variant: Variant,
        effect_type: str,
        impact: str,
        codon_change: str,
        aa_change: str,
    ) -> VariantEffect:
        """Create a VariantEffect with default scores."""
        return VariantEffect(
            variant=variant,
            effect_type=effect_type,
            impact=impact,
            codon_change=codon_change,
            aa_change=aa_change,
        )

    def _calculate_impact_score(self, ref_aa: str, alt_aa: str, effect_type: str) -> float:
        """Calculate impact score (0-1, higher = more severe)."""
        if effect_type == "synonymous":
            return 0.0
        if effect_type == "nonsense":
            return 1.0
        if effect_type == "start_lost":
            return 0.95
        if effect_type == "stop_lost":
            return 0.9

        # Missense: based on amino acid property changes
        ref_props = self.AA_PROPERTIES.get(ref_aa, {})
        alt_props = self.AA_PROPERTIES.get(alt_aa, {})

        score = 0.0
        if ref_props.get("charge") != alt_props.get("charge"):
            score += 0.4
        if ref_props.get("hydrophobic") != alt_props.get("hydrophobic"):
            score += 0.3
        if ref_props.get("size") != alt_props.get("size"):
            score += 0.2

        return min(1.0, score)

    def _calculate_sift_score(self, ref_aa: str, alt_aa: str) -> float:
        """Calculate SIFT-style conservation score (0-1, lower = more deleterious)."""
        if ref_aa == alt_aa:
            return 1.0

        # Conservative substitutions score higher
        similar = {
            ("A", "G"),
            ("S", "T"),
            ("D", "E"),
            ("N", "Q"),
            ("K", "R"),
            ("I", "L"),
            ("I", "V"),
            ("L", "V"),
            ("F", "Y"),
            ("M", "I"),
            ("M", "L"),
        }
        if (ref_aa, alt_aa) in similar or (alt_aa, ref_aa) in similar:
            return 0.7

        # Radical substitutions score lower
        return 0.1

    def _calculate_polyphen_score(self, ref_aa: str, alt_aa: str) -> float:
        """Calculate PolyPhen-style structural impact score (0-1, higher = more damaging)."""
        if ref_aa == alt_aa:
            return 0.0

        # Charge changes are most damaging
        ref_charge = self.AA_PROPERTIES.get(ref_aa, {}).get("charge", 0)
        alt_charge = self.AA_PROPERTIES.get(alt_aa, {}).get("charge", 0)
        if ref_charge != alt_charge:
            return 0.9

        # Hydrophobicity changes
        ref_hydro = self.AA_PROPERTIES.get(ref_aa, {}).get("hydrophobic", False)
        alt_hydro = self.AA_PROPERTIES.get(alt_aa, {}).get("hydrophobic", False)
        if ref_hydro != alt_hydro:
            return 0.7

        return 0.4
