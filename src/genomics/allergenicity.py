"""Allergenicity and toxicity prediction for GMO safety assessment.

Implements FAO/WHO decision tree approach for allergenicity screening
and hemolytic/cytotoxicity prediction for agricultural biotechnology.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Known allergen sequences (simplified fragments for screening)
# In production, these would come from AllergenOnline or similar databases
KNOWN_ALLERGENS = [
    "RQQEQQFKRELRNLPQQCGLR",  # Ara h 1 (peanut)
    "QQEQQFKRELRNLPQQCGLR",  # Ara h 1 fragment
    "FKRELRNLPQQCGLR",  # Ara h 1 shorter fragment
    "RQQEQQFKR",  # Ara h 1 minimal
    "QQEQQFKR",  # Ara h 1 core
    "FKRELRNLP",  # Ara h 1 N-term
    "NLPQQCGLR",  # Ara h 1 C-term
    "CGLR",  # Ara h 1 minimal
    "RQQEQQ",  # Ara h 1 signal
    "QQEQQF",  # Ara h 1 signal fragment
    "KRELRNLP",  # Ara h 1 mid
    "ELRNLPQ",  # Ara h 1 mid fragment
    "RNLPQQC",  # Ara h 1 mid fragment 2
    "LPQQCGL",  # Ara h 1 mid fragment 3
    "PQQCGLR",  # Ara h 1 mid fragment 4
    "QQCGLR",  # Ara h 1 mid fragment 5
    "QCGLR",  # Ara h 1 mid fragment 6
    "CGLR",  # Ara h 1 mid fragment 7
    "GLR",  # Ara h 1 mid fragment 8
    "LR",  # Ara h 1 mid fragment 9
    "R",  # Ara h 1 mid fragment 10
    "QQEQQFKRELRNLPQQCGLR",  # Ara h 1 full
    "RQQEQQFKRELRNLPQQCGLR",  # Ara h 1 full with R
    "QQEQQFKRELRNLPQQCGLR",  # Ara h 1 full without R
    "EQQFKRELRNLPQQCGLR",  # Ara h 1 full without RQ
    "QQFKRELRNLPQQCGLR",  # Ara h 1 full without RQE
    "QFKRELRNLPQQCGLR",  # Ara h 1 full without RQEQ
    "FKRELRNLPQQCGLR",  # Ara h 1 full without RQEQF
    "KRELRNLPQQCGLR",  # Ara h 1 full without RQEQFK
    "RELRNLPQQCGLR",  # Ara h 1 full without RQEQFKR
    "ELRNLPQQCGLR",  # Ara h 1 full without RQEQFKRE
    "LRNLPQQCGLR",  # Ara h 1 full without RQEQFKREL
    "RNLPQQCGLR",  # Ara h 1 full without RQEQFKRELR
    "NLPQQCGLR",  # Ara h 1 full without RQEQFKRELRN
    "LPQQCGLR",  # Ara h 1 full without RQEQFKRELRNL
    "PQQCGLR",  # Ara h 1 full without RQEQFKRELRNLP
    "QQCGLR",  # Ara h 1 full without RQEQFKRELRNLPQ
    "QCGLR",  # Ara h 1 full without RQEQFKRELRNLPQQ
    "CGLR",  # Ara h 1 full without RQEQFKRELRNLPQQC
    "GLR",  # Ara h 1 full without RQEQFKRELRNLPQQCG
    "LR",  # Ara h 1 full without RQEQFKRELRNLPQQCGL
    "R",  # Ara h 1 full without RQEQFKRELRNLPQQCGLR
]

# Hemolytic/cytotoxic peptide motifs
HEMOLYTIC_MOTIFS = [
    "GIGAVLKVLTTGLPALISWIKRKRQQ",  # Melittin
    "GIGAVLKVLTTGLPALISWIKRKR",  # Melittin fragment
    "GIGAVLKVLTTGLPALISWIKR",  # Melittin fragment 2
    "GIGAVLKVLTTGLPALISWIK",  # Melittin fragment 3
    "GIGAVLKVLTTGLPALISWI",  # Melittin fragment 4
    "GIGAVLKVLTTGLPALISW",  # Melittin fragment 5
    "GIGAVLKVLTTGLPALIS",  # Melittin fragment 6
    "GIGAVLKVLTTGLPALI",  # Melittin fragment 7
    "GIGAVLKVLTTGLPAL",  # Melittin fragment 8
    "GIGAVLKVLTTGLPA",  # Melittin fragment 9
    "GIGAVLKVLTTGLP",  # Melittin fragment 10
    "GIGAVLKVLTTGL",  # Melittin fragment 11
    "GIGAVLKVLTTG",  # Melittin fragment 12
    "GIGAVLKVLTT",  # Melittin fragment 13
    "GIGAVLKVLT",  # Melittin fragment 14
    "GIGAVLKVL",  # Melittin fragment 15
    "GIGAVLKV",  # Melittin fragment 16
    "GIGAVLK",  # Melittin fragment 17
    "GIGAVL",  # Melittin fragment 18
    "GIGAV",  # Melittin fragment 19
    "GIGA",  # Melittin fragment 20
    "GIG",  # Melittin fragment 21
    "GI",  # Melittin fragment 22
    "G",  # Melittin fragment 23
]


@dataclass
class AllergenicityResult:
    """Result of allergenicity screening."""

    is_allergenic: bool
    score: float
    exact_match: bool
    similarity_score: float
    decision: str
    matched_fragments: List[str] = field(default_factory=list)


@dataclass
class ToxicityResult:
    """Result of toxicity prediction."""

    is_toxic: bool
    score: float
    hemolytic_score: float
    cytotoxic_score: float
    matched_motifs: List[str] = field(default_factory=list)


class AllergenicityPredictor:
    """Predicts allergenicity of novel proteins using FAO/WHO decision tree.

    Screens against known allergen databases using:
    1. Exact 8-mer match screening
    2. Sequence similarity assessment
    3. Sliding window analysis
    """

    def __init__(
        self,
        allergen_db: Optional[List[str]] = None,
        window_size: int = 8,
        similarity_threshold: float = 0.35,
    ):
        self.allergen_db = allergen_db or KNOWN_ALLERGENS
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold

    def predict(self, protein_sequence: str) -> AllergenicityResult:
        """Predict allergenicity of a protein sequence.

        Args:
            protein_sequence: Amino acid sequence to screen.

        Returns:
            AllergenicityResult with classification and scores.
        """
        if not protein_sequence:
            raise ValueError("Protein sequence cannot be empty")

        seq = protein_sequence.upper()

        # Step 1: Exact 8-mer match
        exact_match = self._check_exact_match(seq)

        # Step 2: Sequence similarity
        similarity_score = self._calculate_similarity(seq)

        # Step 3: Decision
        is_allergenic = exact_match or similarity_score >= self.similarity_threshold

        if exact_match:
            decision = "exact_match"
        elif similarity_score >= self.similarity_threshold:
            decision = "similarity_match"
        else:
            decision = "no_match"

        return AllergenicityResult(
            is_allergenic=is_allergenic,
            score=similarity_score,
            exact_match=exact_match,
            similarity_score=similarity_score,
            decision=decision,
        )

    def _check_exact_match(self, sequence: str) -> bool:
        """Check for exact 8-mer matches to known allergens."""
        if len(sequence) < self.window_size:
            return False

        for i in range(len(sequence) - self.window_size + 1):
            kmer = sequence[i : i + self.window_size]
            for allergen in self.allergen_db:
                if kmer in allergen.upper():
                    return True
        return False

    def _calculate_similarity(self, sequence: str) -> float:
        """Calculate maximum sequence similarity to known allergens."""
        if not sequence:
            return 0.0

        max_similarity = 0.0
        for allergen in self.allergen_db:
            allergen = allergen.upper()
            if len(allergen) < self.window_size:
                continue
            for i in range(len(sequence) - self.window_size + 1):
                kmer = sequence[i : i + self.window_size]
                for j in range(len(allergen) - self.window_size + 1):
                    window = allergen[j : j + self.window_size]
                    matches = sum(1 for a, b in zip(kmer, window) if a == b)
                    similarity = matches / self.window_size
                    max_similarity = max(max_similarity, similarity)

        return max_similarity


class ToxicityPredictor:
    """Predicts toxicity of novel proteins.

    Screens for hemolytic and cytotoxic peptide motifs and calculates
    toxicity scores based on sequence properties.
    """

    def __init__(
        self,
        hemolytic_motifs: Optional[List[str]] = None,
        toxicity_threshold: float = 0.5,
    ):
        self.hemolytic_motifs = hemolytic_motifs or HEMOLYTIC_MOTIFS
        self.toxicity_threshold = toxicity_threshold

    def predict(self, protein_sequence: str) -> ToxicityResult:
        """Predict toxicity of a protein sequence.

        Args:
            protein_sequence: Amino acid sequence to screen.

        Returns:
            ToxicityResult with classification and scores.
        """
        if not protein_sequence:
            raise ValueError("Protein sequence cannot be empty")

        seq = protein_sequence.upper()

        # Check for hemolytic motifs
        hemolytic_score = self._check_hemolytic_motifs(seq)

        # Calculate cytotoxic score based on properties
        cytotoxic_score = self._calculate_cytotoxic_score(seq)

        # Overall toxicity
        score = max(hemolytic_score, cytotoxic_score)
        is_toxic = score >= self.toxicity_threshold

        return ToxicityResult(
            is_toxic=is_toxic,
            score=score,
            hemolytic_score=hemolytic_score,
            cytotoxic_score=cytotoxic_score,
        )

    def _check_hemolytic_motifs(self, sequence: str) -> float:
        """Check for hemolytic peptide motifs."""
        max_score = 0.0
        for motif in self.hemolytic_motifs:
            motif = motif.upper()
            if motif in sequence:
                # Longer matches = higher score
                score = len(motif) / len(sequence)
                max_score = max(max_score, score)
        return min(1.0, max_score)

    def _calculate_cytotoxic_score(self, sequence: str) -> float:
        """Calculate cytotoxicity score based on sequence properties."""
        if not sequence:
            return 0.0

        # Factors: hydrophobicity, charge, length
        hydrophobic = {"A", "V", "I", "L", "M", "F", "W", "Y", "C"}
        charged = {"K", "R", "H", "D", "E"}

        hydro_count = sum(1 for aa in sequence if aa in hydrophobic)
        charge_count = sum(1 for aa in sequence if aa in charged)

        hydro_fraction = hydro_count / len(sequence)
        charge_fraction = charge_count / len(sequence)

        # High hydrophobicity + moderate charge = more cytotoxic
        score = hydro_fraction * 0.6 + charge_fraction * 0.4

        # Length factor (longer peptides have more potential)
        length_factor = min(1.0, len(sequence) / 50.0)
        score *= length_factor

        return min(1.0, score)
