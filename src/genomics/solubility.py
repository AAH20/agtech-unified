"""Protein solubility prediction for agricultural biotechnology.

Predicts whether a protein will be soluble when expressed in a
heterologous system using sequence features (charge, hydrophobicity,
aromaticity).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Amino acid properties
HYDROPHOBIC_AA = {"A", "V", "I", "L", "M", "F", "W", "Y", "C"}
CHARGED_AA = {"K", "R", "H", "D", "E"}
AROMATIC_AA = {"F", "W", "Y"}


@dataclass
class SolubilityResult:
    """Result of solubility prediction."""

    is_soluble: bool
    score: float
    hydrophobicity_score: float
    charge_score: float
    aromaticity_score: float


class SolubilityPredictor:
    """Predicts protein solubility from sequence features.

    Uses a weighted combination of:
    - Hydrophobicity (lower = more soluble)
    - Charge (higher = more soluble)
    - Aromaticity (lower = more soluble)
    - Length (shorter = more soluble)
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def predict(self, protein_sequence: str) -> SolubilityResult:
        """Predict solubility of a protein sequence.

        Args:
            protein_sequence: Amino acid sequence.

        Returns:
            SolubilityResult with classification and scores.
        """
        if not protein_sequence:
            raise ValueError("Protein sequence cannot be empty")

        seq = protein_sequence.upper()

        # Calculate feature scores
        hydro_score = self._hydrophobicity_score(seq)
        charge_score = self._charge_score(seq)
        aromatic_score = self._aromaticity_score(seq)
        length_score = self._length_score(seq)

        # Weighted combination (higher = more soluble)
        # Normalize hydrophobicity: < 50% hydrophobic is good
        hydro_normalized = max(0.0, 1.0 - hydro_score * 1.5)
        score = (
            hydro_normalized * 0.35
            + charge_score * 0.30
            + (1.0 - aromatic_score) * 0.20
            + length_score * 0.15
        )

        return SolubilityResult(
            is_soluble=score >= self.threshold,
            score=score,
            hydrophobicity_score=hydro_score,
            charge_score=charge_score,
            aromaticity_score=aromatic_score,
        )

    def _hydrophobicity_score(self, sequence: str) -> float:
        """Calculate hydrophobicity score (0-1, higher = more hydrophobic)."""
        if not sequence:
            return 0.0
        hydro_count = sum(1 for aa in sequence if aa in HYDROPHOBIC_AA)
        return hydro_count / len(sequence)

    def _charge_score(self, sequence: str) -> float:
        """Calculate charge score (0-1, higher = more charged)."""
        if not sequence:
            return 0.0
        charge_count = sum(1 for aa in sequence if aa in CHARGED_AA)
        return charge_count / len(sequence)

    def _aromaticity_score(self, sequence: str) -> float:
        """Calculate aromaticity score (0-1, higher = more aromatic)."""
        if not sequence:
            return 0.0
        aromatic_count = sum(1 for aa in sequence if aa in AROMATIC_AA)
        return aromatic_count / len(sequence)

    def _length_score(self, sequence: str) -> float:
        """Calculate length score (0-1, higher = shorter = more soluble)."""
        # Proteins < 200 aa are generally more soluble
        if len(sequence) <= 100:
            return 1.0
        elif len(sequence) <= 200:
            return 0.8
        elif len(sequence) <= 400:
            return 0.5
        else:
            return 0.3
