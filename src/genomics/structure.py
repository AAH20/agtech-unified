"""Protein secondary structure prediction for agricultural biotechnology.

Implements a Chou-Fasman style propensity-based method that predicts
alpha-helix (H), beta-sheet (E), and coil (C) at each residue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

# Chou-Fasman conformation parameters (propensity for helix/sheet/coil)
# Values derived from statistical frequencies in known structures.
_HELIX_PROPENSITY = {
    "A": 1.45,
    "R": 0.79,
    "N": 0.73,
    "D": 0.98,
    "C": 0.77,
    "E": 1.53,
    "Q": 1.17,
    "G": 0.53,
    "H": 1.24,
    "I": 1.00,
    "L": 1.34,
    "K": 1.07,
    "M": 1.20,
    "F": 1.12,
    "P": 0.59,
    "S": 0.79,
    "T": 0.82,
    "W": 1.14,
    "Y": 0.61,
    "V": 0.91,
}

_SHEET_PROPENSITY = {
    "A": 0.97,
    "R": 0.90,
    "N": 0.65,
    "D": 0.80,
    "C": 1.19,
    "E": 0.26,
    "Q": 1.10,
    "G": 0.81,
    "H": 0.71,
    "I": 1.60,
    "L": 1.22,
    "K": 0.74,
    "M": 1.67,
    "F": 1.28,
    "P": 0.62,
    "S": 0.72,
    "T": 1.20,
    "W": 1.19,
    "Y": 1.29,
    "V": 1.87,
}

# Amino acids that strongly favor coil/turn
_COIL_BREAKERS = {"P", "G"}


@dataclass
class StructureResult:
    """Result of secondary structure prediction."""

    per_residue: List[str]
    helix_fraction: float
    sheet_fraction: float
    coil_fraction: float
    dominant_structure: str
    confidence: float


class SecondaryStructurePredictor:
    """Predicts protein secondary structure using Chou-Fasman propensities."""

    VALID_AA = set(_HELIX_PROPENSITY.keys())

    def predict(self, protein_sequence: str) -> StructureResult:
        """Predict secondary structure for a protein sequence.

        Args:
            protein_sequence: Amino acid sequence.

        Returns:
            StructureResult with per-residue predictions and aggregate statistics.
        """
        if not protein_sequence:
            raise ValueError("Sequence cannot be empty")

        seq = protein_sequence.upper()
        for aa in seq:
            if aa not in self.VALID_AA:
                raise ValueError(f"Invalid amino acid: {aa}")

        per_residue = self._predict_per_residue(seq)
        helix_frac = per_residue.count("H") / len(seq)
        sheet_frac = per_residue.count("E") / len(seq)
        coil_frac = per_residue.count("C") / len(seq)

        dominant = max(
            {"H": helix_frac, "E": sheet_frac, "C": coil_frac},
            key=lambda k: {"H": helix_frac, "E": sheet_frac, "C": coil_frac}[k],
        )

        # Confidence based on how dominant the top structure is
        max_frac = max(helix_frac, sheet_frac, coil_frac)
        confidence = min(1.0, max_frac * 1.5)  # Scale so 67% -> ~1.0

        return StructureResult(
            per_residue=per_residue,
            helix_fraction=helix_frac,
            sheet_fraction=sheet_frac,
            coil_fraction=coil_frac,
            dominant_structure=dominant,
            confidence=confidence,
        )

    def _predict_per_residue(self, seq: str) -> List[str]:
        """Predict structure label for each residue."""
        labels = []
        for aa in seq:
            h = _HELIX_PROPENSITY.get(aa, 0.0)
            e = _SHEET_PROPENSITY.get(aa, 0.0)

            if aa in _COIL_BREAKERS:
                labels.append("C")
            elif h > e and h > 1.0:
                labels.append("H")
            elif e > h and e > 1.0:
                labels.append("E")
            elif h > e:
                labels.append("H")
            elif e > h:
                labels.append("E")
            else:
                labels.append("C")
        return labels


def predict_secondary_structure(protein_sequence: str) -> StructureResult:
    """Convenience function for secondary structure prediction.

    Args:
        protein_sequence: Amino acid sequence.

    Returns:
        StructureResult with per-residue predictions.
    """
    predictor = SecondaryStructurePredictor()
    return predictor.predict(protein_sequence)
