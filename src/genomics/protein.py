"""Protein sequence analysis for agricultural biotechnology."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)

# Amino acid molecular weights (Da)
AA_WEIGHTS = {
    "A": 89.09,
    "R": 174.20,
    "N": 132.12,
    "D": 133.10,
    "C": 121.15,
    "E": 147.13,
    "Q": 146.15,
    "G": 75.07,
    "H": 155.16,
    "I": 131.17,
    "L": 131.17,
    "K": 146.19,
    "M": 149.21,
    "F": 165.19,
    "P": 115.13,
    "S": 105.09,
    "T": 119.12,
    "W": 204.23,
    "Y": 181.19,
    "V": 117.15,
}

# Hydrophobicity scale (Kyte-Doolittle)
AA_HYDROPHOBICITY = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "E": -3.5,
    "Q": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

# pKa values for isoelectric point calculation
AA_PKA = {
    "A": (2.34, 9.60),
    "R": (2.17, 9.04),
    "N": (2.02, 8.80),
    "D": (1.88, 9.60),
    "C": (1.96, 10.28),
    "E": (2.19, 9.67),
    "Q": (2.17, 9.13),
    "G": (2.34, 9.60),
    "H": (1.82, 9.17),
    "I": (2.36, 9.60),
    "L": (2.36, 9.60),
    "K": (2.18, 8.95),
    "M": (2.28, 9.21),
    "F": (1.83, 9.13),
    "P": (1.99, 10.60),
    "S": (2.21, 9.15),
    "T": (2.09, 9.10),
    "W": (2.83, 9.39),
    "Y": (2.20, 9.11),
    "V": (2.32, 9.62),
}


@dataclass
class ProteinResult:
    """Protein analysis result."""

    length: int
    molecular_weight: float
    isoelectric_point: float
    hydrophobicity: float
    composition: Dict[str, float]
    stability_score: float
    instability_index: float


class ProteinAnalyzer:
    """Protein sequence analyzer for agricultural biotechnology."""

    VALID_AA = set(AA_WEIGHTS.keys())

    def analyze(self, sequence: str) -> ProteinResult:
        """Analyze a protein sequence."""
        if not sequence:
            raise ValueError("Sequence cannot be empty")

        seq = sequence.upper()
        for aa in seq:
            if aa not in self.VALID_AA:
                raise ValueError(f"Invalid amino acid: {aa}")

        length = len(seq)
        mw = self._molecular_weight(seq)
        pI = self._isoelectric_point(seq)
        hydro = self._hydrophobicity(seq)
        comp = self._composition(seq)
        stability = self._stability_score(seq)
        instability = self._instability_index(seq)

        return ProteinResult(
            length=length,
            molecular_weight=mw,
            isoelectric_point=pI,
            hydrophobicity=hydro,
            composition=comp,
            stability_score=stability,
            instability_index=instability,
        )

    def _molecular_weight(self, sequence: str) -> float:
        """Calculate molecular weight in Daltons."""
        weight = 18.015  # Water (H2O) for terminal residues
        for aa in sequence:
            weight += AA_WEIGHTS[aa] - 18.015  # Subtract water for peptide bond
        return weight

    def _isoelectric_point(self, sequence: str) -> float:
        """Estimate isoelectric point using pKa values."""
        # Simplified: use average of acidic and basic pKa values
        acidic = ["D", "E"]
        basic = ["K", "R", "H"]

        acidic_pka = [AA_PKA[aa][1] for aa in sequence if aa in acidic]
        basic_pka = [AA_PKA[aa][0] for aa in sequence if aa in basic]

        if not acidic_pka and not basic_pka:
            return 7.0

        # Weighted average
        all_pka = acidic_pka + basic_pka
        return sum(all_pka) / len(all_pka) if all_pka else 7.0

    def _hydrophobicity(self, sequence: str) -> float:
        """Calculate average hydrophobicity (Kyte-Doolittle)."""
        if not sequence:
            return 0.0
        return sum(AA_HYDROPHOBICITY[aa] for aa in sequence) / len(sequence)

    def _composition(self, sequence: str) -> Dict[str, float]:
        """Calculate amino acid composition."""
        if not sequence:
            return {}
        counts = {}
        for aa in sequence:
            counts[aa] = counts.get(aa, 0) + 1
        return {aa: count / len(sequence) for aa, count in counts.items()}

    def _stability_score(self, sequence: str) -> float:
        """Estimate protein stability (0-1, higher is more stable)."""
        if not sequence:
            return 0.0
        # Factors: moderate GC, presence of stabilizing residues
        stabilizing = {"C", "W", "Y", "F", "P"}
        stabilizing_count = sum(1 for aa in sequence if aa in stabilizing)
        return min(1.0, stabilizing_count / len(sequence) + 0.3)

    def _instability_index(self, sequence: str) -> float:
        """Calculate instability index (higher = less stable)."""
        if not sequence:
            return 0.0
        # Simplified: based on dipeptide composition
        unstable_dipeptides = {"DP", "PS", "TP", "AS", "SD"}
        count = 0
        for i in range(len(sequence) - 1):
            if sequence[i : i + 2] in unstable_dipeptides:
                count += 1
        return count * 10.0 / max(1, len(sequence) - 1)
