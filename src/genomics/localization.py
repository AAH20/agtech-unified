"""Subcellular localization prediction for agricultural protein engineering.

Predicts where a protein will localize within a plant cell using
signal peptide detection, transit peptide recognition, and NLS/NES
motif scanning.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger(__name__)

# Signal peptide pattern (N-terminal hydrophobic region)
SIGNAL_PEPTIDE_PATTERN = re.compile(r"^M[A-Z]{5,20}[AILVMFW]")

# Chloroplast transit peptide pattern
CHLOROPLAST_PATTERN = re.compile(r"^M[A-Z]{0,15}[STA][A-Z]{0,25}[AILV]")

# Mitochondrial targeting signal pattern
MITOCHONDRIA_PATTERN = re.compile(r"^M[RL][A-Z]{3,20}[ST]")

# Nuclear localization signal patterns
NLS_PATTERNS = [
    re.compile(r"KKKRKV"),  # SV40 large T antigen
    re.compile(r"PKKKRKV"),  # Nucleoplasmin
    re.compile(r"KRPAATKKAGQAKKKK"),  # c-Myc
    re.compile(r"RKKR"),  # Minimal NLS
]

# Nuclear export signal pattern
NES_PATTERN = re.compile(r"[LIVFM][A-Z]{2,3}[LIVFM][A-Z]{2,3}[LIVFM][A-Z][LIVFM]")


@dataclass
class LocalizationResult:
    """Result of subcellular localization prediction."""

    localization: str
    confidence: float
    has_signal_peptide: bool = False
    has_nls: bool = False
    has_nes: bool = False
    has_transit_peptide: bool = False
    all_scores: Dict[str, float] = field(default_factory=dict)


class SubcellularLocalizationPredictor:
    """Predicts subcellular localization of plant proteins.

    Uses signal peptide detection, transit peptide recognition,
    and NLS/NES motif scanning to predict localization to:
    - cytoplasm (default)
    - nucleus
    - mitochondria
    - chloroplast
    - secreted (ER -> extracellular)
    """

    def predict(self, protein_sequence: str) -> LocalizationResult:
        """Predict subcellular localization.

        Args:
            protein_sequence: Amino acid sequence.

        Returns:
            LocalizationResult with predicted location and confidence.
        """
        if not protein_sequence:
            raise ValueError("Protein sequence cannot be empty")

        seq = protein_sequence.upper()

        # Detect features
        has_signal = self._has_signal_peptide(seq)
        has_nls = self._has_nls(seq)
        has_nes = self._has_nes(seq)
        has_transit = self._has_transit_peptide(seq)

        # Calculate scores for each compartment
        scores = {
            "cytoplasm": 0.1,
            "nucleus": 0.0,
            "mitochondria": 0.0,
            "chloroplast": 0.0,
            "secreted": 0.0,
        }

        if has_signal:
            scores["secreted"] += 0.6
        if has_nls:
            scores["nucleus"] += 0.5
        if has_nes:
            scores["cytoplasm"] += 0.3
        if has_transit:
            # Distinguish between mitochondria and chloroplast
            if self._is_chloroplast(seq):
                scores["chloroplast"] += 0.6
            elif self._is_mitochondria(seq):
                scores["mitochondria"] += 0.6

        # Determine localization
        localization = max(scores, key=lambda k: scores[k])
        confidence = scores[localization]

        return LocalizationResult(
            localization=localization,
            confidence=min(1.0, confidence),
            has_signal_peptide=has_signal,
            has_nls=has_nls,
            has_nes=has_nes,
            has_transit_peptide=has_transit,
            all_scores=scores,
        )

    def _has_signal_peptide(self, sequence: str) -> bool:
        """Check for N-terminal signal peptide."""
        return bool(SIGNAL_PEPTIDE_PATTERN.match(sequence))

    def _has_nls(self, sequence: str) -> bool:
        """Check for nuclear localization signal."""
        return any(pattern.search(sequence) for pattern in NLS_PATTERNS)

    def _has_nes(self, sequence: str) -> bool:
        """Check for nuclear export signal."""
        return bool(NES_PATTERN.search(sequence))

    def _has_transit_peptide(self, sequence: str) -> bool:
        """Check for organelle transit peptide."""
        return bool(CHLOROPLAST_PATTERN.match(sequence) or MITOCHONDRIA_PATTERN.match(sequence))

    def _is_chloroplast(self, sequence: str) -> bool:
        """Check if transit peptide is chloroplast-targeted."""
        return bool(CHLOROPLAST_PATTERN.match(sequence))

    def _is_mitochondria(self, sequence: str) -> bool:
        """Check if transit peptide is mitochondria-targeted."""
        return bool(MITOCHONDRIA_PATTERN.match(sequence))
