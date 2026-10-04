"""CRISPR guide RNA design and analysis."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class GuideRNA:
    """CRISPR guide RNA with metadata."""
    sequence: str
    pam: str
    gc_content: float
    off_target_score: float
    efficiency_score: float
    position: int


class CRISPRDesigner:
    """CRISPR guide RNA designer for agricultural biotechnology."""

    def __init__(self, pam_sequence: str = "NGG", guide_length: int = 20):
        self.pam_sequence = pam_sequence
        self.guide_length = guide_length

    def design_guide(self, dna_sequence: str) -> List[GuideRNA]:
        """Design guide RNAs for a DNA sequence."""
        if not dna_sequence:
            raise ValueError("Sequence cannot be empty")

        guides = []
        seq = dna_sequence.upper()

        # Find all PAM sites (NGG) — look for "GG" and use preceding base as N
        for i in range(len(seq) - 1):
            if seq[i] == "G" and seq[i + 1] == "G":
                # PAM is NGG where N is at i-1, GG at i,i+1
                pam_start = i - 1  # N position
                guide_start = pam_start - self.guide_length

                if guide_start < 0:
                    continue

                guide_seq = seq[guide_start:pam_start]
                pam = seq[pam_start:pam_start + 3]

                gc_content = self._gc_content(guide_seq)
                off_target = self._off_target_score(guide_seq)
                efficiency = self._efficiency_score(guide_seq, gc_content)

                guides.append(GuideRNA(
                    sequence=guide_seq,
                    pam=pam,
                    gc_content=gc_content,
                    off_target_score=off_target,
                    efficiency_score=efficiency,
                    position=guide_start,
                ))

        if not guides:
            raise ValueError("No PAM site found in sequence")

        return guides

    def _gc_content(self, sequence: str) -> float:
        """Calculate GC content of a sequence."""
        if not sequence:
            return 0.0
        gc = sum(1 for c in sequence if c in "GC")
        return gc / len(sequence)

    def _off_target_score(self, sequence: str) -> float:
        """Estimate off-target score (0-1, lower is better)."""
        # Simplified: penalize low complexity and extreme GC
        gc = self._gc_content(sequence)
        # GC content far from 0.5 increases off-target risk
        gc_penalty = abs(gc - 0.5) * 2
        # Low complexity penalty
        unique_kmers = len(set(sequence[i:i+3] for i in range(len(sequence) - 2)))
        complexity = unique_kmers / max(1, len(sequence) - 2)
        complexity_penalty = 1.0 - complexity
        return min(1.0, (gc_penalty + complexity_penalty) / 2)

    def _efficiency_score(self, sequence: str, gc_content: float) -> float:
        """Estimate guide efficiency (0-1, higher is better)."""
        # Optimal GC content is 40-60%
        gc_score = 1.0 - abs(gc_content - 0.5) * 2
        # Penalize poly-T (terminates transcription)
        poly_t_penalty = 0.0
        for i in range(len(sequence) - 3):
            if sequence[i:i+4] == "TTTT":
                poly_t_penalty = 0.3
                break
        # G at position 20 (adjacent to PAM) is favorable
        g_bonus = 0.1 if sequence[-1] == "G" else 0.0
        return max(0.0, min(1.0, gc_score - poly_t_penalty + g_bonus))
