"""CRISPR guide RNA design and analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

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
                pam = seq[pam_start : pam_start + 3]

                gc_content = self._gc_content(guide_seq)
                off_target = self._off_target_score(guide_seq)
                efficiency = self._efficiency_score(guide_seq, gc_content)

                guides.append(
                    GuideRNA(
                        sequence=guide_seq,
                        pam=pam,
                        gc_content=gc_content,
                        off_target_score=off_target,
                        efficiency_score=efficiency,
                        position=guide_start,
                    )
                )

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
        unique_kmers = len(set(sequence[i : i + 3] for i in range(len(sequence) - 2)))
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
            if sequence[i : i + 4] == "TTTT":
                poly_t_penalty = 0.3
                break
        # G at position 20 (adjacent to PAM) is favorable
        g_bonus = 0.1 if sequence[-1] == "G" else 0.0
        return max(0.0, min(1.0, gc_score - poly_t_penalty + g_bonus))


# Standard genetic code
_CODON_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

# Most common codon per amino acid per organism
_ORGANISM_CODON_PREFERENCE = {
    "human": {
        "F": "TTC",
        "L": "CTG",
        "I": "ATC",
        "M": "ATG",
        "V": "GTG",
        "S": "TCC",
        "P": "CCC",
        "T": "ACC",
        "A": "GCC",
        "Y": "TAC",
        "H": "CAC",
        "Q": "CAG",
        "N": "AAC",
        "K": "AAG",
        "D": "GAC",
        "E": "GAG",
        "C": "TGC",
        "W": "TGG",
        "R": "CGC",
        "G": "GGC",
    },
    "e_coli": {
        "F": "TTC",
        "L": "CTG",
        "I": "ATC",
        "M": "ATG",
        "V": "GTG",
        "S": "TCT",
        "P": "CCG",
        "T": "ACC",
        "A": "GCT",
        "Y": "TAT",
        "H": "CAT",
        "Q": "CAG",
        "N": "AAC",
        "K": "AAA",
        "D": "GAT",
        "E": "GAA",
        "C": "TGC",
        "W": "TGG",
        "R": "CGT",
        "G": "GGT",
    },
    "yeast": {
        "F": "TTC",
        "L": "TTG",
        "I": "ATT",
        "M": "ATG",
        "V": "GTT",
        "S": "TCT",
        "P": "CCA",
        "T": "ACT",
        "A": "GCT",
        "Y": "TAT",
        "H": "CAT",
        "Q": "CAA",
        "N": "AAT",
        "K": "AAA",
        "D": "GAT",
        "E": "GAA",
        "C": "TGT",
        "W": "TGG",
        "R": "AGA",
        "G": "GGT",
    },
}


def codon_optimize(sequence: str, organism: str = "human") -> str:
    """Replace rare codons with common ones for the target organism.

    Args:
        sequence: DNA sequence to optimize (length must be multiple of 3).
        organism: Target organism ("human", "e_coli", or "yeast").

    Returns:
        Optimized DNA sequence with preferred codons.
    """
    if not sequence:
        raise ValueError("Sequence cannot be empty")
    seq = sequence.upper()
    if len(seq) % 3 != 0:
        raise ValueError("Sequence length must be a multiple of 3")
    organism_key = organism.lower()
    if organism_key not in _ORGANISM_CODON_PREFERENCE:
        raise ValueError(f"Unsupported organism: {organism}")
    preference = _ORGANISM_CODON_PREFERENCE[organism_key]
    optimized = []
    for i in range(0, len(seq), 3):
        codon = seq[i : i + 3]
        if codon not in _CODON_TABLE:
            raise ValueError(f"Invalid codon: {codon}")
        aa = _CODON_TABLE[codon]
        if aa == "*":
            optimized.append(codon)
        else:
            optimized.append(preference.get(aa, codon))
    return "".join(optimized)
