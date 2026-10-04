"""Codon optimization and back-translation for agricultural biotechnology.

Provides species-specific codon usage tables and optimization algorithms
for high expression in target crop species (rice, wheat, maize).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Standard genetic code
CODON_TABLE = {
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

# Reverse mapping: amino acid -> list of codons
AA_TO_CODONS: Dict[str, list] = {}
for codon, aa in CODON_TABLE.items():
    AA_TO_CODONS.setdefault(aa, []).append(codon)

# Species-specific codon usage frequencies (simplified from Codon Usage Database)
# Values are relative frequencies per thousand
SPECIES_CODON_TABLES: Dict[str, Dict[str, float]] = {
    "rice": {
        "TTT": 18.2,
        "TTC": 22.1,
        "TTA": 8.5,
        "TTG": 13.2,
        "CTT": 12.5,
        "CTC": 15.8,
        "CTA": 7.2,
        "CTG": 18.5,
        "ATT": 16.8,
        "ATC": 18.5,
        "ATA": 9.2,
        "ATG": 22.1,
        "GTT": 15.2,
        "GTC": 16.8,
        "GTA": 8.5,
        "GTG": 18.2,
        "TCT": 12.5,
        "TCC": 14.2,
        "TCA": 8.5,
        "TCG": 10.2,
        "CCT": 14.2,
        "CCC": 16.5,
        "CCA": 12.8,
        "CCG": 15.2,
        "ACT": 12.5,
        "ACC": 15.8,
        "ACA": 10.2,
        "ACG": 12.5,
        "GCT": 15.8,
        "GCC": 18.5,
        "GCA": 12.5,
        "GCG": 14.2,
        "TAT": 12.5,
        "TAC": 15.2,
        "TAA": 0.8,
        "TAG": 0.5,
        "CAT": 10.2,
        "CAC": 12.5,
        "CAA": 14.2,
        "CAG": 16.8,
        "AAT": 15.2,
        "AAC": 18.5,
        "AAA": 22.1,
        "AAG": 28.5,
        "GAT": 18.5,
        "GAC": 22.1,
        "GAA": 25.2,
        "GAG": 32.1,
        "TGT": 8.5,
        "TGC": 10.2,
        "TGA": 0.6,
        "TGG": 12.5,
        "CGT": 8.5,
        "CGC": 12.5,
        "CGA": 6.2,
        "CGG": 10.2,
        "AGT": 10.2,
        "AGC": 12.5,
        "AGA": 8.5,
        "AGG": 10.2,
        "GGT": 14.2,
        "GGC": 18.5,
        "GGA": 12.5,
        "GGG": 14.2,
    },
    "maize": {
        "TTT": 16.5,
        "TTC": 20.2,
        "TTA": 7.8,
        "TTG": 12.5,
        "CTT": 11.8,
        "CTC": 14.5,
        "CTA": 6.8,
        "CTG": 17.2,
        "ATT": 15.2,
        "ATC": 17.2,
        "ATA": 8.5,
        "ATG": 20.8,
        "GTT": 14.2,
        "GTC": 15.8,
        "GTA": 7.8,
        "GTG": 16.8,
        "TCT": 11.8,
        "TCC": 13.5,
        "TCA": 7.8,
        "TCG": 9.5,
        "CCT": 13.5,
        "CCC": 15.8,
        "CCA": 12.2,
        "CCG": 14.2,
        "ACT": 11.8,
        "ACC": 14.5,
        "ACA": 9.5,
        "ACG": 11.8,
        "GCT": 14.5,
        "GCC": 17.2,
        "GCA": 11.8,
        "GCG": 13.5,
        "TAT": 11.8,
        "TAC": 14.2,
        "TAA": 0.7,
        "TAG": 0.4,
        "CAT": 9.5,
        "CAC": 11.8,
        "CAA": 13.5,
        "CAG": 15.8,
        "AAT": 14.2,
        "AAC": 17.2,
        "AAA": 20.8,
        "AAG": 26.5,
        "GAT": 17.2,
        "GAC": 20.8,
        "GAA": 23.5,
        "GAG": 30.2,
        "TGT": 7.8,
        "TGC": 9.5,
        "TGA": 0.5,
        "TGG": 11.8,
        "CGT": 7.8,
        "CGC": 11.8,
        "CGA": 5.8,
        "CGG": 9.5,
        "AGT": 9.5,
        "AGC": 11.8,
        "AGA": 7.8,
        "AGG": 9.5,
        "GGT": 13.5,
        "GGC": 17.2,
        "GGA": 11.8,
        "GGG": 13.5,
    },
    "wheat": {
        "TTT": 17.5,
        "TTC": 21.2,
        "TTA": 8.2,
        "TTG": 12.8,
        "CTT": 12.2,
        "CTC": 15.2,
        "CTA": 7.0,
        "CTG": 17.8,
        "ATT": 16.2,
        "ATC": 17.8,
        "ATA": 8.8,
        "ATG": 21.5,
        "GTT": 14.8,
        "GTC": 16.2,
        "GTA": 8.2,
        "GTG": 17.5,
        "TCT": 12.2,
        "TCC": 13.8,
        "TCA": 8.2,
        "TCG": 9.8,
        "CCT": 13.8,
        "CCC": 16.2,
        "CCA": 12.5,
        "CCG": 14.8,
        "ACT": 12.2,
        "ACC": 15.2,
        "ACA": 9.8,
        "ACG": 12.2,
        "GCT": 15.2,
        "GCC": 17.8,
        "GCA": 12.2,
        "GCG": 13.8,
        "TAT": 12.2,
        "TAC": 14.8,
        "TAA": 0.7,
        "TAG": 0.5,
        "CAT": 9.8,
        "CAC": 12.2,
        "CAA": 13.8,
        "CAG": 16.2,
        "AAT": 14.8,
        "AAC": 17.8,
        "AAA": 21.5,
        "AAG": 27.5,
        "GAT": 17.8,
        "GAC": 21.5,
        "GAA": 24.2,
        "GAG": 31.2,
        "TGT": 8.2,
        "TGC": 9.8,
        "TGA": 0.5,
        "TGG": 12.2,
        "CGT": 8.2,
        "CGC": 12.2,
        "CGA": 6.0,
        "CGG": 9.8,
        "AGT": 9.8,
        "AGC": 12.2,
        "AGA": 8.2,
        "AGG": 9.8,
        "GGT": 13.8,
        "GGC": 17.8,
        "GGA": 12.2,
        "GGG": 13.8,
    },
}

# Generic/average codon usage for unknown species
GENERIC_CODON_TABLE = {
    "TTT": 17.6,
    "TTC": 20.3,
    "TTA": 7.7,
    "TTG": 12.9,
    "CTT": 13.2,
    "CTC": 19.6,
    "CTA": 7.2,
    "CTG": 40.3,
    "ATT": 16.0,
    "ATC": 20.8,
    "ATA": 7.4,
    "ATG": 22.1,
    "GTT": 11.0,
    "GTC": 14.5,
    "GTA": 7.1,
    "GTG": 28.1,
    "TCT": 15.2,
    "TCC": 17.7,
    "TCA": 12.2,
    "TCG": 4.4,
    "CCT": 17.5,
    "CCC": 19.8,
    "CCA": 16.9,
    "CCG": 6.9,
    "ACT": 13.1,
    "ACC": 18.9,
    "ACA": 15.1,
    "ACG": 6.1,
    "GCT": 16.0,
    "GCC": 28.0,
    "GCA": 10.7,
    "GCG": 10.7,
    "TAT": 12.2,
    "TAC": 15.3,
    "TAA": 1.0,
    "TAG": 0.8,
    "CAT": 10.9,
    "CAC": 15.1,
    "CAA": 12.3,
    "CAG": 34.2,
    "AAT": 17.0,
    "AAC": 19.1,
    "AAA": 24.4,
    "AAG": 31.9,
    "GAT": 21.8,
    "GAC": 30.7,
    "GAA": 29.0,
    "GAG": 39.6,
    "TGT": 10.6,
    "TGC": 12.6,
    "TGA": 1.6,
    "TGG": 13.2,
    "CGT": 4.5,
    "CGC": 10.4,
    "CGA": 6.2,
    "CGG": 11.4,
    "AGT": 12.1,
    "AGC": 19.5,
    "AGA": 12.2,
    "AGG": 12.0,
    "GGT": 10.8,
    "GGC": 22.2,
    "GGA": 10.8,
    "GGG": 16.5,
}


@dataclass
class CodonUsageTable:
    """Codon usage frequency table for a species."""

    _frequencies: Dict[str, float] = field(default_factory=dict)
    _counts: Dict[str, int] = field(default_factory=dict)

    def __init__(self, frequencies: Optional[Dict[str, float]] = None):
        self._frequencies = dict(frequencies) if frequencies else {}
        self._counts: Dict[str, int] = {}

    def add(self, codon: str, count: int) -> None:
        """Add a codon observation to the table."""
        self._counts[codon] = self._counts.get(codon, 0) + count
        total = sum(self._counts.values())
        if total > 0:
            self._frequencies = {c: n / total for c, n in self._counts.items()}

    def get_frequency(self, codon: str) -> float:
        """Get the relative frequency of a codon."""
        return self._frequencies.get(codon, 0.0)

    def most_frequent(self, amino_acid: str) -> Optional[str]:
        """Return the most frequent codon for an amino acid."""
        codons = AA_TO_CODONS.get(amino_acid, [])
        if not codons:
            return None
        best = max(codons, key=lambda c: self._frequencies.get(c, 0.0))
        return best if self._frequencies.get(best, 0.0) > 0 else None

    def get_all_frequencies(self) -> Dict[str, float]:
        """Return a copy of all codon frequencies."""
        return dict(self._frequencies)


class CodonOptimizer:
    """Codon optimizer for target organism expression.

    Optimizes DNA sequences for high expression in target crop species
    using species-specific codon usage tables.
    """

    def __init__(
        self,
        species: str = "rice",
        codon_table: Optional[CodonUsageTable] = None,
    ):
        self.species = species.lower()
        if codon_table:
            self._table = codon_table
        else:
            freqs = SPECIES_CODON_TABLES.get(self.species, GENERIC_CODON_TABLE)
            self._table = CodonUsageTable(freqs)

    def optimize(self, protein_sequence: str) -> str:
        """Optimize a protein sequence to DNA for the target species.

        Args:
            protein_sequence: Amino acid sequence to optimize.

        Returns:
            Optimized DNA sequence.
        """
        if not protein_sequence:
            raise ValueError("Protein sequence cannot be empty")

        seq = protein_sequence.upper()
        for aa in seq:
            if aa not in AA_TO_CODONS:
                raise ValueError(f"Invalid amino acid: {aa}")

        codons = []
        for aa in seq:
            codon = self._table.most_frequent(aa)
            if codon is None:
                # Fallback to first available codon
                codon = AA_TO_CODONS[aa][0]
            codons.append(codon)

        return "".join(codons)

    def back_translate(self, protein_sequence: str) -> str:
        """Back-translate a protein sequence to DNA.

        Adds start codon (ATG) at the beginning and stop codon at the end.

        Args:
            protein_sequence: Amino acid sequence.

        Returns:
            DNA sequence with start and stop codons.
        """
        if not protein_sequence:
            raise ValueError("Protein sequence cannot be empty")

        seq = protein_sequence.upper()
        for aa in seq:
            if aa not in AA_TO_CODONS:
                raise ValueError(f"Invalid amino acid: {aa}")

        # Start codon
        dna = "ATG"

        # Internal codons
        for aa in seq:
            if aa == "M":
                continue  # Already added start codon
            codon = self._table.most_frequent(aa)
            if codon is None:
                codon = AA_TO_CODONS[aa][0]
            dna += codon

        # Stop codon (use most frequent stop codon)
        stop_codons = ["TAA", "TAG", "TGA"]
        best_stop = max(
            stop_codons,
            key=lambda c: self._table.get_frequency(c),
        )
        dna += best_stop

        return dna

    def get_codon_adaptation_index(self, dna_sequence: str) -> float:
        """Calculate Codon Adaptation Index (CAI) for a DNA sequence.

        Args:
            dna_sequence: DNA sequence to analyze.

        Returns:
            CAI score between 0 and 1.
        """
        if not dna_sequence or len(dna_sequence) < 3:
            return 0.0

        seq = dna_sequence.upper()
        scores = []
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i : i + 3]
            if codon in self._table.get_all_frequencies():
                scores.append(self._table.get_frequency(codon))

        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def get_gc_content(self, dna_sequence: str) -> float:
        """Calculate GC content of a DNA sequence.

        Args:
            dna_sequence: DNA sequence.

        Returns:
            GC content as a fraction between 0 and 1.
        """
        if not dna_sequence:
            return 0.0
        seq = dna_sequence.upper()
        gc = sum(1 for c in seq if c in "GC")
        return gc / len(seq)
