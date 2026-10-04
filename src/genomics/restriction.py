"""Restriction site analysis for CRISPR construct design.

Identifies restriction enzyme sites in DNA sequences and suggests
silent mutations to remove or add sites for Golden Gate or traditional
cloning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Common restriction enzymes and their recognition sites
ENZYME_SITES: Dict[str, str] = {
    "EcoRI": "GAATTC",
    "BamHI": "GGATCC",
    "HindIII": "AAGCTT",
    "XbaI": "TCTAGA",
    "SalI": "GTCGAC",
    "PstI": "CTGCAG",
    "SphI": "GCATGC",
    "KpnI": "GGTACC",
    "SacI": "GAGCTC",
    "BglII": "AGATCT",
    "XhoI": "CTCGAG",
    "NheI": "GCTAGC",
    "NotI": "GCGGCCGC",
    "AscI": "GGCGCGCC",
    "PacI": "TTAATTAA",
    "SpeI": "ACTAGT",
    "MluI": "ACGCGT",
    "NcoI": "CCATGG",
    "AgeI": "ACCGGT",
    "BstBI": "TTCGAA",
    "ClaI": "ATCGAT",
    "NdeI": "CATATG",
    "NsiI": "ATGCAT",
    "SbfI": "CCTGCAGG",
    "SwaI": "ATTTAAAT",
    "PmeI": "GTTTAAAC",
    "FseI": "GGCCGGCC",
    "AvrII": "CCTAGG",
    "NruI": "TCGCGA",
    "StuI": "AGGCCT",
    "ScaI": "AGTACT",
    "SnaBI": "TACGTA",
    "BspEI": "TCCGGA",
    "ApaI": "GGGCCC",
    "SacII": "CCGCGG",
    "NarI": "GGCGCC",
    "EcoRV": "GATATC",
    "DraI": "TTTAAA",
    "PvuI": "CGATCG",
    "PvuII": "CAGCTG",
    "HpaI": "GTTAAC",
    "SmaI": "CCCGGG",
    "XmaI": "CCCGGG",
    "BstEII": "GGTNACC",
    "HincII": "GTYRAC",
    "Eco53kI": "GAGCTC",
    "BsaI": "GGTCTC",
    "BsmBI": "CGTCTC",
    "BbsI": "GAAGAC",
    "SapI": "GCTCTTC",
    "AarI": "CACCTGC",
    "BtgZI": "GCGATG",
}


@dataclass
class RestrictionSite:
    """A restriction enzyme recognition site."""

    enzyme: str
    position: int
    sequence: str
    strand: str = "+"


@dataclass
class MutationSuggestion:
    """A suggested silent mutation."""

    original: str
    mutated: str
    position: int
    enzyme: str
    description: str


class RestrictionSiteAnalyzer:
    """Analyzes restriction enzyme sites in DNA sequences."""

    def __init__(self, enzymes: Optional[Dict[str, str]] = None):
        self._enzymes = enzymes or ENZYME_SITES

    def find_sites(self, dna_sequence: str) -> List[RestrictionSite]:
        """Find all restriction enzyme sites in a DNA sequence.

        Args:
            dna_sequence: DNA sequence to analyze.

        Returns:
            List of RestrictionSite objects.
        """
        if not dna_sequence:
            raise ValueError("DNA sequence cannot be empty")

        seq = dna_sequence.upper()
        sites = []

        for enzyme, site in self._enzymes.items():
            site = site.upper()
            start = 0
            while True:
                pos = seq.find(site, start)
                if pos == -1:
                    break
                sites.append(
                    RestrictionSite(
                        enzyme=enzyme,
                        position=pos,
                        sequence=site,
                        strand="+",
                    )
                )
                start = pos + 1

        return sorted(sites, key=lambda s: s.position)

    def get_enzymes(self) -> List[str]:
        """Return list of available enzyme names."""
        return list(self._enzymes.keys())

    def get_site_sequence(self, enzyme: str) -> Optional[str]:
        """Get the recognition sequence for an enzyme."""
        return self._enzymes.get(enzyme)

    def suggest_silent_mutation(
        self,
        site_sequence: str,
        enzyme: str,
    ) -> Optional[MutationSuggestion]:
        """Suggest a silent mutation to remove a restriction site.

        Args:
            site_sequence: The restriction site sequence.
            enzyme: The enzyme name.

        Returns:
            MutationSuggestion or None if no silent mutation possible.
        """
        # Try single-base substitutions that don't change the protein
        # This is a simplified version - a full implementation would
        # need the reading frame and codon table
        site = site_sequence.upper()

        # Common silent mutations for common enzymes
        silent_mutations = {
            "GAATTC": "GAATTT",  # EcoRI -> silent in some frames
            "GGATCC": "GGATCT",  # BamHI -> silent
            "AAGCTT": "AAGCTC",  # HindIII -> silent
        }

        if site in silent_mutations:
            return MutationSuggestion(
                original=site,
                mutated=silent_mutations[site],
                position=0,
                enzyme=enzyme,
                description=f"Silent mutation to remove {enzyme} site",
            )

        return None

    def add_site(
        self,
        dna_sequence: str,
        enzyme: str,
        position: Optional[int] = None,
    ) -> Optional[MutationSuggestion]:
        """Suggest a mutation to add a restriction site.

        Args:
            dna_sequence: DNA sequence to modify.
            enzyme: Enzyme to add site for.
            position: Optional position to add site near.

        Returns:
            MutationSuggestion or None.
        """
        site = self._enzymes.get(enzyme)
        if not site:
            return None

        # Simplified: suggest inserting the site at the beginning
        return MutationSuggestion(
            original="",
            mutated=site,
            position=position or 0,
            enzyme=enzyme,
            description=f"Insert {enzyme} site",
        )
