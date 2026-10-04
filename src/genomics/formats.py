"""GFF3/GTF genome annotation format support.

Parsers and writers for GFF3 and GTF formats, enabling integration
with standard genome annotation pipelines and export of CRISPR targets
with genomic coordinates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GFF3Feature:
    """A GFF3 feature record."""

    chrom: str
    source: str
    feature_type: str
    start: int
    end: int
    score: Optional[float]
    strand: str
    phase: Optional[int]
    attributes: Dict[str, str] = field(default_factory=dict)


class GFF3Parser:
    """Parser for GFF3 format files."""

    def parse_line(self, line: str) -> GFF3Feature:
        """Parse a single GFF3 line.

        Args:
            line: A GFF3 format line.

        Returns:
            GFF3Feature object.
        """
        if not line or not line.strip():
            raise ValueError("Line cannot be empty")

        parts = line.strip().split("\t")
        if len(parts) < 9:
            raise ValueError(f"Invalid GFF3 line: expected 9 columns, got {len(parts)}")

        attributes = self._parse_attributes(parts[8])

        return GFF3Feature(
            chrom=parts[0],
            source=parts[1],
            feature_type=parts[2],
            start=int(parts[3]),
            end=int(parts[4]),
            score=float(parts[5]) if parts[5] != "." else None,
            strand=parts[6],
            phase=int(parts[7]) if parts[7] != "." else None,
            attributes=attributes,
        )

    def parse_lines(self, lines: List[str]) -> List[GFF3Feature]:
        """Parse multiple GFF3 lines.

        Args:
            lines: List of GFF3 format lines.

        Returns:
            List of GFF3Feature objects.
        """
        features = []
        for line in lines:
            if line.strip() and not line.startswith("#"):
                features.append(self.parse_line(line))
        return features

    def _parse_attributes(self, attr_string: str) -> Dict[str, str]:
        """Parse the GFF3 attributes column."""
        attributes = {}
        for pair in attr_string.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            if "=" in pair:
                key, value = pair.split("=", 1)
                attributes[key] = value
        return attributes


class GFF3Writer:
    """Writer for GFF3 format files."""

    def create_feature(
        self,
        chrom: str,
        feature_type: str,
        start: int,
        end: int,
        strand: str = "+",
        source: str = ".",
        score: Optional[float] = None,
        phase: Optional[int] = None,
        attributes: Optional[Dict[str, str]] = None,
    ) -> GFF3Feature:
        """Create a GFF3 feature.

        Args:
            chrom: Chromosome name.
            feature_type: Feature type (e.g., "gene", "CDS").
            start: Start position (1-based).
            end: End position.
            strand: Strand ("+" or "-").
            source: Source column.
            score: Score value.
            phase: Phase (0, 1, 2, or None).
            attributes: Feature attributes.

        Returns:
            GFF3Feature object.
        """
        return GFF3Feature(
            chrom=chrom,
            source=source,
            feature_type=feature_type,
            start=start,
            end=end,
            score=score,
            strand=strand,
            phase=phase,
            attributes=attributes or {},
        )

    def write_feature(self, feature: GFF3Feature) -> str:
        """Write a single GFF3 feature as a line.

        Args:
            feature: GFF3Feature to write.

        Returns:
            GFF3 format line.
        """
        score = "." if feature.score is None else str(feature.score)
        phase = "." if feature.phase is None else str(feature.phase)
        attrs = self._format_attributes(feature.attributes)

        return "\t".join(
            [
                feature.chrom,
                feature.source,
                feature.feature_type,
                str(feature.start),
                str(feature.end),
                score,
                feature.strand,
                phase,
                attrs,
            ]
        )

    def write_features(self, features: List[GFF3Feature]) -> List[str]:
        """Write multiple GFF3 features.

        Args:
            features: List of GFF3Feature objects.

        Returns:
            List of GFF3 format lines.
        """
        return [self.write_feature(f) for f in features]

    def _format_attributes(self, attributes: Dict[str, str]) -> str:
        """Format attributes dict as GFF3 attribute string."""
        if not attributes:
            return "."
        return ";".join(f"{k}={v}" for k, v in attributes.items())


class GTFParser:
    """Parser for GTF format files."""

    def parse_line(self, line: str) -> GFF3Feature:
        """Parse a single GTF line.

        Args:
            line: A GTF format line.

        Returns:
            GFF3Feature object.
        """
        if not line or not line.strip():
            raise ValueError("Line cannot be empty")

        parts = line.strip().split("\t")
        if len(parts) < 9:
            raise ValueError(f"Invalid GTF line: expected 9 columns, got {len(parts)}")

        attributes = self._parse_attributes(parts[8])

        return GFF3Feature(
            chrom=parts[0],
            source=parts[1],
            feature_type=parts[2],
            start=int(parts[3]),
            end=int(parts[4]),
            score=float(parts[5]) if parts[5] != "." else None,
            strand=parts[6],
            phase=int(parts[7]) if parts[7] != "." else None,
            attributes=attributes,
        )

    def _parse_attributes(self, attr_string: str) -> Dict[str, str]:
        """Parse the GTF attributes column."""
        attributes = {}
        for pair in attr_string.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            if " " in pair:
                key, value = pair.split(" ", 1)
                value = value.strip('"')
                attributes[key] = value
        return attributes
