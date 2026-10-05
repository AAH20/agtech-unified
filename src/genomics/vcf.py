"""VCF (Variant Call Format) parsing and writing.

Supports VCF v4.2 including multi-allelic sites, INFO flags,
genotype fields, and symbolic alleles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class VCFRecord:
    """A single VCF variant record."""

    chrom: str
    pos: int
    id: str
    ref: str
    alt: Union[str, List[str]]
    qual: Optional[float]
    filter: str
    info: Dict[str, Union[str, bool]] = field(default_factory=dict)
    format: Optional[List[str]] = None
    samples: Dict[str, Dict[str, str]] = field(default_factory=dict)


def _parse_info(info_str: str) -> Dict[str, Union[str, bool]]:
    """Parse an INFO column string into a dict."""
    if info_str == "." or not info_str:
        return {}
    result: Dict[str, Union[str, bool]] = {}
    for item in info_str.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
        else:
            result[item] = True
    return result


def _format_info(info: Dict[str, Union[str, bool]]) -> str:
    """Format an info dict into a VCF INFO column string."""
    if not info:
        return "."
    parts = []
    for key, value in info.items():
        if value is True:
            parts.append(key)
        else:
            parts.append(f"{key}={value}")
    return ";".join(parts)


def _format_alt(alt: Union[str, List[str]]) -> str:
    """Format ALT field."""
    if isinstance(alt, list):
        return ",".join(alt)
    return alt


def parse_vcf(text: str) -> List[VCFRecord]:
    """Parse VCF-formatted text into a list of VCFRecord objects.

    Args:
        text: VCF file content as a string.

    Returns:
        List of VCFRecord objects.
    """
    records: List[VCFRecord] = []
    format_fields: Optional[List[str]] = None
    sample_names: List[str] = []

    for line in text.splitlines():
        if line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            cols = line.split("\t")
            if len(cols) > 9:
                sample_names = cols[9:]
            continue
        if not line or line.startswith("#"):
            continue

        cols = line.split("\t")
        if len(cols) < 8:
            continue

        chrom = cols[0]
        pos = int(cols[1])
        rec_id = cols[2]
        ref = cols[3]
        alt_str = cols[4]
        qual_str = cols[5]
        filter_str = cols[6]
        info_str = cols[7]

        qual = float(qual_str) if qual_str != "." else None

        if alt_str == "." or not alt_str:
            alt: Union[str, List[str]] = "."
        elif "," in alt_str:
            alt = alt_str.split(",")
        else:
            alt = alt_str

        info = _parse_info(info_str)

        rec_format: Optional[List[str]] = None
        samples: Dict[str, Dict[str, str]] = {}

        if len(cols) > 8:
            format_fields = cols[8].split(":")
            rec_format = format_fields
            for i, sample_name in enumerate(sample_names):
                sample_idx = 9 + i
                if sample_idx < len(cols):
                    values = cols[sample_idx].split(":")
                    samples[sample_name] = {}
                    for j, fmt_key in enumerate(format_fields):
                        if j < len(values):
                            samples[sample_name][fmt_key] = values[j]

        records.append(
            VCFRecord(
                chrom=chrom,
                pos=pos,
                id=rec_id,
                ref=ref,
                alt=alt,
                qual=qual,
                filter=filter_str,
                info=info,
                format=rec_format,
                samples=samples,
            )
        )

    return records


def write_vcf(records: List[VCFRecord]) -> str:
    """Write VCFRecord objects to VCF-formatted text.

    Args:
        records: List of VCFRecord objects.

    Returns:
        VCF-formatted string.
    """
    lines = [
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
    ]

    # Determine if any record has genotype data
    has_genotypes = any(r.format is not None for r in records)
    sample_names: List[str] = []
    if has_genotypes:
        for r in records:
            if r.format is not None and r.samples:
                sample_names = list(r.samples.keys())
                break
        if sample_names:
            header_cols = lines[-1].split("\t") + ["FORMAT"] + sample_names
            lines[-1] = "\t".join(header_cols)

    for rec in records:
        cols = [
            rec.chrom,
            str(rec.pos),
            rec.id,
            rec.ref,
            _format_alt(rec.alt),
            str(rec.qual) if rec.qual is not None else ".",
            rec.filter,
            _format_info(rec.info),
        ]

        if has_genotypes:
            if rec.format is not None:
                cols.append(":".join(rec.format))
                for name in sample_names:
                    if name in rec.samples:
                        sample_vals = rec.samples[name]
                        vals = [sample_vals.get(k, ".") for k in rec.format]
                        cols.append(":".join(vals))
                    else:
                        cols.append(".")
            else:
                cols.append(".")

        lines.append("\t".join(cols))

    return "\n".join(lines) + "\n"
