"""Gene Ontology (GO) enrichment analysis.

Implements hypergeometric over-representation test for GO term enrichment
with multiple testing correction (Bonferroni, Benjamini-Hochberg).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

logger = logging.getLogger(__name__)


@dataclass
class GOEnrichmentResult:
    """Result of GO enrichment test for a single term."""

    go_id: str
    go_name: str
    namespace: str
    p_value: float
    adjusted_p_value: float
    enrichment_ratio: float
    query_gene_count: int
    background_gene_count: int
    query_genes: List[str]


def hypergeometric_pvalue(k: int, K: int, n: int, N: int) -> float:
    """Compute hypergeometric survival function (over-representation p-value).

    P(X >= k) where X ~ Hypergeometric(N, K, n).

    Args:
        k: Number of query genes annotated to the term.
        K: Number of background genes annotated to the term.
        n: Size of query gene set.
        N: Size of background gene set.

    Returns:
        P-value for over-representation (one-tailed).
    """
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n})")
    if K > N:
        raise ValueError(f"K ({K}) cannot exceed N ({N})")
    if k < 0 or K < 0 or n < 0 or N < 0:
        raise ValueError("All parameters must be non-negative")
    if k == 0:
        return 1.0

    # P(X >= k) = sum_{i=k}^{min(n,K)} C(K,i) * C(N-K, n-i) / C(N, n)
    # Use log-space computation for numerical stability
    log_denom = _log_comb(N, n)

    p_value = 0.0
    for i in range(k, min(n, K) + 1):
        if n - i > N - K:
            continue
        log_num = _log_comb(K, i) + _log_comb(N - K, n - i)
        p_value += math.exp(log_num - log_denom)

    return min(1.0, max(0.0, p_value))


def _log_comb(n: int, k: int) -> float:
    """Log of binomial coefficient C(n, k) using log-gamma."""
    if k < 0 or k > n:
        return -math.inf
    if k == 0 or k == n:
        return 0.0
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def bonferroni(pvalues: Sequence[float]) -> List[float]:
    """Bonferroni correction: multiply each p-value by number of tests."""
    m = len(pvalues)
    return [min(1.0, p * m) for p in pvalues]


def benjamini_hochberg(pvalues: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg FDR correction.

    Returns adjusted p-values that are monotonically non-decreasing.
    """
    m = len(pvalues)
    if m == 0:
        return []

    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        idx, p = indexed[rank - 1]
        val = min(prev, p * m / rank)
        adjusted[idx] = val
        prev = val
    return adjusted


class GOEnrichmentAnalyzer:
    """Performs GO enrichment analysis using hypergeometric tests.

    Given a background gene set with GO annotations and a query gene set,
    identifies GO terms significantly over-represented in the query set.
    """

    def __init__(
        self,
        background: Dict[str, Set[str]],
        go_names: Optional[Dict[str, str]] = None,
        go_namespaces: Optional[Dict[str, str]] = None,
    ):
        """Initialize with background gene set and GO annotations.

        Args:
            background: Mapping of gene_id -> set of GO term IDs.
            go_names: Optional mapping of GO ID -> human-readable name.
            go_namespaces: Optional mapping of GO ID -> namespace
                (e.g., "biological_process", "molecular_function", "cellular_component").
        """
        if not background:
            raise ValueError("Background cannot be empty")
        self.background = background
        self.go_names = go_names or {}
        self.go_namespaces = go_namespaces or {}
        self._background_size = len(background)
        self._term_to_genes = self._build_term_index()

    def _build_term_index(self) -> Dict[str, Set[str]]:
        """Build inverted index: GO term -> set of background genes."""
        term_index: Dict[str, Set[str]] = {}
        for gene, terms in self.background.items():
            for term in terms:
                term_index.setdefault(term, set()).add(gene)
        return term_index

    def enrich(
        self,
        query_genes: Sequence[str],
        p_threshold: float = 0.05,
        min_genes: int = 1,
        correction: Optional[str] = "bonferroni",
    ) -> List[GOEnrichmentResult]:
        """Run GO enrichment analysis on query gene set.

        Args:
            query_genes: Gene IDs to test for enrichment.
            p_threshold: P-value threshold for significance.
            min_genes: Minimum number of query genes annotated to a term.
            correction: Multiple testing correction method:
                "bonferroni", "bh" (Benjamini-Hochberg), or None.

        Returns:
            List of GOEnrichmentResult sorted by p-value ascending.
        """
        if not query_genes:
            raise ValueError("Query gene list cannot be empty")

        query_set = set(query_genes)
        n = len(query_set)

        # Filter to genes present in background
        query_in_background = query_set & set(self.background.keys())
        if not query_in_background:
            logger.warning("No query genes found in background")
            return []

        n = len(query_in_background)

        # Count query genes per GO term
        term_query_count: Dict[str, int] = {}
        term_query_genes: Dict[str, List[str]] = {}
        for gene in query_in_background:
            for term in self.background[gene]:
                term_query_count[term] = term_query_count.get(term, 0) + 1
                term_query_genes.setdefault(term, []).append(gene)

        # Run hypergeometric test for each term
        results = []
        for term, k in term_query_count.items():
            if k < min_genes:
                continue
            K = len(self._term_to_genes.get(term, set()))
            if K == 0:
                continue

            p_value = hypergeometric_pvalue(k=k, K=K, n=n, N=self._background_size)

            # Enrichment ratio: (k/n) / (K/N)
            enrichment_ratio = (k / n) / (K / self._background_size) if K > 0 else 0.0

            results.append(
                GOEnrichmentResult(
                    go_id=term,
                    go_name=self.go_names.get(term, term),
                    namespace=self.go_namespaces.get(term, "unknown"),
                    p_value=p_value,
                    adjusted_p_value=p_value,  # Will be updated below
                    enrichment_ratio=enrichment_ratio,
                    query_gene_count=k,
                    background_gene_count=K,
                    query_genes=sorted(term_query_genes[term]),
                )
            )

        # Apply multiple testing correction
        if correction and results:
            pvalues = [r.p_value for r in results]
            if correction == "bonferroni":
                adjusted = bonferroni(pvalues)
            elif correction == "bh":
                adjusted = benjamini_hochberg(pvalues)
            else:
                raise ValueError(f"Unknown correction method: {correction}")
            for r, adj in zip(results, adjusted):
                r.adjusted_p_value = adj

        # Filter by p-value threshold and sort
        results = [r for r in results if r.p_value < p_threshold]
        results.sort(key=lambda r: r.p_value)

        return results


def run_go_enrichment(
    background: Dict[str, Set[str]],
    query_genes: Sequence[str],
    p_threshold: float = 0.05,
    min_genes: int = 1,
    correction: Optional[str] = "bonferroni",
    go_names: Optional[Dict[str, str]] = None,
) -> List[GOEnrichmentResult]:
    """Convenience function for GO enrichment analysis.

    Args:
        background: Mapping of gene_id -> set of GO term IDs.
        query_genes: Gene IDs to test for enrichment.
        p_threshold: P-value threshold for significance.
        min_genes: Minimum number of query genes annotated to a term.
        correction: Multiple testing correction method.
        go_names: Optional mapping of GO ID -> human-readable name.

    Returns:
        List of GOEnrichmentResult sorted by p-value ascending.
    """
    analyzer = GOEnrichmentAnalyzer(background, go_names=go_names)
    return analyzer.enrich(
        query_genes,
        p_threshold=p_threshold,
        min_genes=min_genes,
        correction=correction,
    )
