"""Sequence alignment algorithms: Needleman-Wunsch (global) and Smith-Waterman (local).

Pure-Python implementations with affine-gap-free scoring suitable for
DNA and protein sequences in agricultural genomics pipelines.
"""

from __future__ import annotations

from typing import List, Tuple


def _init_matrix(rows: int, cols: int) -> List[List[int]]:
    """Create a rows x cols matrix initialized to 0."""
    return [[0] * cols for _ in range(rows)]


def _traceback_global(
    score_matrix: List[List[int]],
    seq_a: str,
    seq_b: str,
    match_score: int,
    mismatch_score: int,
    gap_score: int,
) -> Tuple[str, str]:
    """Trace back through a Needleman-Wunsch score matrix."""
    aligned_a: List[str] = []
    aligned_b: List[str] = []
    i, j = len(seq_a), len(seq_b)

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            diag = score_matrix[i - 1][j - 1]
            if seq_a[i - 1] == seq_b[j - 1]:
                diag += match_score
            else:
                diag += mismatch_score
            if score_matrix[i][j] == diag:
                aligned_a.append(seq_a[i - 1])
                aligned_b.append(seq_b[j - 1])
                i -= 1
                j -= 1
                continue

        if i > 0:
            up = score_matrix[i - 1][j] + gap_score
            if score_matrix[i][j] == up:
                aligned_a.append(seq_a[i - 1])
                aligned_b.append("-")
                i -= 1
                continue

        # j > 0
        aligned_a.append("-")
        aligned_b.append(seq_b[j - 1])
        j -= 1

    return "".join(reversed(aligned_a)), "".join(reversed(aligned_b))


def _traceback_local(
    score_matrix: List[List[int]],
    seq_a: str,
    seq_b: str,
    match_score: int,
    mismatch_score: int,
    gap_score: int,
) -> Tuple[str, str, int, int, int, int]:
    """Trace back through a Smith-Waterman score matrix from the max cell."""
    # Find the maximum score position
    max_score = 0
    max_i, max_j = 0, 0
    for i in range(1, len(seq_a) + 1):
        for j in range(1, len(seq_b) + 1):
            if score_matrix[i][j] > max_score:
                max_score = score_matrix[i][j]
                max_i, max_j = i, j

    aligned_a: List[str] = []
    aligned_b: List[str] = []
    i, j = max_i, max_j

    while i > 0 and j > 0 and score_matrix[i][j] > 0:
        diag = score_matrix[i - 1][j - 1]
        if seq_a[i - 1] == seq_b[j - 1]:
            diag += match_score
        else:
            diag += mismatch_score

        if score_matrix[i][j] == diag:
            aligned_a.append(seq_a[i - 1])
            aligned_b.append(seq_b[j - 1])
            i -= 1
            j -= 1
        elif score_matrix[i][j] == score_matrix[i - 1][j] + gap_score:
            aligned_a.append(seq_a[i - 1])
            aligned_b.append("-")
            i -= 1
        else:
            aligned_a.append("-")
            aligned_b.append(seq_b[j - 1])
            j -= 1

    return "".join(reversed(aligned_a)), "".join(reversed(aligned_b)), max_i, max_j, i, j


def needleman_wunsch(
    seq_a: str,
    seq_b: str,
    match_score: int = 1,
    mismatch_score: int = -1,
    gap_score: int = -2,
) -> Tuple[str, str, int]:
    """Perform global sequence alignment using the Needleman-Wunsch algorithm.

    Args:
        seq_a: First sequence (DNA or protein).
        seq_b: Second sequence (DNA or protein).
        match_score: Score for a match (default 1).
        mismatch_score: Score for a mismatch (default -1).
        gap_score: Score for a gap (default -2).

    Returns:
        Tuple of (aligned_seq_a, aligned_seq_b, alignment_score).
    """
    if not seq_a and not seq_b:
        return "", "", 0

    rows = len(seq_a) + 1
    cols = len(seq_b) + 1
    score_matrix = _init_matrix(rows, cols)

    # Initialize first row and column with gap penalties
    for i in range(1, rows):
        score_matrix[i][0] = i * gap_score
    for j in range(1, cols):
        score_matrix[0][j] = j * gap_score

    # Fill the matrix
    for i in range(1, rows):
        for j in range(1, cols):
            if seq_a[i - 1] == seq_b[j - 1]:
                diag = score_matrix[i - 1][j - 1] + match_score
            else:
                diag = score_matrix[i - 1][j - 1] + mismatch_score
            up = score_matrix[i - 1][j] + gap_score
            left = score_matrix[i][j - 1] + gap_score
            score_matrix[i][j] = max(diag, up, left)

    aligned_a, aligned_b = _traceback_global(
        score_matrix, seq_a, seq_b, match_score, mismatch_score, gap_score
    )
    return aligned_a, aligned_b, score_matrix[rows - 1][cols - 1]


def smith_waterman(
    seq_a: str,
    seq_b: str,
    match_score: int = 1,
    mismatch_score: int = -1,
    gap_score: int = -2,
) -> Tuple[str, str, int]:
    """Perform local sequence alignment using the Smith-Waterman algorithm.

    Args:
        seq_a: First sequence (DNA or protein).
        seq_b: Second sequence (DNA or protein).
        match_score: Score for a match (default 1).
        mismatch_score: Score for a mismatch (default -1).
        gap_score: Score for a gap (default -2).

    Returns:
        Tuple of (aligned_seq_a, aligned_seq_b, alignment_score).
    """
    if not seq_a or not seq_b:
        return "", "", 0

    rows = len(seq_a) + 1
    cols = len(seq_b) + 1
    score_matrix = _init_matrix(rows, cols)

    # Fill the matrix
    for i in range(1, rows):
        for j in range(1, cols):
            if seq_a[i - 1] == seq_b[j - 1]:
                diag = score_matrix[i - 1][j - 1] + match_score
            else:
                diag = score_matrix[i - 1][j - 1] + mismatch_score
            up = score_matrix[i - 1][j] + gap_score
            left = score_matrix[i][j - 1] + gap_score
            score_matrix[i][j] = max(0, diag, up, left)

    aligned_a, aligned_b, _, _, _, _ = _traceback_local(
        score_matrix, seq_a, seq_b, match_score, mismatch_score, gap_score
    )
    max_score = max(max(row) for row in score_matrix)
    return aligned_a, aligned_b, max_score
