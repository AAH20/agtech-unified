"""Tests for sequence alignment (Needleman-Wunsch and Smith-Waterman)."""

from src.genomics.alignment import needleman_wunsch, smith_waterman


class TestNeedlemanWunsch:
    """Global alignment tests."""

    def test_identical_sequences(self):
        a, b, score = needleman_wunsch("ACGT", "ACGT", 1, -1, -2)
        assert a == "ACGT"
        assert b == "ACGT"
        assert score == 4

    def test_completely_different(self):
        a, b, score = needleman_wunsch("AAAA", "TTTT", 1, -1, -2)
        assert len(a) == 4
        assert len(b) == 4
        assert score == -4

    def test_gap_insertion(self):
        a, b, score = needleman_wunsch("ACGT", "AGT", 1, -1, -2)
        assert a == "ACGT"
        assert b == "A-GT"
        assert score == 1  # 3 matches - 1 gap

    def test_empty_first_sequence(self):
        a, b, score = needleman_wunsch("", "ACGT", 1, -1, -2)
        assert a == "----"
        assert b == "ACGT"
        assert score == -8

    def test_empty_second_sequence(self):
        a, b, score = needleman_wunsch("ACGT", "", 1, -1, -2)
        assert a == "ACGT"
        assert b == "----"
        assert score == -8

    def test_both_empty(self):
        a, b, score = needleman_wunsch("", "", 1, -1, -2)
        assert a == ""
        assert b == ""
        assert score == 0

    def test_partial_overlap(self):
        a, b, score = needleman_wunsch("GATTACA", "GCATGCU", 1, -1, -1)
        # Classic NW example
        assert len(a) == len(b)
        assert score >= 0

    def test_alignment_length_preserved(self):
        a, b, score = needleman_wunsch("ACGTACGT", "ACGT", 1, -1, -2)
        assert len(a) == len(b)
        assert len(a) >= 8

    def test_score_consistency(self):
        """Score should equal sum of column scores."""
        a, b, score = needleman_wunsch("ACGT", "AGT", 1, -1, -2)
        col_score = 0
        for ca, cb in zip(a, b):
            if ca == "-" or cb == "-":
                col_score += -2
            elif ca == cb:
                col_score += 1
            else:
                col_score += -1
        assert score == col_score

    def test_custom_scores(self):
        a, b, score = needleman_wunsch("AC", "AG", 2, -3, -1)
        # Match A-A (2), then either C-G mismatch (-3) or gap+gap (-2)
        # Best: A-A match (2) + C-G mismatch (-3) = -1
        # Or: A-A (2) + gap in b (-1) + C-gap (-1) = 0
        assert score == 0

    def test_longer_sequences(self):
        seq1 = "ACGTACGTACGT"
        seq2 = "ACGTACGT"
        a, b, score = needleman_wunsch(seq1, seq2, 1, -1, -2)
        assert len(a) == len(b)
        assert score == 8 - 8  # 8 matches, 4 gaps * -2


class TestSmithWaterman:
    """Local alignment tests."""

    def test_identical_sequences(self):
        a, b, score = smith_waterman("ACGT", "ACGT", 1, -1, -2)
        assert a == "ACGT"
        assert b == "ACGT"
        assert score == 4

    def test_no_similarity(self):
        a, b, score = smith_waterman("AAAA", "TTTT", 1, -1, -2)
        assert score == 0
        assert a == ""
        assert b == ""

    def test_local_match(self):
        a, b, score = smith_waterman("ACGTACGT", "ACGT", 1, -1, -2)
        assert a == "ACGT"
        assert b == "ACGT"
        assert score == 4

    def test_partial_local_match(self):
        a, b, score = smith_waterman("GATTACA", "GCATGCU", 1, -1, -1)
        assert score > 0
        assert len(a) == len(b)

    def test_empty_sequences(self):
        a, b, score = smith_waterman("", "ACGT", 1, -1, -2)
        assert score == 0
        assert a == ""
        assert b == ""

    def test_score_never_negative(self):
        """SW score should never go below 0."""
        a, b, score = smith_waterman("TTTTAAAA", "AAAATTTT", 1, -1, -2)
        assert score >= 0

    def test_subsequence_match(self):
        """Local alignment finds best matching substring."""
        a, b, score = smith_waterman("XXACGTYY", "ACGT", 1, -1, -2)
        assert a == "ACGT"
        assert b == "ACGT"
        assert score == 4

    def test_alignment_length_preserved(self):
        a, b, score = smith_waterman("ACGTACGT", "ACGT", 1, -1, -2)
        assert len(a) == len(b)

    def test_score_consistency(self):
        """Score should equal sum of column scores."""
        a, b, score = smith_waterman("GATTACA", "GCATGCU", 1, -1, -1)
        col_score = 0
        for ca, cb in zip(a, b):
            if ca == "-" or cb == "-":
                col_score += -1
            elif ca == cb:
                col_score += 1
            else:
                col_score += -1
        assert score == col_score

    def test_both_empty(self):
        a, b, score = smith_waterman("", "", 1, -1, -2)
        assert a == ""
        assert b == ""
        assert score == 0

    def test_single_char_match(self):
        a, b, score = smith_waterman("A", "A", 1, -1, -2)
        assert a == "A"
        assert b == "A"
        assert score == 1

    def test_single_char_mismatch(self):
        a, b, score = smith_waterman("A", "T", 1, -1, -2)
        assert score == 0
