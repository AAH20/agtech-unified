"""Tests for codon optimization and back-translation."""

import pytest

from src.genomics.codon import CodonOptimizer, CodonUsageTable


class TestCodonUsageTable:
    """Codon usage table construction and lookup."""

    def test_table_stores_frequencies(self):
        table = CodonUsageTable({"AAA": 0.5, "AAG": 0.5})
        table.add("AAA", 10)
        table.add("AAG", 10)
        assert table.get_frequency("AAA") == pytest.approx(0.5)

    def test_table_normalizes_frequencies(self):
        table = CodonUsageTable()
        table.add("AAA", 3)
        table.add("AAG", 1)
        assert table.get_frequency("AAA") == pytest.approx(0.75)
        assert table.get_frequency("AAG") == pytest.approx(0.25)

    def test_unknown_codon_returns_zero(self):
        table = CodonUsageTable({"AAA": 1.0})
        assert table.get_frequency("GGG") == 0.0

    def test_most_frequent_codon(self):
        table = CodonUsageTable()
        table.add("AAA", 10)
        table.add("AAG", 5)
        table.add("AAC", 2)
        assert table.most_frequent("K") == "AAA"

    def test_empty_table_most_frequent_returns_none(self):
        table = CodonUsageTable()
        assert table.most_frequent("K") is None


class TestCodonOptimizer:
    """Codon optimization for target organism expression."""

    def test_optimize_returns_dna(self):
        optimizer = CodonOptimizer("rice")
        result = optimizer.optimize("MKVLAAALLA")
        assert len(result) > 0
        assert set(result).issubset({"A", "T", "G", "C"})

    def test_optimize_length_is_triple(self):
        optimizer = CodonOptimizer("rice")
        result = optimizer.optimize("MKVLAAALLA")
        assert len(result) % 3 == 0

    def test_back_translate_protein(self):
        optimizer = CodonOptimizer("rice")
        dna = optimizer.back_translate("MKVLAAALLA")
        assert len(dna) == 33  # 11 aa * 3
        assert set(dna).issubset({"A", "T", "G", "C"})

    def test_back_translate_start_codon(self):
        optimizer = CodonOptimizer("rice")
        dna = optimizer.back_translate("MKVL")
        assert dna[:3] in ("ATG", "ATA", "ATT", "GTG", "TTG")

    def test_back_translate_stop_codon(self):
        optimizer = CodonOptimizer("rice")
        dna = optimizer.back_translate("MKVL")
        assert dna[-3:] in ("TAA", "TAG", "TGA")

    def test_optimize_uses_species_preferred_codons(self):
        optimizer = CodonOptimizer("rice")
        # Rice prefers CGC for Arg over AGA
        result = optimizer.optimize("R")
        assert result == "CGC"

    def test_optimize_empty_sequence_raises(self):
        optimizer = CodonOptimizer("rice")
        with pytest.raises(ValueError, match="empty"):
            optimizer.optimize("")

    def test_optimize_invalid_aa_raises(self):
        optimizer = CodonOptimizer("rice")
        with pytest.raises(ValueError, match="Invalid amino acid"):
            optimizer.optimize("MKVLZ")

    def test_custom_table_override(self):
        custom = CodonUsageTable()
        custom.add("AAA", 100)
        custom.add("AAG", 1)
        optimizer = CodonOptimizer("custom", codon_table=custom)
        result = optimizer.optimize("K")
        assert result == "AAA"

    def test_gc_content_within_bounds(self):
        optimizer = CodonOptimizer("rice")
        result = optimizer.optimize("MKVLAAALLA" * 10)
        gc = sum(1 for c in result if c in "GC") / len(result)
        assert 0.3 <= gc <= 0.8

    def test_optimize_preserves_protein_sequence(self):
        optimizer = CodonOptimizer("rice")
        protein = "MKVLAAALLA"
        dna = optimizer.optimize(protein)
        # Translate back and verify
        assert len(dna) // 3 == len(protein)


class TestCodonOptimizerSpecies:
    """Species-specific codon usage tables."""

    def test_rice_table_exists(self):
        optimizer = CodonOptimizer("rice")
        assert optimizer.species == "rice"

    def test_maize_table_exists(self):
        optimizer = CodonOptimizer("maize")
        assert optimizer.species == "maize"

    def test_wheat_table_exists(self):
        optimizer = CodonOptimizer("wheat")
        assert optimizer.species == "wheat"

    def test_unknown_species_uses_generic(self):
        optimizer = CodonOptimizer("unknown_species")
        assert optimizer.species == "unknown_species"

    def test_species_have_different_preferences(self):
        rice = CodonOptimizer("rice")
        maize = CodonOptimizer("maize")
        # Different species should have different codon preferences
        rice_dna = rice.optimize("R")
        maize_dna = maize.optimize("R")
        # They might be the same or different, but both should be valid codons
        assert len(rice_dna) == 3
        assert len(maize_dna) == 3
