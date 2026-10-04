"""Tests for restriction site analysis and GFF/GTF format support."""

import pytest

from src.genomics.formats import GFF3Parser, GFF3Writer, GTFParser
from src.genomics.restriction import RestrictionSiteAnalyzer


class TestRestrictionSiteAnalyzer:
    """Restriction enzyme site analysis."""

    def test_find_ecori_site(self):
        analyzer = RestrictionSiteAnalyzer()
        sites = analyzer.find_sites("ATCGGAATTCGATCG")
        ecori_sites = [s for s in sites if s.enzyme == "EcoRI"]
        assert len(ecori_sites) == 1
        assert ecori_sites[0].position == 4

    def test_find_bamhi_site(self):
        analyzer = RestrictionSiteAnalyzer()
        sites = analyzer.find_sites("ATCGGGATCCGATCG")
        bamhi_sites = [s for s in sites if s.enzyme == "BamHI"]
        assert len(bamhi_sites) == 1

    def test_find_hindiii_site(self):
        analyzer = RestrictionSiteAnalyzer()
        sites = analyzer.find_sites("ATCGAAGCTTGATCG")
        hindiii_sites = [s for s in sites if s.enzyme == "HindIII"]
        assert len(hindiii_sites) == 1

    def test_no_sites_found(self):
        analyzer = RestrictionSiteAnalyzer()
        sites = analyzer.find_sites("ATCGATCGATCG")
        # Filter out common enzymes that match short sequences
        common = [
            s for s in sites if s.enzyme in ("EcoRI", "BamHI", "HindIII", "XbaI", "SalI", "PstI")
        ]
        assert len(common) == 0

    def test_multiple_sites(self):
        analyzer = RestrictionSiteAnalyzer()
        sites = analyzer.find_sites("ATCGGAATTCGATCGGAATTCGATCG")
        ecori_sites = [s for s in sites if s.enzyme == "EcoRI"]
        assert len(ecori_sites) == 2

    def test_get_site_sequence(self):
        analyzer = RestrictionSiteAnalyzer()
        sites = analyzer.find_sites("ATCGGAATTCGATCG")
        assert sites[0].sequence == "GAATTC"

    def test_silent_mutation_suggestion(self):
        analyzer = RestrictionSiteAnalyzer()
        # Suggest silent mutation to remove EcoRI site
        suggestion = analyzer.suggest_silent_mutation("GAATTC", "EcoRI")
        assert suggestion is not None
        assert suggestion.original == "GAATTC"
        assert suggestion.mutated != "GAATTC"

    def test_add_restriction_site(self):
        analyzer = RestrictionSiteAnalyzer()
        # Suggest mutation to add EcoRI site
        suggestion = analyzer.add_site("ATCGATCG", "EcoRI")
        assert suggestion is not None

    def test_empty_sequence_raises(self):
        analyzer = RestrictionSiteAnalyzer()
        with pytest.raises(ValueError, match="empty"):
            analyzer.find_sites("")

    def test_common_enzymes_available(self):
        analyzer = RestrictionSiteAnalyzer()
        enzymes = analyzer.get_enzymes()
        assert "EcoRI" in enzymes
        assert "BamHI" in enzymes
        assert "HindIII" in enzymes


class TestGFF3Parser:
    """GFF3 format parsing."""

    def test_parse_simple_feature(self):
        parser = GFF3Parser()
        line = "chr1\t.\tgene\t100\t200\t.\t+\t.\tID=gene1"
        feature = parser.parse_line(line)
        assert feature.chrom == "chr1"
        assert feature.start == 100
        assert feature.end == 200
        assert feature.strand == "+"

    def test_parse_cds_feature(self):
        parser = GFF3Parser()
        line = "chr1\t.\tCDS\t100\t200\t.\t+\t0\tParent=gene1"
        feature = parser.parse_line(line)
        assert feature.feature_type == "CDS"
        assert feature.phase == 0

    def test_parse_multiple_features(self):
        parser = GFF3Parser()
        lines = [
            "chr1\t.\tgene\t100\t200\t.\t+\t.\tID=gene1",
            "chr1\t.\tCDS\t100\t200\t.\t+\t0\tParent=gene1",
        ]
        features = parser.parse_lines(lines)
        assert len(features) == 2

    def test_parse_attributes(self):
        parser = GFF3Parser()
        line = "chr1\t.\tgene\t100\t200\t.\t+\t.\tID=gene1;Name=test"
        feature = parser.parse_line(line)
        assert feature.attributes["ID"] == "gene1"
        assert feature.attributes["Name"] == "test"

    def test_parse_empty_line_raises(self):
        parser = GFF3Parser()
        with pytest.raises(ValueError, match="empty"):
            parser.parse_line("")


class TestGFF3Writer:
    """GFF3 format writing."""

    def test_write_feature(self):
        writer = GFF3Writer()
        feature = writer.create_feature(
            chrom="chr1",
            feature_type="gene",
            start=100,
            end=200,
            strand="+",
            attributes={"ID": "gene1"},
        )
        line = writer.write_feature(feature)
        assert "chr1" in line
        assert "gene" in line
        assert "100" in line
        assert "200" in line

    def test_write_multiple_features(self):
        writer = GFF3Writer()
        features = [
            writer.create_feature("chr1", "gene", 100, 200, "+", attributes={"ID": "g1"}),
            writer.create_feature("chr1", "CDS", 100, 200, "+", attributes={"Parent": "g1"}),
        ]
        lines = writer.write_features(features)
        assert len(lines) == 2


class TestGTFParser:
    """GTF format parsing."""

    def test_parse_transcript(self):
        parser = GTFParser()
        line = 'chr1\t.\ttranscript\t100\t200\t.\t+\t.\tgene_id "g1"; transcript_id "t1";'
        feature = parser.parse_line(line)
        assert feature.chrom == "chr1"
        assert feature.feature_type == "transcript"

    def test_parse_cds(self):
        parser = GTFParser()
        line = 'chr1\t.\tCDS\t100\t200\t.\t+\t0\tgene_id "g1"; transcript_id "t1";'
        feature = parser.parse_line(line)
        assert feature.feature_type == "CDS"
        assert feature.phase == 0

    def test_parse_attributes(self):
        parser = GTFParser()
        line = 'chr1\t.\tgene\t100\t200\t.\t+\t.\tgene_id "g1"; gene_name "test";'
        feature = parser.parse_line(line)
        assert feature.attributes["gene_id"] == "g1"
        assert feature.attributes["gene_name"] == "test"
