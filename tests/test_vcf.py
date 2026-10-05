"""Tests for VCF parsing and writing."""

from src.genomics.vcf import VCFRecord, parse_vcf, write_vcf


class TestVCFRecord:
    """VCFRecord dataclass tests."""

    def test_create_record(self):
        rec = VCFRecord(
            chrom="chr1",
            pos=100,
            id="rs123",
            ref="A",
            alt="G",
            qual=30.0,
            filter="PASS",
            info={"DP": 10},
        )
        assert rec.chrom == "chr1"
        assert rec.pos == 100
        assert rec.ref == "A"
        assert rec.alt == "G"

    def test_record_with_multiple_alts(self):
        rec = VCFRecord(
            chrom="chr1",
            pos=200,
            id=".",
            ref="A",
            alt=["G", "T"],
            qual=50.0,
            filter="PASS",
            info={},
        )
        assert rec.alt == ["G", "T"]


class TestParseVCF:
    """VCF parsing tests."""

    def test_parse_minimal(self):
        text = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        text += "chr1\t100\trs123\tA\tG\t30\tPASS\t.\n"
        records = parse_vcf(text)
        assert len(records) == 1
        assert records[0].chrom == "chr1"
        assert records[0].pos == 100
        assert records[0].ref == "A"
        assert records[0].alt == "G"

    def test_parse_multiple_records(self):
        text = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        text += "chr1\t100\trs123\tA\tG\t30\tPASS\t.\n"
        text += "chr1\t200\trs456\tC\tT\t40\tPASS\t.\n"
        text += "chr2\t300\t.\tG\tA\t50\tq10\t.\n"
        records = parse_vcf(text)
        assert len(records) == 3
        assert records[1].chrom == "chr1"
        assert records[2].chrom == "chr2"

    def test_parse_with_info_fields(self):
        text = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        text += "chr1\t100\trs123\tA\tG\t30\tPASS\tDP=10;AF=0.5\n"
        records = parse_vcf(text)
        assert records[0].info["DP"] == "10"
        assert records[0].info["AF"] == "0.5"

    def test_parse_with_flag_info(self):
        text = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        text += "chr1\t100\trs123\tA\tG\t30\tPASS\tDB;DP=10\n"
        records = parse_vcf(text)
        assert records[0].info["DB"] is True
        assert records[0].info["DP"] == "10"

    def test_parse_multiple_alts(self):
        text = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        text += "chr1\t100\trs123\tA\tG,T\t30\tPASS\t.\n"
        records = parse_vcf(text)
        assert records[0].alt == ["G", "T"]

    def test_parse_with_genotype(self):
        text = "##fileformat=VCFv4.2\n"
        text += "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\n"
        text += "chr1\t100\trs123\tA\tG\t30\tPASS\t.\tGT\t0/1\n"
        records = parse_vcf(text)
        assert records[0].format == ["GT"]
        assert records[0].samples["SAMPLE1"] == {"GT": "0/1"}

    def test_parse_multiple_samples(self):
        text = "##fileformat=VCFv4.2\n"
        text += "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
        text += "chr1\t100\trs123\tA\tG\t30\tPASS\t.\tGT\t0/1\t1/1\n"
        records = parse_vcf(text)
        assert records[0].samples["S1"] == {"GT": "0/1"}
        assert records[0].samples["S2"] == {"GT": "1/1"}

    def test_parse_ignores_meta_lines(self):
        text = "##fileformat=VCFv4.2\n##contig=<ID=chr1>\n"
        text += "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        text += "chr1\t100\trs123\tA\tG\t30\tPASS\t.\n"
        records = parse_vcf(text)
        assert len(records) == 1

    def test_parse_empty_vcf(self):
        text = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        records = parse_vcf(text)
        assert len(records) == 0

    def test_parse_indel(self):
        text = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        text += "chr1\t100\t.\tAT\tA\t30\tPASS\t.\n"
        records = parse_vcf(text)
        assert records[0].ref == "AT"
        assert records[0].alt == "A"

    def test_parse_symbolic_alt(self):
        text = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        text += "chr1\t100\t.\tA\t<DEL>\t30\tPASS\t.\n"
        records = parse_vcf(text)
        assert records[0].alt == "<DEL>"

    def test_parse_star_alt(self):
        text = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        text += "chr1\t100\t.\tA\t*\t30\tPASS\t.\n"
        records = parse_vcf(text)
        assert records[0].alt == "*"

    def test_parse_missing_qual(self):
        text = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        text += "chr1\t100\trs123\tA\tG\t.\tPASS\t.\n"
        records = parse_vcf(text)
        assert records[0].qual is None

    def test_parse_missing_filter(self):
        text = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        text += "chr1\t100\trs123\tA\tG\t30\t.\t.\n"
        records = parse_vcf(text)
        assert records[0].filter == "."


class TestWriteVCF:
    """VCF writing tests."""

    def test_write_minimal(self):
        rec = VCFRecord(
            chrom="chr1",
            pos=100,
            id="rs123",
            ref="A",
            alt="G",
            qual=30.0,
            filter="PASS",
            info={},
        )
        text = write_vcf([rec])
        assert "chr1" in text
        assert "100" in text
        assert "A" in text
        assert "G" in text

    def test_write_multiple_records(self):
        recs = [
            VCFRecord("chr1", 100, "rs1", "A", "G", 30.0, "PASS", {}),
            VCFRecord("chr1", 200, "rs2", "C", "T", 40.0, "PASS", {}),
        ]
        text = write_vcf(recs)
        lines = [l for l in text.strip().split("\n") if not l.startswith("#")]
        assert len(lines) == 2

    def test_write_with_info(self):
        rec = VCFRecord(
            "chr1",
            100,
            "rs123",
            "A",
            "G",
            30.0,
            "PASS",
            {"DP": "10", "AF": "0.5"},
        )
        text = write_vcf([rec])
        assert "DP=10" in text
        assert "AF=0.5" in text

    def test_write_with_flag_info(self):
        rec = VCFRecord(
            "chr1",
            100,
            "rs123",
            "A",
            "G",
            30.0,
            "PASS",
            {"DB": True, "DP": "10"},
        )
        text = write_vcf([rec])
        assert "DB" in text
        assert "DP=10" in text

    def test_write_with_genotype(self):
        rec = VCFRecord(
            "chr1",
            100,
            "rs123",
            "A",
            "G",
            30.0,
            "PASS",
            {},
            format=["GT"],
            samples={"S1": {"GT": "0/1"}},
        )
        text = write_vcf([rec])
        assert "GT" in text
        assert "0/1" in text
        assert "S1" in text

    def test_write_multiple_alts(self):
        rec = VCFRecord(
            "chr1",
            100,
            "rs123",
            "A",
            ["G", "T"],
            30.0,
            "PASS",
            {},
        )
        text = write_vcf([rec])
        assert "G,T" in text

    def test_roundtrip(self):
        """Parse -> write -> parse should preserve records."""
        text = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        text += "chr1\t100\trs123\tA\tG\t30\tPASS\tDP=10\n"
        text += "chr1\t200\trs456\tC\tT\t40\tPASS\t.\n"
        records = parse_vcf(text)
        written = write_vcf(records)
        records2 = parse_vcf(written)
        assert len(records) == len(records2)
        for r1, r2 in zip(records, records2):
            assert r1.chrom == r2.chrom
            assert r1.pos == r2.pos
            assert r1.ref == r2.ref
            assert r1.alt == r2.alt
            assert r1.qual == r2.qual
            assert r1.filter == r2.filter
            assert r1.info == r2.info

    def test_roundtrip_with_genotype(self):
        text = "##fileformat=VCFv4.2\n"
        text += "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"
        text += "chr1\t100\trs123\tA\tG\t30\tPASS\t.\tGT\t0/1\n"
        records = parse_vcf(text)
        written = write_vcf(records)
        records2 = parse_vcf(written)
        assert records2[0].samples["S1"]["GT"] == "0/1"

    def test_write_includes_header(self):
        rec = VCFRecord("chr1", 100, "rs123", "A", "G", 30.0, "PASS", {})
        text = write_vcf([rec])
        assert "##fileformat=VCFv4.2" in text
        assert "#CHROM" in text

    def test_write_missing_qual(self):
        rec = VCFRecord("chr1", 100, "rs123", "A", "G", None, "PASS", {})
        text = write_vcf([rec])
        assert "\t.\t" in text

    def test_write_empty_records(self):
        text = write_vcf([])
        assert "##fileformat=VCFv4.2" in text
        assert "#CHROM" in text
