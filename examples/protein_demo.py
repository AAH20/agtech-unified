"""Protein demo: sequence analysis for agricultural biotechnology."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.genomics.protein import ProteinAnalyzer


def main():
    analyzer = ProteinAnalyzer()
    result = analyzer.analyze("MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETYGDMADCCEKQEPERNECFLSHKDDSPDLPKLKPDPNTLCDEFKADEKKFWGKYLYEIARRHPYFYAPELLYYANKYNGVFQECCQAEDKGACLLPKIETMREKVLASSARQRLRCASIQKFGERALKAWSVARLSQKFPKAEFVEVTKLVTDLTKVHKECCHGDLLECADDRADLAKYICDNQDTISSKLKECCDKPVNGFNLSYVDDEAFKKDCKGKKYKDNGFVQVNYKEEFDKLVKDYQEKFLGKYFEHLLPDENQDLKGDKAKALGDFRHKLMKYLKGCMGDQYDKLKKYVEKHGKTLNDKQK")
    print(f"Molecular Weight: {result.molecular_weight:.1f} Da")
    print(f"Isoelectric Point: {result.isoelectric_point:.2f}")


if __name__ == "__main__":
    main()
