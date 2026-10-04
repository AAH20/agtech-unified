# Tutorial 08: Genomics

## Overview

Learn how to use the genomics module for CRISPR guide RNA design and protein analysis.

## CRISPR Guide RNA Design

```python
from src.genomics.crispr import CRISPRDesigner

designer = CRISPRDesigner()
guides = designer.design_guides(
    sequence="ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",
    pam="NGG",
    guide_length=20,
)

for guide in guides:
    print(f"Guide: {guide.sequence}, PAM: {guide.pam}, Score: {guide.score:.2f}")
```

## Protein Analysis

```python
from src.genomics.protein import ProteinAnalyzer

analyzer = ProteinAnalyzer()
result = analyzer.analyze("MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETYGDMADCCEKQEPERNECFLSHKDDSPDLPKLKPDPNTLCDEFKADEKKFWGKYLYEIARRHPYFYAPELLYYANKYNGVFQECCQAEDKGACLLPKIETMREKVLASSARQRLRCASIQKFGERALKAWSVARLSQKFPKAEFVEVTKLVTDLTKVHKECCHGDLLECADDRADLAKYICDNQDTISSKLKECCDKPVNGFNLSYVDDEAFKKDCKGKKYKDNGFVQVNYKEEFDKLVKDYQEKFLGKYFEHLLPDENQDLKGDKAKALGDFRHKLMKYLKGCMGDQYDKLKKYVEKHGKTLNDKQK")
print(f"Molecular weight: {result.molecular_weight:.1f} Da")
print(f"Isoelectric point: {result.isoelectric_point:.2f}")
```

## Next Steps

- [Tutorial 09: End-to-End Farm](09-end-to-end-farm.md)
- [Tutorial 10: Deployment](10-deployment.md)
