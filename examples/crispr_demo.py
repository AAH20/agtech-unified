"""CRISPR demo: guide RNA design for crop improvement."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.genomics.crispr import CRISPRDesigner


def main():
    designer = CRISPRDesigner()
    guides = designer.design_guides(
        sequence="ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",
        pam="NGG",
        guide_length=20,
    )
    for guide in guides:
        print(f"Guide: {guide.sequence}, PAM: {guide.pam}, Score: {guide.score:.2f}")


if __name__ == "__main__":
    main()
