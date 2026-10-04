"""Ontology demo: AGROVOC integration for agricultural knowledge."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.digital_twin.ontology import AGROVOCOntology


def main():
    ontology = AGROVOCOntology()
    results = ontology.search("wheat")
    for r in results:
        print(f"Concept: {r.label}, URI: {r.uri}")


if __name__ == "__main__":
    main()
