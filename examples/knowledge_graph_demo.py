"""Knowledge graph demo: agricultural entity management."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.digital_twin.knowledge_graph import AgriKnowledgeGraph


def main():
    kg = AgriKnowledgeGraph()
    kg.add_entity("Field-A", type="Field", area=10.5)
    kg.add_entity("Crop-Wheat", type="Crop", variety="Winter Wheat")
    kg.add_relationship("Field-A", "grows", "Crop-Wheat")
    results = kg.query("Field-A")
    print(f"Field-A relationships: {results}")
    print(f"Total entities: {len(kg.entities)}")


if __name__ == "__main__":
    main()
