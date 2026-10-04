"""Context broker demo: FIWARE NGSI-LD integration."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.iot.context_broker import NGSILDBroker


def main():
    broker = NGSILDBroker(url="http://localhost:1026")
    entity = {
        "id": "urn:ngsi-ld:Field:001",
        "type": "Field",
        "area": {"type": "Property", "value": 10.5},
        "crop": {"type": "Property", "value": "Wheat"},
    }
    print(f"Entity: {entity['id']}")
    print(f"Type: {entity['type']}")
    print(f"Area: {entity['area']['value']} ha")


if __name__ == "__main__":
    main()
