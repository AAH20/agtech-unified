"""API gateway demo: REST API for farm management."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.decision_support.api_gateway import APIGateway


def main():
    gateway = APIGateway()
    print(f"Gateway: {gateway}")
    print("Endpoints:")
    print("  GET  /health")
    print("  POST /farms")
    print("  POST /sensors/readings")
    print("  POST /recommendations")


if __name__ == "__main__":
    main()
