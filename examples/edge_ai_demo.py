"""Edge AI demo: ONNX model inference on edge devices."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.optimization.edge_ai import EdgeAIInference


def main():
    engine = EdgeAIInference(model_path="models/crop_disease.onnx")
    print(f"Model: {engine.model_path}")
    print(f"Device: {engine.device}")
    print("Ready for inference on edge devices")


if __name__ == "__main__":
    main()
