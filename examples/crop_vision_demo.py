"""Crop vision demo: disease detection and yield prediction."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.optimization.crop_vision import DiseaseDetection, YieldPrediction


def main():
    detector = DiseaseDetection()
    print(f"Disease Detection: {detector}")
    predictor = YieldPrediction()
    print(f"Yield Prediction: {predictor}")
    print("Crop vision pipeline ready")


if __name__ == "__main__":
    main()
