"""GPS anti-spoofing demo: detect fake GPS signals."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.decision_support.gps_antispoof import GPSReading, GPSSpoofingDetector


def main():
    detector = GPSSpoofingDetector()
    reading = GPSReading(
        latitude=40.7128,
        longitude=-74.0060,
        timestamp="2026-10-04T12:00:00Z",
        speed=0.5,
        heading=90.0,
    )
    result = detector.check(reading)
    print(f"Spoofed: {result.is_spoofed}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reason: {result.reason}")


if __name__ == "__main__":
    main()
