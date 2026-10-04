"""Security demo: zero-trust authentication and GPS anti-spoofing."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.decision_support.security import ZeroTrustAuth
from src.decision_support.gps_antispoof import GPSReading, GPSSpoofingDetector


def main():
    auth = ZeroTrustAuth(secret_key="demo-secret-key-32-chars-min")
    token = auth.generate_token(user_id="farmer-1", roles=["admin"])
    print(f"JWT Token: {token[:50]}...")

    reading = GPSReading(
        latitude=40.7128,
        longitude=-74.0060,
        timestamp="2026-10-04T12:00:00Z",
        speed=0.5,
        heading=90.0,
    )
    detector = GPSSpoofingDetector()
    result = detector.check(reading)
    print(f"GPS Spoofing Detected: {result.is_spoofed}")


if __name__ == "__main__":
    main()
