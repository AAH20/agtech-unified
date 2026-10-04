"""GPS anti-spoofing detection for agricultural IoT.

Provides signal quality analysis, anomaly detection, and spoofing checks
for GPS readings from field sensors and autonomous vehicles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

EARTH_RADIUS_KM = 6371.0


@dataclass
class GPSReading:
    """A single GPS reading from an IoT device."""

    latitude: float
    longitude: float
    altitude: float
    timestamp: float


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in kilometers."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_KM * c


class GPSSpoofingDetector:
    """Detects GPS spoofing via signal analysis and anomaly detection."""

    def __init__(
        self,
        max_speed: float = 50.0,
        max_position_jump: float = 0.01,
        min_satellites: int = 4,
        min_snr: float = 20.0,
        max_hdop: float = 5.0,
    ):
        self.max_speed = max_speed
        self.max_position_jump = max_position_jump
        self.min_satellites = min_satellites
        self.min_snr = min_snr
        self.max_hdop = max_hdop

    def analyze_signal(
        self,
        snr: float,
        cno: float,
        satellite_count: int,
        hdop: float,
    ) -> Dict[str, Any]:
        """Analyze GPS signal quality metrics.

        Returns a dict with quality rating, list of issues, and spoofing risk flag.
        """
        issues: List[str] = []

        if snr < self.min_snr:
            issues.append("low_snr")
        if satellite_count < self.min_satellites:
            issues.append("insufficient_satellites")
        if hdop > self.max_hdop:
            issues.append("high_hdop")

        if len(issues) == 0:
            quality = "good"
        elif len(issues) <= 2:
            quality = "fair"
        else:
            quality = "poor"

        return {
            "quality": quality,
            "issues": issues,
            "spoofing_risk": len(issues) > 0,
        }

    def detect_anomaly(self, history: List[GPSReading]) -> List[Dict[str, Any]]:
        """Detect anomalies in a sequence of GPS readings.

        Checks for impossible speed and position jumps between consecutive readings.
        Returns a list of anomaly dicts with type, index, and details.
        """
        anomalies: List[Dict[str, Any]] = []
        if len(history) < 2:
            return anomalies

        for i in range(1, len(history)):
            prev = history[i - 1]
            curr = history[i]
            dt = curr.timestamp - prev.timestamp
            if dt <= 0:
                continue

            dist_km = haversine_distance(
                prev.latitude,
                prev.longitude,
                curr.latitude,
                curr.longitude,
            )
            speed_kmh = (dist_km / dt) * 3600.0

            if speed_kmh > self.max_speed:
                anomalies.append(
                    {
                        "type": "impossible_speed",
                        "index": i,
                        "speed_kmh": speed_kmh,
                        "max_speed": self.max_speed,
                    }
                )

            if dist_km > self.max_position_jump:
                anomalies.append(
                    {
                        "type": "position_jump",
                        "index": i,
                        "distance_km": dist_km,
                        "max_jump": self.max_position_jump,
                    }
                )

        return anomalies

    def check_spoofing(
        self,
        reading: GPSReading,
        history: Optional[List[GPSReading]] = None,
    ) -> Dict[str, Any]:
        """Check a GPS reading for spoofing indicators.

        Combines anomaly detection against history with signal quality heuristics.
        Returns a dict with is_spoofed, confidence score, and indicator list.
        """
        indicators: List[str] = []
        confidence = 0.0

        if history:
            anomalies = self.detect_anomaly(history + [reading])
            for anomaly in anomalies:
                indicators.append(anomaly["type"])
                confidence += 0.5

        # Heuristic: readings at exactly 0,0 are suspicious
        if reading.latitude == 0.0 and reading.longitude == 0.0:
            indicators.append("null_island")
            confidence += 0.3

        confidence = min(1.0, confidence)
        is_spoofed = confidence > 0.0

        return {
            "is_spoofed": is_spoofed,
            "confidence": confidence,
            "indicators": indicators,
        }
