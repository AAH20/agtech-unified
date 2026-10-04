"""Tests for GPSSpoofingDetector — signal analysis, anomaly detection."""

import pytest

from src.decision_support.gps_antispoof import (
    GPSReading,
    GPSSpoofingDetector,
    haversine_distance,
)


class TestSignalAnalysis:
    def test_analyze_signal_good_quality(self):
        detector = GPSSpoofingDetector()
        result = detector.analyze_signal(snr=35.0, cno=40.0, satellite_count=8, hdop=1.2)
        assert result["quality"] == "good"
        assert result["issues"] == []
        assert result["spoofing_risk"] is False

    def test_analyze_signal_low_snr(self):
        detector = GPSSpoofingDetector()
        result = detector.analyze_signal(snr=15.0, cno=40.0, satellite_count=8, hdop=1.2)
        assert "low_snr" in result["issues"]
        assert result["spoofing_risk"] is True

    def test_analyze_signal_insufficient_satellites(self):
        detector = GPSSpoofingDetector()
        result = detector.analyze_signal(snr=35.0, cno=40.0, satellite_count=2, hdop=1.2)
        assert "insufficient_satellites" in result["issues"]

    def test_analyze_signal_high_hdop(self):
        detector = GPSSpoofingDetector()
        result = detector.analyze_signal(snr=35.0, cno=40.0, satellite_count=8, hdop=8.0)
        assert "high_hdop" in result["issues"]

    def test_analyze_signal_poor_quality_multiple_issues(self):
        detector = GPSSpoofingDetector()
        result = detector.analyze_signal(snr=10.0, cno=15.0, satellite_count=2, hdop=10.0)
        assert result["quality"] == "poor"
        assert len(result["issues"]) >= 3


class TestAnomalyDetection:
    def test_detect_anomaly_clean_history(self):
        detector = GPSSpoofingDetector(max_speed=50.0)
        history = [
            GPSReading(latitude=40.0, longitude=-74.0, altitude=10.0, timestamp=0.0),
            GPSReading(latitude=40.00005, longitude=-74.0, altitude=10.0, timestamp=1.0),
            GPSReading(latitude=40.0001, longitude=-74.0, altitude=10.0, timestamp=2.0),
        ]
        anomalies = detector.detect_anomaly(history)
        assert anomalies == []

    def test_detect_anomaly_impossible_speed(self):
        detector = GPSSpoofingDetector(max_speed=50.0)
        history = [
            GPSReading(latitude=40.0, longitude=-74.0, altitude=10.0, timestamp=0.0),
            GPSReading(latitude=41.0, longitude=-74.0, altitude=10.0, timestamp=1.0),
        ]
        anomalies = detector.detect_anomaly(history)
        assert len(anomalies) >= 1
        assert anomalies[0]["type"] == "impossible_speed"

    def test_detect_anomaly_position_jump(self):
        detector = GPSSpoofingDetector(max_speed=500.0, max_position_jump=0.005)
        history = [
            GPSReading(latitude=40.0, longitude=-74.0, altitude=10.0, timestamp=0.0),
            GPSReading(latitude=40.1, longitude=-74.0, altitude=10.0, timestamp=0.5),
        ]
        anomalies = detector.detect_anomaly(history)
        assert any(a["type"] == "position_jump" for a in anomalies)


class TestSpoofingCheck:
    def test_check_spoofing_clean_reading(self):
        detector = GPSSpoofingDetector()
        reading = GPSReading(latitude=40.0, longitude=-74.0, altitude=10.0, timestamp=0.0)
        result = detector.check_spoofing(reading)
        assert result["is_spoofed"] is False
        assert result["confidence"] == 0.0

    def test_check_spoofing_detects_anomaly(self):
        detector = GPSSpoofingDetector(max_speed=50.0)
        history = [
            GPSReading(latitude=40.0, longitude=-74.0, altitude=10.0, timestamp=0.0),
        ]
        reading = GPSReading(latitude=41.0, longitude=-74.0, altitude=10.0, timestamp=1.0)
        result = detector.check_spoofing(reading, history=history)
        assert result["is_spoofed"] is True
        assert result["confidence"] > 0.0
        assert "impossible_speed" in result["indicators"]


class TestHaversineDistance:
    def test_haversine_distance_calculation(self):
        # Distance between two known points ~111 km per degree latitude
        d = haversine_distance(40.0, -74.0, 41.0, -74.0)
        assert 110.0 < d < 112.0

    def test_haversine_distance_zero(self):
        d = haversine_distance(40.0, -74.0, 40.0, -74.0)
        assert d == pytest.approx(0.0, abs=1e-9)
