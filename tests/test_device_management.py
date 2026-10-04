"""Test IoT device management: registration, provisioning, health monitoring."""
import time

import pytest

from src.iot.device_management import (
    Device,
    DeviceRegistry,
    DeviceStatus,
    HealthStatus,
)


def make_device(device_id="dev-001", device_type="soil_sensor"):
    return Device(
        device_id=device_id,
        device_type=device_type,
        firmware_version="1.0.0",
        metadata={"field": "north"},
    )


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


def test_register_device():
    """A registered device is stored and retrievable."""
    registry = DeviceRegistry()
    device = make_device()

    registry.register(device)

    assert registry.count() == 1
    assert registry.get("dev-001") is device


def test_register_duplicate_raises():
    """Registering the same device_id twice raises ValueError."""
    registry = DeviceRegistry()
    registry.register(make_device())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(make_device())


def test_unregister_device():
    """Unregistered devices are removed from the registry."""
    registry = DeviceRegistry()
    registry.register(make_device())

    removed = registry.unregister("dev-001")

    assert removed is True
    assert registry.count() == 0
    assert registry.get("dev-001") is None


# ----------------------------------------------------------------------
# Provisioning
# ----------------------------------------------------------------------


def test_provision_device():
    """Provisioning applies config and marks the device online."""
    registry = DeviceRegistry()
    registry.register(make_device())

    registry.provision("dev-001", {"sampling_interval": 30, "threshold": 0.5})

    device = registry.get("dev-001")
    assert device.status == DeviceStatus.ONLINE
    assert device.config == {"sampling_interval": 30, "threshold": 0.5}
    assert device.provisioned_at is not None


def test_provision_unknown_device_raises():
    """Provisioning an unregistered device raises ValueError."""
    registry = DeviceRegistry()

    with pytest.raises(ValueError, match="not found"):
        registry.provision("nope", {})


# ----------------------------------------------------------------------
# Health monitoring
# ----------------------------------------------------------------------


def test_update_health_records_metrics():
    """Health updates store metrics and refresh last_seen."""
    registry = DeviceRegistry()
    registry.register(make_device())
    before = time.time()

    registry.update_health("dev-001", {"battery": 80.0, "signal": -70.0})

    device = registry.get("dev-001")
    assert device.health["battery"] == 80.0
    assert device.health["signal"] == -70.0
    assert device.last_seen >= before


def test_check_health_healthy():
    """A device with good battery and signal is HEALTHY."""
    registry = DeviceRegistry()
    registry.register(make_device())
    registry.update_health("dev-001", {"battery": 90.0, "signal": -60.0})

    assert registry.check_health("dev-001") == HealthStatus.HEALTHY


def test_check_health_critical_low_battery():
    """A device below the battery threshold is CRITICAL."""
    registry = DeviceRegistry()
    registry.register(make_device())
    registry.update_health("dev-001", {"battery": 5.0, "signal": -60.0})

    assert registry.check_health("dev-001") == HealthStatus.CRITICAL


def test_check_health_degraded_weak_signal():
    """A device with weak signal (but good battery) is DEGRADED."""
    registry = DeviceRegistry()
    registry.register(make_device())
    registry.update_health("dev-001", {"battery": 90.0, "signal": -95.0})

    assert registry.check_health("dev-001") == HealthStatus.DEGRADED


def test_offline_devices_detected():
    """Devices silent beyond the threshold are reported offline."""
    registry = DeviceRegistry()
    registry.register(make_device("dev-001"))
    registry.register(make_device("dev-002"))
    registry.heartbeat("dev-001")
    registry.get("dev-002").last_seen = time.time() - 3600.0

    offline = registry.get_offline_devices(threshold_seconds=600)

    assert [d.device_id for d in offline] == ["dev-002"]
