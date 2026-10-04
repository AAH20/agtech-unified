"""IoT device management: registration, provisioning, and health monitoring.

Provides DeviceRegistry for tracking agricultural IoT devices (soil sensors,
weather stations, irrigation controllers) through their lifecycle: register,
provision with config, monitor health metrics, and detect offline devices.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DeviceStatus(str, Enum):
    """Lifecycle status of an IoT device."""

    OFFLINE = "offline"
    ONLINE = "online"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class HealthStatus(str, Enum):
    """Health classification derived from device metrics."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class Device:
    """A managed IoT device in the agricultural deployment."""

    device_id: str
    device_type: str
    firmware_version: str = "0.0.0"
    status: DeviceStatus = DeviceStatus.OFFLINE
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    health: Dict[str, float] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    provisioned_at: Optional[float] = None
    last_seen: Optional[float] = None


class DeviceRegistry:
    """Registry for IoT device lifecycle management.

    Tracks devices from registration through provisioning to ongoing health
    monitoring. All operations are in-memory; persistence can be layered on.
    """

    def __init__(self) -> None:
        self._devices: Dict[str, Device] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, device: Device) -> Device:
        """Register a new device. Raises ValueError on duplicate id."""
        if device.device_id in self._devices:
            raise ValueError(f"Device '{device.device_id}' already registered")
        self._devices[device.device_id] = device
        return device

    def unregister(self, device_id: str) -> bool:
        """Remove a device. Returns True if it existed."""
        return self._devices.pop(device_id, None) is not None

    def get(self, device_id: str) -> Optional[Device]:
        """Look up a device by id, or None if not registered."""
        return self._devices.get(device_id)

    def count(self) -> int:
        """Number of registered devices."""
        return len(self._devices)

    def list_devices(
        self,
        status: Optional[DeviceStatus] = None,
        device_type: Optional[str] = None,
    ) -> List[Device]:
        """List devices, optionally filtered by status and/or type."""
        devices = list(self._devices.values())
        if status is not None:
            devices = [d for d in devices if d.status == status]
        if device_type is not None:
            devices = [d for d in devices if d.device_type == device_type]
        return devices

    # ------------------------------------------------------------------
    # Provisioning
    # ------------------------------------------------------------------

    def provision(self, device_id: str, config: Dict[str, Any]) -> Device:
        """Apply configuration to a device and mark it online.

        Raises ValueError if the device is not registered.
        """
        device = self._require(device_id)
        device.config = dict(config)
        device.status = DeviceStatus.ONLINE
        device.provisioned_at = time.time()
        return device

    # ------------------------------------------------------------------
    # Health monitoring
    # ------------------------------------------------------------------

    def update_health(self, device_id: str, metrics: Dict[str, float]) -> Device:
        """Record health metrics (battery, signal, etc.) for a device."""
        device = self._require(device_id)
        device.health.update(metrics)
        device.last_seen = time.time()
        return device

    def heartbeat(self, device_id: str) -> Device:
        """Record a liveness heartbeat, refreshing last_seen."""
        device = self._require(device_id)
        device.last_seen = time.time()
        return device

    def check_health(
        self,
        device_id: str,
        battery_threshold: float = 20.0,
        signal_threshold: float = -90.0,
    ) -> HealthStatus:
        """Classify device health from its latest metrics.

        CRITICAL when battery is low, DEGRADED when signal is weak,
        HEALTHY when both are acceptable, UNKNOWN when no metrics exist.
        """
        device = self._require(device_id)
        if not device.health:
            return HealthStatus.UNKNOWN

        battery = device.health.get("battery")
        signal = device.health.get("signal")

        if battery is not None and battery < battery_threshold:
            return HealthStatus.CRITICAL
        if signal is not None and signal < signal_threshold:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def get_offline_devices(self, threshold_seconds: float = 300.0) -> List[Device]:
        """Return devices silent longer than the threshold.

        Devices that have never reported (last_seen is None) count as offline.
        """
        now = time.time()
        offline = []
        for device in self._devices.values():
            if device.last_seen is None or (now - device.last_seen) > threshold_seconds:
                offline.append(device)
        return offline

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require(self, device_id: str) -> Device:
        device = self._devices.get(device_id)
        if device is None:
            raise ValueError(f"Device '{device_id}' not found")
        return device
