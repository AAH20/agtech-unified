"""IoT device management: registration, provisioning, and health monitoring.

Provides DeviceRegistry for tracking agricultural IoT devices (soil sensors,
weather stations, irrigation controllers) through their lifecycle: register,
provision with config, monitor health metrics, and detect offline devices.

Also provides DeviceAuth for authentication, CommandQueue for downlink,
DeviceShadow for digital twin sync, ConfigVersioning for config history,
and HealthAlert for alerting.
"""

from __future__ import annotations

import hashlib
import secrets
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


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


# ===========================================================================
# Device Authentication (IOT-043)
# ===========================================================================


class DeviceAuth:
    """Device authentication and authorization.

    Manages API keys, tokens, and per-device ACLs for authenticating
    IoT devices. Uses SHA-256 hashing for key storage.
    """

    def __init__(self) -> None:
        self._api_keys: Dict[str, str] = {}  # device_id -> hashed key
        self._tokens: Dict[str, Tuple[str, float]] = {}  # device_id -> (token, expiry)
        self._acls: Dict[str, Set[str]] = {}  # device_id -> set of permissions

    def generate_api_key(self, device_id: str) -> str:
        """Generate a new API key for a device. Returns the plaintext key."""
        key = secrets.token_urlsafe(32)
        self._api_keys[device_id] = self._hash(key)
        return key

    def validate_api_key(self, device_id: str, key: str) -> bool:
        """Validate an API key for a device."""
        if device_id not in self._api_keys:
            return False
        return self._api_keys[device_id] == self._hash(key)

    def revoke_api_key(self, device_id: str) -> None:
        """Revoke a device's API key."""
        self._api_keys.pop(device_id, None)

    def generate_token(self, device_id: str, ttl: int = 3600) -> str:
        """Generate a time-limited token for a device."""
        token = secrets.token_urlsafe(32)
        expiry = time.time() + ttl
        self._tokens[device_id] = (token, expiry)
        return token

    def validate_token(self, device_id: str, token: str) -> bool:
        """Validate a token for a device. Returns False if expired."""
        if device_id not in self._tokens:
            return False
        stored_token, expiry = self._tokens[device_id]
        if time.time() > expiry:
            return False
        return stored_token == token

    def set_acl(self, device_id: str, permissions: List[str]) -> None:
        """Set permissions for a device."""
        self._acls[device_id] = set(permissions)

    def has_permission(self, device_id: str, permission: str) -> bool:
        """Check if a device has a specific permission."""
        if device_id not in self._acls:
            return False
        return permission in self._acls[device_id]

    @staticmethod
    def _hash(key: str) -> str:
        """Hash an API key using SHA-256."""
        return hashlib.sha256(key.encode()).hexdigest()


# ===========================================================================
# Command Downlink (IOT-044)
# ===========================================================================


class CommandQueue:
    """Command dispatch queue for sending commands to devices.

    Supports command queuing, status tracking, and timeout handling.
    """

    def __init__(self) -> None:
        self._commands: Dict[str, Dict[str, Any]] = {}

    def send(
        self,
        device_id: str,
        command: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 300.0,
    ) -> str:
        """Send a command to a device. Returns the command ID."""
        cmd_id = str(uuid.uuid4())
        self._commands[cmd_id] = {
            "command_id": cmd_id,
            "device_id": device_id,
            "command": command,
            "params": params or {},
            "status": "pending",
            "created_at": time.time(),
            "timeout": timeout,
            "result": None,
        }
        return cmd_id

    def get_pending(self, device_id: str) -> List[Dict[str, Any]]:
        """Get all pending commands for a device."""
        return [
            cmd
            for cmd in self._commands.values()
            if cmd["device_id"] == device_id and cmd["status"] == "pending"
        ]

    def acknowledge(self, command_id: str, result: str = "success") -> bool:
        """Acknowledge a command as completed."""
        if command_id not in self._commands:
            return False
        self._commands[command_id]["status"] = "completed"
        self._commands[command_id]["result"] = result
        return True

    def get_status(self, command_id: str) -> Optional[str]:
        """Get the status of a command."""
        if command_id not in self._commands:
            return None
        cmd = self._commands[command_id]
        if cmd["status"] == "pending":
            elapsed = time.time() - cmd["created_at"]
            if elapsed > cmd["timeout"]:
                cmd["status"] = "timed_out"
        return cmd["status"]


# ===========================================================================
# Device Shadow (IOT-045)
# ===========================================================================


class DeviceShadow:
    """Device shadow (digital twin) for async state management.

    Maintains desired and reported state separately, with version tracking
    and delta computation.
    """

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id
        self._desired: Dict[str, Any] = {}
        self._reported: Dict[str, Any] = {}
        self._version: int = 0

    def set_desired(self, state: Dict[str, Any]) -> None:
        """Set the desired state."""
        self._desired.update(state)
        self._version += 1

    def set_reported(self, state: Dict[str, Any]) -> None:
        """Set the reported state."""
        self._reported.update(state)
        self._version += 1

    def get(self) -> Dict[str, Any]:
        """Get the full shadow state."""
        return {
            "device_id": self.device_id,
            "desired": dict(self._desired),
            "reported": dict(self._reported),
            "version": self._version,
        }

    def get_delta(self) -> Dict[str, Any]:
        """Get the delta between desired and reported state."""
        delta = {}
        for key, desired_value in self._desired.items():
            if key not in self._reported or self._reported[key] != desired_value:
                delta[key] = desired_value
        return delta

    def get_version(self) -> int:
        """Get the current shadow version."""
        return self._version


# ===========================================================================
# Config Versioning (IOT-006)
# ===========================================================================


class ConfigVersioning:
    """Device configuration versioning and rollback.

    Stores configuration history with rollback and diff support.
    """

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id
        self._history: List[Dict[str, Any]] = []

    def save(self, config: Dict[str, Any]) -> int:
        """Save a configuration version. Returns the version number."""
        version = len(self._history)
        self._history.append(
            {
                "version": version,
                "config": dict(config),
                "saved_at": time.time(),
            }
        )
        return version

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the full configuration history."""
        return list(self._history)

    def rollback(self) -> Optional[Dict[str, Any]]:
        """Rollback to the previous configuration. Returns the restored config."""
        if len(self._history) < 2:
            return None
        self._history.pop()  # Remove current
        return dict(self._history[-1]["config"])  # Return previous

    def diff(self, version_a: int, version_b: int) -> Dict[str, Any]:
        """Compute the diff between two configuration versions."""
        if version_a >= len(self._history) or version_b >= len(self._history):
            raise ValueError("Invalid version number")
        config_a = self._history[version_a]["config"]
        config_b = self._history[version_b]["config"]
        changes = {}
        all_keys = set(config_a.keys()) | set(config_b.keys())
        for key in all_keys:
            if config_a.get(key) != config_b.get(key):
                changes[key] = {
                    "old": config_a.get(key),
                    "new": config_b.get(key),
                }
        return changes


# ===========================================================================
# Health Alert (IOT-007)
# ===========================================================================


class HealthAlert:
    """Device health monitoring and alerting.

    Generates alerts on health status changes with deduplication.
    """

    def __init__(self, cooldown_seconds: float = 300.0) -> None:
        self._last_alert: Dict[str, Tuple[HealthStatus, float]] = {}
        self._cooldown = cooldown_seconds

    def check_and_alert(self, device_id: str, status: HealthStatus) -> bool:
        """Check health and generate an alert if needed.

        Returns True if an alert was generated, False otherwise.
        """
        now = time.time()

        # Check if we already alerted for this status recently
        if device_id in self._last_alert:
            last_status, last_time = self._last_alert[device_id]
            if last_status == status and (now - last_time) < self._cooldown:
                return False

        # Generate alert for critical/degraded, or recovery
        if status in (HealthStatus.CRITICAL, HealthStatus.DEGRADED):
            self._last_alert[device_id] = (status, now)
            return True
        elif status == HealthStatus.HEALTHY and device_id in self._last_alert:
            last_status, _ = self._last_alert[device_id]
            if last_status in (HealthStatus.CRITICAL, HealthStatus.DEGRADED):
                self._last_alert[device_id] = (status, now)
                return True  # Recovery alert

        return False
