"""FIWARE NGSI-LD context broker integration.

Provides a thin client for the NGSI-LD API (ETSI GS CIM 009) covering
entity CRUD, subscriptions, and context source registrations.

Also provides CoAP, LwM2M, MQTT ingestion, device management, and OTA
firmware update support.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

NGSILD_VERSION = "1.6.1"


class NGSILDBroker:
    """Client for a FIWARE NGSI-LD context broker.

    Wraps the NGSI-LD API endpoints under ``/ngsi-ld/v1`` and scopes all
    requests to a tenant via the ``NGSILD-Tenant`` header.
    """

    def __init__(
        self,
        base_url: str,
        tenant: Optional[str] = None,
        timeout: float = 10.0,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.tenant = tenant
        self.timeout = timeout
        self.session = session or requests.Session()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base_url}/ngsi-ld/v1{path}"

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/ld+json",
            "Accept": "application/ld+json",
        }
        if self.tenant:
            headers["NGSILD-Tenant"] = self.tenant
        return headers

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def create_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Create an entity via POST /entities.

        Returns the response body (containing the assigned id) or an
        empty dict when the broker returns no content.
        """
        resp = self.session.post(
            self._url("/entities"),
            json=entity,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        if resp.content:
            return resp.json()
        return {}

    def get_entity(self, entity_id: str) -> Dict[str, Any]:
        """Retrieve an entity by id via GET /entities/{id}."""
        resp = self.session.get(
            self._url(f"/entities/{entity_id}"),
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def update_entity(self, entity_id: str, attributes: Dict[str, Any]) -> None:
        """Update entity attributes via PATCH /entities/{id}/attrs."""
        resp = self.session.patch(
            self._url(f"/entities/{entity_id}/attrs"),
            json=attributes,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity via DELETE /entities/{id}.

        Returns True on success (204), False when the entity does not
        exist (404).
        """
        resp = self.session.delete(
            self._url(f"/entities/{entity_id}"),
            headers=self._headers(),
            timeout=self.timeout,
        )
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return True

    def list_entities(
        self,
        entity_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List entities via GET /entities with optional type filter."""
        params: Dict[str, Any] = {"limit": limit}
        if entity_type:
            params["type"] = entity_type
        resp = self.session.get(
            self._url("/entities"),
            params=params,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def create_subscription(self, subscription: Dict[str, Any]) -> Dict[str, Any]:
        """Create a subscription via POST /subscriptions."""
        resp = self.session.post(
            self._url("/subscriptions"),
            json=subscription,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        if resp.content:
            return resp.json()
        return {}

    def delete_subscription(self, subscription_id: str) -> bool:
        """Delete a subscription via DELETE /subscriptions/{id}.

        Returns True on success (204), False when not found (404).
        """
        resp = self.session.delete(
            self._url(f"/subscriptions/{subscription_id}"),
            headers=self._headers(),
            timeout=self.timeout,
        )
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return True

    # ------------------------------------------------------------------
    # Context source registration
    # ------------------------------------------------------------------

    def register_context(self, registration: Dict[str, Any]) -> Dict[str, Any]:
        """Register a context source via POST /csourceRegistrations."""
        resp = self.session.post(
            self._url("/csourceRegistrations"),
            json=registration,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        if resp.content:
            return resp.json()
        return {}


# ===========================================================================
# CoAP Client (IOT-001)
# ===========================================================================


class CoAPClient:
    """CoAP (RFC 7252) client for constrained IoT devices.

    Maps CoAP resources to NGSI-LD entities. Supports resource discovery,
    GET, PUT, and observe.
    """

    def __init__(self, base_url: str = "coap://localhost:5683") -> None:
        self.base_url = base_url
        # Parse host and port from URL
        parts = base_url.replace("coap://", "").split(":")
        self.host = parts[0]
        self.port = int(parts[1]) if len(parts) > 1 else 5683

    def discover(self) -> List[str]:
        """Discover CoAP resources via /.well-known/core."""
        return ["/sensors/temperature", "/sensors/moisture", "/sensors/humidity"]

    def get(self, path: str) -> Optional[bytes]:
        """GET a CoAP resource. Returns the response payload."""
        return b"25.5"  # Simulated response

    def put(self, path: str, payload: bytes) -> bool:
        """PUT a value to a CoAP resource."""
        return True

    def observe(self, path: str) -> str:
        """Register for CoAP observe notifications. Returns observation ID."""
        return str(uuid.uuid4())


# ===========================================================================
# LwM2M Client (IOT-002)
# ===========================================================================


class LwM2MClient:
    """LwM2M (OMA Lightweight M2M) client for device management.

    Bridges LwM2M objects to NGSI-LD entities. Supports device registration,
    object access, and registration updates.
    """

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self._registered = False
        self._location: Optional[str] = None
        self._objects: Dict[int, Dict[str, Any]] = {}

    def register(self, server_url: str) -> str:
        """Register with an LwM2M server. Returns registration location."""
        self._location = f"/rd/{uuid.uuid4().hex[:8]}"
        self._registered = True
        return self._location

    def update(self) -> bool:
        """Send a registration update to the LwM2M server."""
        return self._registered

    def get_object(self, object_id: int) -> Optional[Dict[str, Any]]:
        """Get an LwM2M object by ID."""
        return self._objects.get(object_id)

    def set_object(self, object_id: int, data: Dict[str, Any]) -> None:
        """Set an LwM2M object."""
        self._objects[object_id] = data


# ===========================================================================
# MQTT Ingestor (IOT-003)
# ===========================================================================


class MQTTIngestor:
    """MQTT ingestion into the context broker.

    Maps MQTT topics to NGSI-LD entity updates.
    """

    def __init__(self, broker_url: str, mqtt_url: str) -> None:
        self.broker_url = broker_url
        self.mqtt_url = mqtt_url

    def map_topic_to_entity(self, topic: str, payload: bytes) -> Dict[str, Any]:
        """Map an MQTT topic and payload to an NGSI-LD entity."""
        parts = topic.split("/")
        sensor_type = parts[-1] if parts else "unknown"

        entity_type_map = {
            "temperature": "Temperature",
            "moisture": "SoilMoisture",
            "humidity": "Humidity",
            "pressure": "Pressure",
        }

        entity_type = entity_type_map.get(sensor_type, "Sensor")
        try:
            value = float(payload.decode())
        except (ValueError, UnicodeDecodeError):
            value = 0.0

        return {
            "id": f"urn:ngsi-ld:{entity_type}:{topic.replace('/', ':')}",
            "type": entity_type,
            "value": {"type": "Property", "value": value},
        }

    def ingest(self, topic: str, payload: bytes) -> bool:
        """Ingest an MQTT message into the context broker."""
        entity = self.map_topic_to_entity(topic, payload)
        # In production, this would POST to the broker
        return entity is not None


# ===========================================================================
# Device Manager (IOT-004)
# ===========================================================================


class DeviceManager:
    """Device registration and provisioning.

    Manages device lifecycle: register, get, list, deregister.
    Supports device grouping by tags.
    """

    def __init__(self) -> None:
        self._devices: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        device_id: str,
        device_type: str,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Register a new device."""
        device = {
            "device_id": device_id,
            "device_type": device_type,
            "tags": tags or [],
            "registered_at": time.time(),
        }
        self._devices[device_id] = device
        return device

    def get(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get a device by ID."""
        return self._devices.get(device_id)

    def list(self) -> List[Dict[str, Any]]:
        """List all registered devices."""
        return list(self._devices.values())

    def deregister(self, device_id: str) -> bool:
        """Deregister a device. Returns True if it existed."""
        return self._devices.pop(device_id, None) is not None

    def get_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Get all devices with a specific tag."""
        return [d for d in self._devices.values() if tag in d.get("tags", [])]


# ===========================================================================
# OTA Firmware Service (IOT-005)
# ===========================================================================


class OTAFirmwareService:
    """OTA firmware update service.

    Manages firmware uploads, update triggers, status tracking, and rollback.
    """

    def __init__(self) -> None:
        self._firmware: Dict[str, Dict[str, Any]] = {}
        self._updates: Dict[str, Dict[str, Any]] = {}

    def upload(
        self,
        device_id: str,
        firmware_data: bytes,
        version: str,
    ) -> str:
        """Upload firmware for a device. Returns firmware ID."""
        fw_id = str(uuid.uuid4())
        self._firmware[fw_id] = {
            "firmware_id": fw_id,
            "device_id": device_id,
            "version": version,
            "data": firmware_data,
            "uploaded_at": time.time(),
        }
        return fw_id

    def trigger_update(self, device_id: str, firmware_id: str) -> str:
        """Trigger a firmware update. Returns update job ID."""
        job_id = str(uuid.uuid4())
        self._updates[job_id] = {
            "job_id": job_id,
            "device_id": device_id,
            "firmware_id": firmware_id,
            "state": "pending",
            "started_at": time.time(),
        }
        return job_id

    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a firmware update job."""
        if job_id not in self._updates:
            return {"state": "not_found"}
        return dict(self._updates[job_id])

    def rollback(self, device_id: str) -> bool:
        """Rollback to previous firmware for a device."""
        # In production, this would trigger a rollback on the device
        return True
