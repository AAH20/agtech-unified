"""FIWARE smart data models for agriculture.

Builds NGSI-LD compliant entity payloads for the core agricultural
model types defined by the FIWARE Smart Data Models programme
(https://smartdatamodels.org): Crop, Soil, Weather, and Device.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class AgriculturalSmartModels:
    """Factory for NGSI-LD agricultural entity payloads.

    Every builder returns a dict ready for POST /entities on an
    NGSI-LD broker: an ``id``, a ``type``, and NGSI-LD properties
    (``{"type": "Property", "value": ...}``) or ``GeoProperty`` for
    locations.
    """

    # ------------------------------------------------------------------
    # Crop
    # ------------------------------------------------------------------

    def crop_entity(
        self,
        entity_id: str,
        growth_stage: str = "planted",
        location: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """Build a Crop entity.

        :param entity_id: NGSI-LD URN, e.g. ``urn:ngsi-ld:Crop:field1``.
        :param growth_stage: phenological stage (planted, flowering, ...).
        :param location: optional ``(latitude, longitude)`` tuple.
        """
        entity: Dict[str, Any] = {
            "id": entity_id,
            "type": "Crop",
            "growthStage": {"type": "Property", "value": growth_stage},
        }
        if location is not None:
            entity["location"] = self._geo_property(location)
        return entity

    # ------------------------------------------------------------------
    # Soil
    # ------------------------------------------------------------------

    def soil_entity(
        self,
        entity_id: str,
        moisture: float = 0.0,
        texture: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a Soil entity.

        :param entity_id: NGSI-LD URN, e.g. ``urn:ngsi-LD:Soil:plot1``.
        :param moisture: volumetric water content (0-1).
        :param texture: soil texture class (e.g. "loam", "clay").
        """
        entity: Dict[str, Any] = {
            "id": entity_id,
            "type": "Soil",
            "moisture": {"type": "Property", "value": moisture},
        }
        if texture is not None:
            entity["texture"] = {"type": "Property", "value": texture}
        return entity

    # ------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------

    def weather_entity(
        self,
        entity_id: str,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Build a Weather entity.

        :param entity_id: NGSI-LD URN, e.g. ``urn:ngsi-LD:Weather:station1``.
        :param temperature: air temperature in degrees Celsius.
        :param humidity: relative humidity (0-1).
        """
        entity: Dict[str, Any] = {
            "id": entity_id,
            "type": "Weather",
        }
        if temperature is not None:
            entity["temperature"] = {"type": "Property", "value": temperature}
        if humidity is not None:
            entity["humidity"] = {"type": "Property", "value": humidity}
        return entity

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    def device_entity(
        self,
        entity_id: str,
        status: str = "active",
        battery_level: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Build a Device entity.

        :param entity_id: NGSI-LD URN, e.g. ``urn:ngsi-LD:Device:sensor1``.
        :param status: device status ("active", "inactive", ...).
        :param battery_level: battery charge (0-1).
        """
        entity: Dict[str, Any] = {
            "id": entity_id,
            "type": "Device",
            "status": {"type": "Property", "value": status},
        }
        if battery_level is not None:
            entity["batteryLevel"] = {"type": "Property", "value": battery_level}
        return entity

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------

    def model_schema(self, model_type: str) -> List[str]:
        """Return the property names defined for a model type.

        :raises ValueError: if the model type is not supported.
        """
        schemas = {
            "Crop": ["growthStage", "location"],
            "Soil": ["moisture", "texture"],
            "Weather": ["temperature", "humidity"],
            "Device": ["status", "batteryLevel"],
        }
        if model_type not in schemas:
            raise ValueError(f"Unknown model type: {model_type}")
        return schemas[model_type]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _geo_property(coords: Tuple[float, float]) -> Dict[str, Any]:
        """Build a GeoProperty value in GeoJSON Point form."""
        return {
            "type": "GeoProperty",
            "value": {
                "type": "Point",
                "coordinates": [coords[0], coords[1]],
            },
        }
