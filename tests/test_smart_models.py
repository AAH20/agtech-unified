"""Tests for FIWARE smart data models for agriculture."""
import pytest

from src.iot.smart_models import AgriculturalSmartModels


@pytest.fixture
def models():
    return AgriculturalSmartModels()


class TestCropModel:
    """Crop entity model (NGSI-LD)."""

    def test_crop_entity_has_required_fields(self, models):
        """Crop entity carries id, type, and growth-stage property."""
        crop = models.crop_entity("urn:crop:field1", growth_stage="flowering")
        assert crop["id"] == "urn:crop:field1"
        assert crop["type"] == "Crop"
        assert crop["growthStage"]["value"] == "flowering"

        # Default growth stage is 'planted'.
        assert models.crop_entity("urn:crop:field1")["growthStage"]["value"] == "planted"

    def test_crop_entity_includes_geometry(self, models):
        """Crop entity includes a geo:json location property."""
        crop = models.crop_entity("urn:crop:field1", location=(40.5, -3.7))
        assert crop["location"]["type"] == "GeoProperty"
        assert crop["location"]["value"]["type"] == "Point"
        assert crop["location"]["value"]["coordinates"] == [40.5, -3.7]


class TestSoilModel:
    """Soil entity model (NGSI-LD)."""

    def test_soil_entity_has_required_fields(self, models):
        """Soil entity carries id, type, and moisture property."""
        soil = models.soil_entity("urn:soil:plot1", moisture=0.42)
        assert soil["id"] == "urn:soil:plot1"
        assert soil["type"] == "Soil"
        assert soil["moisture"]["value"] == 0.42

    def test_soil_entity_includes_texture(self, models):
        """Soil entity includes a texture property."""
        soil = models.soil_entity("urn:soil:plot1", texture="loam")
        assert soil["texture"]["value"] == "loam"


class TestWeatherModel:
    """Weather entity model (NGSI-LD)."""

    def test_weather_entity_has_required_fields(self, models):
        """Weather entity carries id, type, and temperature property."""
        weather = models.weather_entity("urn:weather:station1", temperature=22.5)
        assert weather["id"] == "urn:weather:station1"
        assert weather["type"] == "Weather"
        assert weather["temperature"]["value"] == 22.5

    def test_weather_entity_includes_humidity(self, models):
        """Weather entity includes a humidity property."""
        weather = models.weather_entity("urn:weather:station1", humidity=0.65)
        assert weather["humidity"]["value"] == 0.65


class TestDeviceModel:
    """Device entity model (NGSI-LD)."""

    def test_device_entity_has_required_fields(self, models):
        """Device entity carries id, type, and status property."""
        device = models.device_entity("urn:device:sensor1", status="active")
        assert device["id"] == "urn:device:sensor1"
        assert device["type"] == "Device"
        assert device["status"]["value"] == "active"

    def test_device_entity_includes_battery(self, models):
        """Device entity includes a battery-level property."""
        device = models.device_entity("urn:device:sensor1", battery_level=0.87)
        assert device["batteryLevel"]["value"] == 0.87


class TestModelCatalog:
    """Catalog of available smart data models."""

    def test_model_schema_returns_property_names(self, models):
        """model_schema returns the property names for a known model."""
        schema = models.model_schema("Crop")
        assert "growthStage" in schema
        assert "location" in schema

    def test_model_schema_unknown_raises(self, models):
        """model_schema raises for an unknown model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            models.model_schema("Nonexistent")
