"""Tests for IoT gap fixes: device auth, command downlink, device shadow,
cross-field validation, temporal validation, quality scoring, unit conversion.

TDD: these tests define the expected behavior for the new features.
"""

import time

import pytest

from src.iot.context_broker import (
    CoAPClient,
    DeviceManager,
    LwM2MClient,
    MQTTIngestor,
    OTAFirmwareService,
)
from src.iot.data_pipeline import (
    BackpressureHandler,
    DataQualityMetrics,
    DataValidator,
    RetentionPolicy,
)
from src.iot.data_validation import (
    CrossFieldValidator,
    QualityScorer,
    TemporalValidator,
    UnitConverter,
)
from src.iot.device_management import (
    CommandQueue,
    ConfigVersioning,
    DeviceAuth,
    DeviceShadow,
    HealthAlert,
    HealthStatus,
)
from src.iot.smart_models import ModelValidator

# ===========================================================================
# Device Authentication (IOT-043)
# ===========================================================================


class TestDeviceAuth:
    """Device authentication and authorization."""

    def test_generate_api_key(self):
        """Generating an API key returns a non-empty string."""
        auth = DeviceAuth()
        key = auth.generate_api_key("dev-001")
        assert isinstance(key, str)
        assert len(key) > 0

    def test_validate_correct_api_key(self):
        """A device with the correct API key passes validation."""
        auth = DeviceAuth()
        key = auth.generate_api_key("dev-001")
        assert auth.validate_api_key("dev-001", key) is True

    def test_validate_wrong_api_key(self):
        """A device with the wrong API key fails validation."""
        auth = DeviceAuth()
        auth.generate_api_key("dev-001")
        assert auth.validate_api_key("dev-001", "wrong-key") is False

    def test_validate_unknown_device(self):
        """Validating an unknown device returns False."""
        auth = DeviceAuth()
        assert auth.validate_api_key("unknown", "any-key") is False

    def test_revoke_api_key(self):
        """Revoking an API key prevents further validation."""
        auth = DeviceAuth()
        key = auth.generate_api_key("dev-001")
        assert auth.validate_api_key("dev-001", key) is True
        auth.revoke_api_key("dev-001")
        assert auth.validate_api_key("dev-001", key) is False

    def test_generate_token(self):
        """Generating a token returns a non-empty string."""
        auth = DeviceAuth()
        token = auth.generate_token("dev-001", ttl=3600)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_validate_token(self):
        """A valid token passes validation."""
        auth = DeviceAuth()
        token = auth.generate_token("dev-001", ttl=3600)
        assert auth.validate_token("dev-001", token) is True

    def test_validate_expired_token(self):
        """An expired token fails validation."""
        auth = DeviceAuth()
        token = auth.generate_token("dev-001", ttl=-1)  # Already expired
        assert auth.validate_token("dev-001", token) is False

    def test_per_device_acl(self):
        """Per-device ACLs restrict operations."""
        auth = DeviceAuth()
        auth.set_acl("dev-001", ["read", "write"])
        assert auth.has_permission("dev-001", "read") is True
        assert auth.has_permission("dev-001", "write") is True
        assert auth.has_permission("dev-001", "delete") is False

    def test_acl_default_deny(self):
        """Devices with no ACL default to deny all."""
        auth = DeviceAuth()
        assert auth.has_permission("unknown", "read") is False


# ===========================================================================
# Command Downlink (IOT-044)
# ===========================================================================


class TestCommandQueue:
    """Command dispatch to devices."""

    def test_send_command(self):
        """Sending a command adds it to the queue."""
        queue = CommandQueue()
        cmd_id = queue.send("dev-001", "open_valve", {"duration": 300})
        assert cmd_id is not None
        assert len(cmd_id) > 0

    def test_get_pending_commands(self):
        """Pending commands are returned for a device."""
        queue = CommandQueue()
        queue.send("dev-001", "open_valve", {"duration": 300})
        queue.send("dev-001", "close_valve", {})
        queue.send("dev-002", "start_sampling", {})

        pending = queue.get_pending("dev-001")
        assert len(pending) == 2
        assert all(c["device_id"] == "dev-001" for c in pending)

    def test_acknowledge_command(self):
        """Acknowledging a command marks it as completed."""
        queue = CommandQueue()
        cmd_id = queue.send("dev-001", "open_valve", {})
        queue.acknowledge(cmd_id, "success")

        pending = queue.get_pending("dev-001")
        assert len(pending) == 0

    def test_command_timeout(self):
        """Commands past their timeout are marked as timed out."""
        queue = CommandQueue()
        cmd_id = queue.send("dev-001", "open_valve", {}, timeout=0.01)
        time.sleep(0.02)

        status = queue.get_status(cmd_id)
        assert status == "timed_out"

    def test_command_status_tracking(self):
        """Command status is tracked through its lifecycle."""
        queue = CommandQueue()
        cmd_id = queue.send("dev-001", "open_valve", {})
        assert queue.get_status(cmd_id) == "pending"

        queue.acknowledge(cmd_id, "success")
        assert queue.get_status(cmd_id) == "completed"


# ===========================================================================
# Device Shadow (IOT-045)
# ===========================================================================


class TestDeviceShadow:
    """Device shadow (digital twin) sync."""

    def test_set_desired_state(self):
        """Setting desired state stores it in the shadow."""
        shadow = DeviceShadow("dev-001")
        shadow.set_desired({"sampling_interval": 60})

        state = shadow.get()
        assert state["desired"]["sampling_interval"] == 60

    def test_set_reported_state(self):
        """Setting reported state stores it in the shadow."""
        shadow = DeviceShadow("dev-001")
        shadow.set_reported({"sampling_interval": 30, "battery": 85.0})

        state = shadow.get()
        assert state["reported"]["sampling_interval"] == 30
        assert state["reported"]["battery"] == 85.0

    def test_delta_computation(self):
        """Delta shows differences between desired and reported."""
        shadow = DeviceShadow("dev-001")
        shadow.set_desired({"sampling_interval": 60, "threshold": 0.5})
        shadow.set_reported({"sampling_interval": 30, "threshold": 0.5})

        delta = shadow.get_delta()
        assert "sampling_interval" in delta
        assert "threshold" not in delta  # Same value, no delta

    def test_version_tracking(self):
        """Shadow version increments on each update."""
        shadow = DeviceShadow("dev-001")
        v1 = shadow.get_version()
        shadow.set_desired({"sampling_interval": 60})
        v2 = shadow.get_version()
        assert v2 > v1

    def test_conflict_resolution(self):
        """Conflict resolution merges desired and reported."""
        shadow = DeviceShadow("dev-001")
        shadow.set_desired({"sampling_interval": 60})
        shadow.set_reported({"sampling_interval": 60})

        # When desired matches reported, no delta
        delta = shadow.get_delta()
        assert delta == {}


# ===========================================================================
# Config Versioning (IOT-006)
# ===========================================================================


class TestConfigVersioning:
    """Device configuration versioning and rollback."""

    def test_save_config_version(self):
        """Saving a config creates a versioned entry."""
        versioning = ConfigVersioning("dev-001")
        versioning.save({"sampling_interval": 30})
        versioning.save({"sampling_interval": 60})

        history = versioning.get_history()
        assert len(history) == 2
        assert history[0]["config"]["sampling_interval"] == 30
        assert history[1]["config"]["sampling_interval"] == 60

    def test_rollback(self):
        """Rolling back restores a previous config."""
        versioning = ConfigVersioning("dev-001")
        versioning.save({"sampling_interval": 30})
        versioning.save({"sampling_interval": 60})

        old_config = versioning.rollback()
        assert old_config["sampling_interval"] == 30

    def test_config_diff(self):
        """Config diff shows changes between versions."""
        versioning = ConfigVersioning("dev-001")
        versioning.save({"sampling_interval": 30, "threshold": 0.5})
        versioning.save({"sampling_interval": 60, "threshold": 0.5})

        diff = versioning.diff(0, 1)
        assert "sampling_interval" in diff
        assert "threshold" not in diff


# ===========================================================================
# Health Alert (IOT-007)
# ===========================================================================


class TestHealthAlert:
    """Device health monitoring and alerting."""

    def test_alert_on_critical_health(self):
        """Critical health triggers an alert."""
        alert = HealthAlert()
        triggered = alert.check_and_alert("dev-001", HealthStatus.CRITICAL)
        assert triggered is True

    def test_no_alert_on_healthy(self):
        """Healthy status does not trigger an alert."""
        alert = HealthAlert()
        triggered = alert.check_and_alert("dev-001", HealthStatus.HEALTHY)
        assert triggered is False

    def test_alert_deduplication(self):
        """Repeated critical alerts are deduplicated."""
        alert = HealthAlert()
        alert.check_and_alert("dev-001", HealthStatus.CRITICAL)
        # Second alert within cooldown period
        triggered = alert.check_and_alert("dev-001", HealthStatus.CRITICAL)
        assert triggered is False

    def test_alert_recovery(self):
        """Recovery from critical triggers a recovery alert."""
        alert = HealthAlert()
        alert.check_and_alert("dev-001", HealthStatus.CRITICAL)
        recovered = alert.check_and_alert("dev-001", HealthStatus.HEALTHY)
        assert recovered is True


# ===========================================================================
# Cross-Field Validation (IOT-052)
# ===========================================================================


class TestCrossFieldValidator:
    """Cross-field validation for conditional rules."""

    def test_conditional_rule_pass(self):
        """A conditional rule passes when condition is met and constraint holds."""
        validator = CrossFieldValidator()
        validator.add_rule(
            condition={"field": "irrigation", "equals": "on"},
            constraint={"field": "flow_rate", "min": 0.1},
        )
        result = validator.validate({"irrigation": "on", "flow_rate": 5.0})
        assert result["valid"] is True

    def test_conditional_rule_fail(self):
        """A conditional rule fails when condition is met but constraint violated."""
        validator = CrossFieldValidator()
        validator.add_rule(
            condition={"field": "irrigation", "equals": "on"},
            constraint={"field": "flow_rate", "min": 0.1},
        )
        result = validator.validate({"irrigation": "on", "flow_rate": 0.0})
        assert result["valid"] is False

    def test_conditional_rule_skip(self):
        """A conditional rule is skipped when condition is not met."""
        validator = CrossFieldValidator()
        validator.add_rule(
            condition={"field": "irrigation", "equals": "on"},
            constraint={"field": "flow_rate", "min": 0.1},
        )
        result = validator.validate({"irrigation": "off", "flow_rate": 0.0})
        assert result["valid"] is True

    def test_field_dependency(self):
        """Field dependency requires both fields to be present."""
        validator = CrossFieldValidator()
        validator.add_dependency("start_time", "end_time")
        result = validator.validate({"start_time": 100})
        assert result["valid"] is False

    def test_multiple_rules(self):
        """Multiple rules are all evaluated."""
        validator = CrossFieldValidator()
        validator.add_rule(
            condition={"field": "mode", "equals": "active"},
            constraint={"field": "sampling_rate", "min": 1},
        )
        validator.add_rule(
            condition={"field": "mode", "equals": "active"},
            constraint={"field": "sampling_rate", "max": 100},
        )
        result = validator.validate({"mode": "active", "sampling_rate": 50})
        assert result["valid"] is True


# ===========================================================================
# Temporal Validation (IOT-053)
# ===========================================================================


class TestTemporalValidator:
    """Temporal validation for timestamps."""

    def test_reject_future_timestamp(self):
        """A timestamp far in the future is rejected."""
        validator = TemporalValidator(max_future_seconds=60)
        future_time = time.time() + 3600
        result = validator.validate_timestamp(future_time)
        assert result["valid"] is False

    def test_accept_current_timestamp(self):
        """A current timestamp is accepted."""
        validator = TemporalValidator(max_future_seconds=60)
        result = validator.validate_timestamp(time.time())
        assert result["valid"] is True

    def test_flag_stale_data(self):
        """A very old timestamp is flagged as stale."""
        validator = TemporalValidator(stale_threshold_seconds=300)
        old_time = time.time() - 3600
        result = validator.validate_timestamp(old_time)
        assert result["stale"] is True

    def test_monotonic_ordering(self):
        """Out-of-order timestamps are detected."""
        validator = TemporalValidator()
        result = validator.validate_sequence([100.0, 200.0, 150.0])
        assert result["valid"] is False

    def test_monotonic_ordering_pass(self):
        """In-order timestamps pass."""
        validator = TemporalValidator()
        result = validator.validate_sequence([100.0, 200.0, 300.0])
        assert result["valid"] is True


# ===========================================================================
# Unit Conversion (IOT-054)
# ===========================================================================


class TestUnitConverter:
    """Unit conversion and validation."""

    def test_celsius_to_fahrenheit(self):
        """Celsius to Fahrenheit conversion is correct."""
        converter = UnitConverter()
        result = converter.convert(0.0, "celsius", "fahrenheit")
        assert abs(result - 32.0) < 0.01

    def test_fahrenheit_to_celsius(self):
        """Fahrenheit to Celsius conversion is correct."""
        converter = UnitConverter()
        result = converter.convert(32.0, "fahrenheit", "celsius")
        assert abs(result - 0.0) < 0.01

    def test_same_unit(self):
        """Converting to the same unit returns the original value."""
        converter = UnitConverter()
        result = converter.convert(25.0, "celsius", "celsius")
        assert result == 25.0

    def test_invalid_unit(self):
        """An invalid unit raises ValueError."""
        converter = UnitConverter()
        with pytest.raises(ValueError):
            converter.convert(25.0, "celsius", "invalid_unit")

    def test_unit_consistency_check(self):
        """Unit consistency check validates compatible units."""
        converter = UnitConverter()
        assert converter.is_compatible("celsius", "fahrenheit") is True
        assert converter.is_compatible("celsius", "kelvin") is True
        assert converter.is_compatible("celsius", "percent") is False

    def test_enforce_si_units(self):
        """Enforcing SI units converts non-SI values."""
        converter = UnitConverter()
        result = converter.to_si(32.0, "fahrenheit", "temperature")
        assert abs(result - 273.15) < 1.0  # ~0°C in Kelvin


# ===========================================================================
# Quality Scoring (IOT-055)
# ===========================================================================


class TestQualityScorer:
    """Data quality scoring."""

    def test_completeness_score(self):
        """Completeness score reflects missing fields."""
        scorer = QualityScorer()
        required = ["sensor_id", "value", "timestamp"]
        score = scorer.completeness({"sensor_id": "s1", "value": 25.0}, required)
        assert score < 1.0  # Missing timestamp

    def test_completeness_full(self):
        """All required fields present gives full completeness."""
        scorer = QualityScorer()
        required = ["sensor_id", "value", "timestamp"]
        score = scorer.completeness(
            {"sensor_id": "s1", "value": 25.0, "timestamp": 100.0}, required
        )
        assert score == 1.0

    def test_freshness_score(self):
        """Freshness score decreases with age."""
        scorer = QualityScorer()
        now = time.time()
        fresh = scorer.freshness(now, max_age=300)
        stale = scorer.freshness(now - 600, max_age=300)
        assert fresh > stale

    def test_overall_quality_score(self):
        """Overall quality combines multiple factors."""
        scorer = QualityScorer()
        score = scorer.overall(
            completeness=1.0,
            freshness=0.8,
            anomaly_rate=0.0,
        )
        assert 0.0 <= score <= 1.0

    def test_quality_with_anomalies(self):
        """Anomaly rate reduces quality score."""
        scorer = QualityScorer()
        clean = scorer.overall(completeness=1.0, freshness=1.0, anomaly_rate=0.0)
        dirty = scorer.overall(completeness=1.0, freshness=1.0, anomaly_rate=0.5)
        assert clean > dirty


# ===========================================================================
# Data Validator for Pipeline (IOT-008)
# ===========================================================================


class TestDataValidator:
    """Sensor data validation in the pipeline."""

    def test_valid_reading_passes(self):
        """A valid sensor reading passes validation."""
        validator = DataValidator()
        reading = {
            "sensor_id": "s1",
            "value": 25.0,
            "unit": "celsius",
            "timestamp": time.time(),
        }
        result = validator.validate(reading)
        assert result["valid"] is True

    def test_invalid_reading_fails(self):
        """An invalid sensor reading fails validation."""
        validator = DataValidator()
        reading = {
            "sensor_id": "s1",
            "value": 9999.0,  # Out of range
            "unit": "celsius",
            "timestamp": time.time(),
        }
        result = validator.validate(reading)
        assert result["valid"] is False

    def test_missing_required_field(self):
        """Missing required field fails validation."""
        validator = DataValidator()
        reading = {"value": 25.0, "unit": "celsius"}
        result = validator.validate(reading)
        assert result["valid"] is False


# ===========================================================================
# Data Quality Metrics (IOT-009)
# ===========================================================================


class TestDataQualityMetrics:
    """Data quality metrics tracking."""

    def test_record_valid_reading(self):
        """Recording a valid reading updates metrics."""
        metrics = DataQualityMetrics()
        metrics.record("s1", valid=True)
        stats = metrics.get_stats("s1")
        assert stats["total"] == 1
        assert stats["valid"] == 1

    def test_record_invalid_reading(self):
        """Recording an invalid reading updates metrics."""
        metrics = DataQualityMetrics()
        metrics.record("s1", valid=False)
        stats = metrics.get_stats("s1")
        assert stats["total"] == 1
        assert stats["invalid"] == 1

    def test_quality_percentage(self):
        """Quality percentage is computed correctly."""
        metrics = DataQualityMetrics()
        metrics.record("s1", valid=True)
        metrics.record("s1", valid=True)
        metrics.record("s1", valid=False)
        stats = metrics.get_stats("s1")
        assert abs(stats["quality_pct"] - 66.67) < 0.1

    def test_latency_tracking(self):
        """Latency is tracked per sensor."""
        metrics = DataQualityMetrics()
        metrics.record("s1", valid=True, latency_ms=50.0)
        metrics.record("s1", valid=True, latency_ms=100.0)
        stats = metrics.get_stats("s1")
        assert stats["avg_latency_ms"] == 75.0


# ===========================================================================
# Backpressure Handler (IOT-012)
# ===========================================================================


class TestBackpressureHandler:
    """Backpressure handling for slow consumers."""

    def test_under_limit(self):
        """Queue under limit accepts messages."""
        handler = BackpressureHandler(max_size=10)
        assert handler.accept("msg1") is True

    def test_over_limit(self):
        """Queue over limit rejects messages."""
        handler = BackpressureHandler(max_size=2)
        handler.accept("msg1")
        handler.accept("msg2")
        assert handler.accept("msg3") is False

    def test_drain_clears_backpressure(self):
        """Draining messages clears backpressure."""
        handler = BackpressureHandler(max_size=2)
        handler.accept("msg1")
        handler.accept("msg2")
        assert handler.accept("msg3") is False
        handler.drain()
        assert handler.accept("msg3") is True

    def test_queue_depth(self):
        """Queue depth is tracked correctly."""
        handler = BackpressureHandler(max_size=10)
        handler.accept("msg1")
        handler.accept("msg2")
        assert handler.get_depth() == 2


# ===========================================================================
# Retention Policy (IOT-013)
# ===========================================================================


class TestRetentionPolicy:
    """Data retention policy enforcement."""

    def test_recent_data_kept(self):
        """Recent data is kept."""
        policy = RetentionPolicy(retention_days=30)
        now = time.time()
        assert policy.should_keep(now) is True

    def test_old_data_deleted(self):
        """Old data is deleted."""
        policy = RetentionPolicy(retention_days=30)
        old_time = time.time() - (31 * 86400)
        assert policy.should_keep(old_time) is False

    def test_tiered_storage(self):
        """Data is tiered by age."""
        policy = RetentionPolicy(retention_days=90)
        now = time.time()
        assert policy.get_tier(now) == "hot"
        assert policy.get_tier(now - (10 * 86400)) == "warm"
        assert policy.get_tier(now - (60 * 86400)) == "cold"


# ===========================================================================
# CoAP Client (IOT-001)
# ===========================================================================


class TestCoAPClient:
    """CoAP protocol support."""

    def test_coap_client_init(self):
        """CoAP client initializes with host and port."""
        client = CoAPClient("coap://localhost:5683")
        assert client.host == "localhost"
        assert client.port == 5683

    def test_coap_resource_discovery(self):
        """CoAP resource discovery returns resources."""
        client = CoAPClient("coap://localhost:5683")
        resources = client.discover()
        assert isinstance(resources, list)

    def test_coap_get(self):
        """CoAP GET retrieves a resource."""
        client = CoAPClient("coap://localhost:5683")
        result = client.get("/sensors/temperature")
        assert result is not None

    def test_coap_observe(self):
        """CoAP observe registers for notifications."""
        client = CoAPClient("coap://localhost:5683")
        observed = client.observe("/sensors/temperature")
        assert observed is not None


# ===========================================================================
# LwM2M Client (IOT-002)
# ===========================================================================


class TestLwM2MClient:
    """LwM2M protocol support."""

    def test_lwm2m_client_init(self):
        """LwM2M client initializes with endpoint."""
        client = LwM2MClient("agtech-device-001")
        assert client.endpoint == "agtech-device-001"

    def test_lwm2m_register(self):
        """LwM2M registration returns a location."""
        client = LwM2MClient("agtech-device-001")
        location = client.register("localhost:5684")
        assert location is not None

    def test_lwm2m_device_object(self):
        """LwM2M Device object (Object 3) is accessible."""
        client = LwM2MClient("agtech-device-001")
        client.set_object(3, {"manufacturer": "AgTech", "model": "SoilSensor-v2"})
        device_obj = client.get_object(3)
        assert device_obj is not None
        assert device_obj["manufacturer"] == "AgTech"

    def test_lwm2m_update(self):
        """LwM2M update refreshes registration."""
        client = LwM2MClient("agtech-device-001")
        client.register("localhost:5684")
        result = client.update()
        assert result is True


# ===========================================================================
# MQTT Ingestor (IOT-003)
# ===========================================================================


class TestMQTTIngestor:
    """MQTT ingestion into context broker."""

    def test_ingestor_init(self):
        """MQTT ingestor initializes with broker URL."""
        ingestor = MQTTIngestor("http://broker:1026", "mqtt://localhost:1883")
        assert ingestor.broker_url == "http://broker:1026"

    def test_topic_to_entity_mapping(self):
        """MQTT topic maps to NGSI-LD entity."""
        ingestor = MQTTIngestor("http://broker:1026", "mqtt://localhost:1883")
        entity = ingestor.map_topic_to_entity("sensors/temperature", b"25.5")
        assert entity["type"] == "Temperature"
        assert "value" in entity

    def test_ingest_message(self):
        """Ingesting an MQTT message creates an entity."""
        ingestor = MQTTIngestor("http://broker:1026", "mqtt://localhost:1883")
        result = ingestor.ingest("sensors/temperature", b"25.5")
        assert result is True


# ===========================================================================
# Device Manager (IOT-004)
# ===========================================================================


class TestDeviceManager:
    """Device registration and provisioning."""

    def test_register_device(self):
        """Registering a device adds it to the manager."""
        manager = DeviceManager()
        device = manager.register("dev-001", "soil_sensor")
        assert device["device_id"] == "dev-001"
        assert device["device_type"] == "soil_sensor"

    def test_get_device(self):
        """Getting a device returns its info."""
        manager = DeviceManager()
        manager.register("dev-001", "soil_sensor")
        device = manager.get("dev-001")
        assert device is not None
        assert device["device_id"] == "dev-001"

    def test_list_devices(self):
        """Listing devices returns all registered devices."""
        manager = DeviceManager()
        manager.register("dev-001", "soil_sensor")
        manager.register("dev-002", "weather_station")
        devices = manager.list()
        assert len(devices) == 2

    def test_deregister_device(self):
        """Deregistering removes a device."""
        manager = DeviceManager()
        manager.register("dev-001", "soil_sensor")
        assert manager.deregister("dev-001") is True
        assert manager.get("dev-001") is None

    def test_device_groups(self):
        """Devices can be grouped by tags."""
        manager = DeviceManager()
        manager.register("dev-001", "soil_sensor", tags=["field-a"])
        manager.register("dev-002", "soil_sensor", tags=["field-b"])
        group = manager.get_by_tag("field-a")
        assert len(group) == 1


# ===========================================================================
# OTA Firmware Service (IOT-005)
# ===========================================================================


class TestOTAFirmwareService:
    """OTA firmware update support."""

    def test_upload_firmware(self):
        """Uploading firmware stores it."""
        service = OTAFirmwareService()
        fw_id = service.upload("dev-001", b"firmware-binary", version="1.1.0")
        assert fw_id is not None

    def test_trigger_update(self):
        """Triggering an update creates an update job."""
        service = OTAFirmwareService()
        fw_id = service.upload("dev-001", b"firmware-binary", version="1.1.0")
        job_id = service.trigger_update("dev-001", fw_id)
        assert job_id is not None

    def test_update_status(self):
        """Update status is tracked."""
        service = OTAFirmwareService()
        fw_id = service.upload("dev-001", b"firmware-binary", version="1.1.0")
        job_id = service.trigger_update("dev-001", fw_id)
        status = service.get_status(job_id)
        assert status["state"] in ["pending", "in_progress", "completed", "failed"]

    def test_rollback(self):
        """Rollback reverts to previous firmware."""
        service = OTAFirmwareService()
        fw_id = service.upload("dev-001", b"firmware-v1", version="1.0.0")
        service.trigger_update("dev-001", fw_id)
        result = service.rollback("dev-001")
        assert result is True


# ===========================================================================
# Model Validator (IOT-010)
# ===========================================================================


class TestModelValidator:
    """Smart data model validation."""

    def test_valid_crop_model(self):
        """A valid Crop model passes validation."""
        validator = ModelValidator()
        model = {
            "id": "urn:ngsi-ld:Crop:field1",
            "type": "Crop",
            "growthStage": {"type": "Property", "value": "flowering"},
        }
        result = validator.validate(model, "Crop")
        assert result["valid"] is True

    def test_invalid_type(self):
        """An invalid type fails validation."""
        validator = ModelValidator()
        model = {
            "id": "urn:ngsi-ld:Crop:field1",
            "type": "Crop",
            "growthStage": {"type": "Property", "value": "flowering"},
        }
        result = validator.validate(model, "Soil")
        assert result["valid"] is False

    def test_missing_required_field(self):
        """Missing required field fails validation."""
        validator = ModelValidator()
        model = {
            "id": "urn:ngsi-ld:Crop:field1",
            "type": "Crop",
        }
        result = validator.validate(model, "Crop")
        assert result["valid"] is False

    def test_invalid_property_structure(self):
        """Invalid property structure fails validation."""
        validator = ModelValidator()
        model = {
            "id": "urn:ngsi-ld:Crop:field1",
            "type": "Crop",
            "growthStage": {"type": "InvalidType", "value": "flowering"},
        }
        result = validator.validate(model, "Crop")
        assert result["valid"] is False


# ===========================================================================
# Horizontal Scaling (IOT-011)
# ===========================================================================


class TestHorizontalScaling:
    """Horizontal scaling support."""

    def test_partition_assignment(self):
        """Partition assignment is consistent."""
        from src.iot.data_pipeline import PartitionAssigner

        assigner = PartitionAssigner(num_partitions=4)
        p1 = assigner.get_partition("sensor-001")
        p2 = assigner.get_partition("sensor-001")
        assert p1 == p2  # Consistent hashing

    def test_partition_distribution(self):
        """Partitions are distributed across sensors."""
        from src.iot.data_pipeline import PartitionAssigner

        assigner = PartitionAssigner(num_partitions=4)
        partitions = set()
        for i in range(100):
            partitions.add(assigner.get_partition(f"sensor-{i}"))
        assert len(partitions) > 1  # Should use multiple partitions

    def test_consumer_group(self):
        """Consumer group tracks membership."""
        from src.iot.data_pipeline import ConsumerGroup

        group = ConsumerGroup("test-group")
        group.join("consumer-1")
        group.join("consumer-2")
        assert group.size() == 2
        group.leave("consumer-1")
        assert group.size() == 1
