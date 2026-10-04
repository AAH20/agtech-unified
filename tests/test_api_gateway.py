"""Tests for the AgTech Unified API Gateway (REST endpoints)."""

import pytest
from fastapi.testclient import TestClient

from src.decision_support.api_gateway import APIGateway, TenantRegistry


@pytest.fixture()
def gateway():
    """Create a fresh APIGateway with isolated registry."""
    registry = TenantRegistry()
    gw = APIGateway(registry=registry)
    return gw


@pytest.fixture()
def client(gateway):
    """Create a TestClient for the gateway."""
    return TestClient(gateway.app)


@pytest.fixture()
def tenant_id(client):
    """Create a tenant and return its ID."""
    resp = client.post("/api/v1/tenants", headers={"name": "Test Farm Co"})
    assert resp.status_code == 201
    return resp.json()["id"]


@pytest.fixture()
def farm_id(client, tenant_id):
    """Create a farm and return its ID."""
    resp = client.post(
        f"/api/v1/tenants/{tenant_id}/farms",
        json={
            "name": "North Field",
            "location": "40.7128,-74.0060",
            "crop_type": "corn",
            "area_hectares": 50.0,
        },
    )
    assert resp.status_code == 201
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_check(client):
    """Health endpoint returns healthy status."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert data["tenants"] == 0


# ---------------------------------------------------------------------------
# Tenant CRUD
# ---------------------------------------------------------------------------


def test_create_tenant(client):
    """Creating a tenant returns 201 with tenant data."""
    resp = client.post("/api/v1/tenants", headers={"name": "Green Valley Farms"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Green Valley Farms"
    assert "id" in data


def test_list_tenants(client, tenant_id):
    """Listing tenants returns all created tenants."""
    client.post("/api/v1/tenants", headers={"name": "Second Farm"})
    resp = client.get("/api/v1/tenants")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    names = {t["name"] for t in data}
    assert "Test Farm Co" in names
    assert "Second Farm" in names


def test_delete_tenant(client, tenant_id):
    """Deleting a tenant removes it."""
    resp = client.delete(f"/api/v1/tenants/{tenant_id}")
    assert resp.status_code == 204
    resp2 = client.get("/api/v1/tenants")
    assert resp2.json() == []


# ---------------------------------------------------------------------------
# Farm CRUD
# ---------------------------------------------------------------------------


def test_create_farm(client, tenant_id):
    """Creating a farm under a tenant."""
    resp = client.post(
        f"/api/v1/tenants/{tenant_id}/farms",
        json={
            "name": "South Field",
            "location": "41.0,-73.0",
            "crop_type": "wheat",
            "area_hectares": 30.0,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "South Field"
    assert data["tenant_id"] == tenant_id
    assert data["crop_type"] == "wheat"
    assert data["area_hectares"] == 30.0


def test_list_farms(client, tenant_id, farm_id):
    """Listing farms for a tenant."""
    client.post(
        f"/api/v1/tenants/{tenant_id}/farms",
        json={
            "name": "Field B",
            "location": "loc2",
            "crop_type": "soy",
            "area_hectares": 20.0,
        },
    )
    resp = client.get(f"/api/v1/tenants/{tenant_id}/farms")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    names = {f["name"] for f in data}
    assert names == {"North Field", "Field B"}


def test_get_farm(client, tenant_id, farm_id):
    """Getting a specific farm by ID."""
    resp = client.get(f"/api/v1/tenants/{tenant_id}/farms/{farm_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == farm_id
    assert data["name"] == "North Field"
    assert data["tenant_id"] == tenant_id


# ---------------------------------------------------------------------------
# Sensor readings
# ---------------------------------------------------------------------------


def test_add_and_list_readings(client, tenant_id, farm_id):
    """Adding and listing sensor readings for a farm."""
    resp = client.post(
        f"/api/v1/tenants/{tenant_id}/farms/{farm_id}/readings",
        json={"sensor_id": "sensor-001", "metric": "soil_moisture", "value": 0.35, "unit": "ratio"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["sensor_id"] == "sensor-001"
    assert data["metric"] == "soil_moisture"
    assert data["value"] == 0.35
    assert data["unit"] == "ratio"
    assert "timestamp" in data

    resp2 = client.get(f"/api/v1/tenants/{tenant_id}/farms/{farm_id}/readings")
    assert resp2.status_code == 200
    readings = resp2.json()
    assert len(readings) == 1
    assert readings[0]["sensor_id"] == "sensor-001"


# ---------------------------------------------------------------------------
# Actuator commands
# ---------------------------------------------------------------------------


def test_send_and_list_commands(client, tenant_id, farm_id):
    """Sending and listing actuator commands for a farm."""
    resp = client.post(
        f"/api/v1/tenants/{tenant_id}/farms/{farm_id}/commands",
        json={"actuator_type": "irrigation_valve", "action": "open"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["actuator_type"] == "irrigation_valve"
    assert data["action"] == "open"
    assert data["status"] == "pending"
    assert "id" in data
    assert "issued_at" in data

    resp2 = client.get(f"/api/v1/tenants/{tenant_id}/farms/{farm_id}/commands")
    assert resp2.status_code == 200
    commands = resp2.json()
    assert len(commands) == 1
    assert commands[0]["actuator_type"] == "irrigation_valve"


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def test_recommendations(client, tenant_id):
    """Getting recommendations for a farm state."""
    resp = client.post(
        f"/api/v1/tenants/{tenant_id}/recommendations",
        json={
            "soil_moisture": 0.1,
            "temperature": 25.0,
            "crop_height": 0.5,
            "nutrient_level": 0.5,
            "pest_pressure": 0.0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["recommendations"]) >= 1
    assert data["priority_score"] > 0.0
    assert data["algorithm"] == "rule_based"
    assert "irrigate" in data["actions"]
