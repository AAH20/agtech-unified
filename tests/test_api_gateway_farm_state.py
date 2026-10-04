"""Tests for API gateway using shared FarmState from integration layer."""

import pytest
from fastapi.testclient import TestClient

from src.decision_support.api_gateway import APIGateway, TenantRegistry
from src.integration.farm_state import FarmState


@pytest.fixture()
def gateway():
    registry = TenantRegistry()
    gw = APIGateway(registry=registry)
    return gw


@pytest.fixture()
def client(gateway):
    return TestClient(gateway.app)


@pytest.fixture()
def tenant_id(client):
    resp = client.post("/api/v1/tenants", headers={"name": "Test Farm Co"})
    assert resp.status_code == 201
    return resp.json()["id"]


@pytest.fixture()
def farm_id(client, tenant_id):
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


class TestAPIGatewayFarmState:
    """API gateway uses shared FarmState for validation."""

    def test_recommend_validates_farm_state(self):
        """Recommendation endpoint validates FarmState through shared model."""
        # This should work — valid FarmState values
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.3,
            nutrient_level=0.5,
            pest_pressure=0.1,
        )
        assert state.is_valid() is True

    def test_recommend_rejects_invalid_farm_state(self):
        """Invalid FarmState values are rejected by validation."""
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.3,
            nutrient_level=0.5,
            pest_pressure=0.1,
        )
        # Manually corrupt the state
        state.soil_moisture = 1.5
        assert state.is_valid() is False

    def test_recommend_endpoint_with_valid_data(self, client, tenant_id):
        """Recommendation endpoint works with valid farm state data."""
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

    def test_recommend_endpoint_boundary_values(self, client, tenant_id):
        """Recommendation endpoint accepts boundary FarmState values."""
        resp = client.post(
            f"/api/v1/tenants/{tenant_id}/recommendations",
            json={
                "soil_moisture": 0.0,
                "temperature": -50.0,
                "crop_height": 0.0,
                "nutrient_level": 0.0,
                "pest_pressure": 0.0,
            },
        )
        assert resp.status_code == 200

    def test_recommend_endpoint_upper_boundary(self, client, tenant_id):
        """Recommendation endpoint accepts upper boundary FarmState values."""
        resp = client.post(
            f"/api/v1/tenants/{tenant_id}/recommendations",
            json={
                "soil_moisture": 1.0,
                "temperature": 60.0,
                "crop_height": 10.0,
                "nutrient_level": 1.0,
                "pest_pressure": 1.0,
            },
        )
        assert resp.status_code == 200
