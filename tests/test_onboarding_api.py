"""Tests for the onboarding REST API (src/onboarding/api.py).

Covers:
- Health endpoint
- Tier recommendation endpoint
- Module listing endpoints
- Setup status endpoints
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.onboarding.api import create_onboarding_api


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the onboarding API."""
    app = create_onboarding_api()
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_returns_healthy_status(self, client):
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert data["status"] == "healthy"

    def test_health_returns_version(self, client):
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "version" in data


class TestTierRecommendation:
    """Tests for POST /api/v1/onboarding/recommend."""

    def test_recommend_startup(self, client):
        resp = client.post(
            "/api/v1/onboarding/recommend",
            json={"employees": 5, "revenue": 50000.0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "startup"

    def test_recommend_smb(self, client):
        resp = client.post(
            "/api/v1/onboarding/recommend",
            json={"employees": 100, "revenue": 1000000.0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "smb"

    def test_recommend_enterprise(self, client):
        resp = client.post(
            "/api/v1/onboarding/recommend",
            json={"employees": 1000, "revenue": 100000000.0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "enterprise"

    def test_recommend_returns_score(self, client):
        resp = client.post(
            "/api/v1/onboarding/recommend",
            json={"employees": 50, "revenue": 500000.0},
        )
        data = resp.json()
        assert "total_score" in data
        assert isinstance(data["total_score"], float)
        assert data["total_score"] >= 0

    def test_recommend_returns_factors(self, client):
        resp = client.post(
            "/api/v1/onboarding/recommend",
            json={"employees": 50, "revenue": 500000.0},
        )
        data = resp.json()
        assert "factors" in data
        assert "employees" in data["factors"]
        assert "revenue" in data["factors"]
        assert "fields" in data["factors"]
        assert "sensors" in data["factors"]
        assert "dedicated_it" in data["factors"]

    def test_recommend_returns_modules(self, client):
        resp = client.post(
            "/api/v1/onboarding/recommend",
            json={"employees": 5, "revenue": 50000.0},
        )
        data = resp.json()
        assert "modules" in data
        assert isinstance(data["modules"], list)
        assert "basics" in data["modules"]

    def test_recommend_returns_install_order(self, client):
        resp = client.post(
            "/api/v1/onboarding/recommend",
            json={"employees": 5, "revenue": 50000.0},
        )
        data = resp.json()
        assert "install_order" in data
        assert isinstance(data["install_order"], list)
        assert len(data["install_order"]) > 0

    def test_recommend_with_all_factors(self, client):
        resp = client.post(
            "/api/v1/onboarding/recommend",
            json={
                "employees": 8,
                "revenue": 200000.0,
                "fields": 30,
                "sensors": 50,
                "has_dedicated_it": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] in ("startup", "smb", "enterprise")

    def test_recommend_missing_employees_returns_422(self, client):
        resp = client.post(
            "/api/v1/onboarding/recommend",
            json={"revenue": 50000.0},
        )
        assert resp.status_code == 422

    def test_recommend_negative_employees_returns_422(self, client):
        resp = client.post(
            "/api/v1/onboarding/recommend",
            json={"employees": -1, "revenue": 50000.0},
        )
        assert resp.status_code == 422

    def test_recommend_missing_revenue_returns_422(self, client):
        resp = client.post(
            "/api/v1/onboarding/recommend",
            json={"employees": 10},
        )
        assert resp.status_code == 422

    def test_recommend_modules_match_tier(self, client):
        resp = client.post(
            "/api/v1/onboarding/recommend",
            json={"employees": 1000, "revenue": 100000000.0},
        )
        data = resp.json()
        assert "integrations" in data["modules"]
        assert "sla" in data["modules"]

    def test_recommend_install_order_is_valid(self, client):
        resp = client.post(
            "/api/v1/onboarding/recommend",
            json={"employees": 1000, "revenue": 100000000.0},
        )
        data = resp.json()
        # basics must come before sensors, sensors before alerts
        order = data["install_order"]
        assert order.index("basics") < order.index("sensors")
        assert order.index("sensors") < order.index("alerts")


class TestModuleListing:
    """Tests for GET /api/v1/onboarding/modules."""

    def test_list_all_modules_returns_200(self, client):
        resp = client.get("/api/v1/onboarding/modules")
        assert resp.status_code == 200

    def test_list_all_modules_returns_list(self, client):
        resp = client.get("/api/v1/onboarding/modules")
        data = resp.json()
        assert isinstance(data, list)

    def test_list_all_modules_contains_all_tiers(self, client):
        resp = client.get("/api/v1/onboarding/modules")
        data = resp.json()
        tiers = {item["tier"] for item in data}
        assert tiers == {"startup", "smb", "enterprise"}

    def test_list_all_modules_has_total(self, client):
        resp = client.get("/api/v1/onboarding/modules")
        data = resp.json()
        for item in data:
            assert "total" in item
            assert item["total"] == len(item["modules"])

    def test_list_all_modules_startup(self, client):
        resp = client.get("/api/v1/onboarding/modules")
        data = resp.json()
        startup = next(item for item in data if item["tier"] == "startup")
        assert "basics" in startup["modules"]
        assert "sensors" in startup["modules"]
        assert "alerts" in startup["modules"]

    def test_list_all_modules_enterprise_has_integrations(self, client):
        resp = client.get("/api/v1/onboarding/modules")
        data = resp.json()
        enterprise = next(item for item in data if item["tier"] == "enterprise")
        assert "integrations" in enterprise["modules"]
        assert "sla" in enterprise["modules"]


class TestModuleListingByTier:
    """Tests for GET /api/v1/onboarding/modules/{tier}."""

    def test_get_modules_startup(self, client):
        resp = client.get("/api/v1/onboarding/modules/startup")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "startup"
        assert "basics" in data["modules"]

    def test_get_modules_smb(self, client):
        resp = client.get("/api/v1/onboarding/modules/smb")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "smb"
        assert "analytics" in data["modules"]
        assert "reporting" in data["modules"]

    def test_get_modules_enterprise(self, client):
        resp = client.get("/api/v1/onboarding/modules/enterprise")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "enterprise"
        assert "integrations" in data["modules"]
        assert "sla" in data["modules"]

    def test_get_modules_unknown_tier_returns_404(self, client):
        resp = client.get("/api/v1/onboarding/modules/nonexistent")
        assert resp.status_code == 404

    def test_get_modules_returns_total(self, client):
        resp = client.get("/api/v1/onboarding/modules/startup")
        data = resp.json()
        assert "total" in data
        assert data["total"] == len(data["modules"])


class TestSetupStatus:
    """Tests for GET /api/v1/onboarding/status."""

    def test_status_returns_200(self, client):
        resp = client.get("/api/v1/onboarding/status")
        assert resp.status_code == 200

    def test_status_returns_dict(self, client):
        resp = client.get("/api/v1/onboarding/status")
        data = resp.json()
        assert isinstance(data, dict)

    def test_status_has_health_field(self, client):
        resp = client.get("/api/v1/onboarding/status")
        data = resp.json()
        assert "health_status" in data

    def test_status_returns_org_id(self, client):
        resp = client.get("/api/v1/onboarding/status")
        data = resp.json()
        assert "org_id" in data

    def test_status_returns_progress(self, client):
        resp = client.get("/api/v1/onboarding/status")
        data = resp.json()
        assert "progress" in data

    def test_status_returns_tier(self, client):
        resp = client.get("/api/v1/onboarding/status")
        data = resp.json()
        assert "tier" in data

    def test_status_returns_profile_complete(self, client):
        resp = client.get("/api/v1/onboarding/status")
        data = resp.json()
        assert "profile_complete" in data


class TestSetupStatusByOrgId:
    """Tests for GET /api/v1/onboarding/status/{org_id}."""

    def test_status_unknown_org_returns_404(self, client):
        resp = client.get("/api/v1/onboarding/status/nonexistent-org-id")
        assert resp.status_code == 404


class TestSessionManagement:
    """Tests for session creation and status."""

    def test_create_session_returns_201(self, client):
        resp = client.post(
            "/api/v1/onboarding/sessions",
            json={"employees": 100, "revenue": 1000000.0},
        )
        assert resp.status_code == 201

    def test_create_session_returns_org_id(self, client):
        resp = client.post(
            "/api/v1/onboarding/sessions",
            json={"employees": 100, "revenue": 1000000.0},
        )
        data = resp.json()
        assert "org_id" in data

    def test_create_session_returns_tier(self, client):
        resp = client.post(
            "/api/v1/onboarding/sessions",
            json={"employees": 100, "revenue": 1000000.0},
        )
        data = resp.json()
        assert data["tier"] == "smb"

    def test_create_session_returns_profile_complete(self, client):
        resp = client.post(
            "/api/v1/onboarding/sessions",
            json={"employees": 100, "revenue": 1000000.0},
        )
        data = resp.json()
        assert data["profile_complete"] is True

    def test_create_session_returns_setup_steps(self, client):
        resp = client.post(
            "/api/v1/onboarding/sessions",
            json={"employees": 100, "revenue": 1000000.0},
        )
        data = resp.json()
        assert "setup_steps" in data
        assert len(data["setup_steps"]) > 0

    def test_get_session_status_returns_200(self, client):
        create_resp = client.post(
            "/api/v1/onboarding/sessions",
            json={"employees": 100, "revenue": 1000000.0},
        )
        org_id = create_resp.json()["org_id"]
        resp = client.get(f"/api/v1/onboarding/status/{org_id}")
        assert resp.status_code == 200

    def test_get_session_status_returns_tier(self, client):
        create_resp = client.post(
            "/api/v1/onboarding/sessions",
            json={"employees": 100, "revenue": 1000000.0},
        )
        org_id = create_resp.json()["org_id"]
        resp = client.get(f"/api/v1/onboarding/status/{org_id}")
        data = resp.json()
        assert data["tier"] == "smb"

    def test_get_session_status_returns_progress(self, client):
        create_resp = client.post(
            "/api/v1/onboarding/sessions",
            json={"employees": 100, "revenue": 1000000.0},
        )
        org_id = create_resp.json()["org_id"]
        resp = client.get(f"/api/v1/onboarding/status/{org_id}")
        data = resp.json()
        assert data["progress"] >= 0.0

    def test_create_session_with_invalid_data_returns_422(self, client):
        resp = client.post(
            "/api/v1/onboarding/sessions",
            json={"employees": -1, "revenue": 1000000.0},
        )
        assert resp.status_code == 422
