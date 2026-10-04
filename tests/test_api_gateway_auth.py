"""Tests for API gateway authentication and audit logging."""

import pytest
from fastapi.testclient import TestClient

from src.decision_support.api_gateway import APIGateway, TenantRegistry
from src.decision_support.security import ZeroTrustAuth

SECRET = "test-secret-key-that-is-long-enough-for-hs256"


@pytest.fixture
def auth():
    return ZeroTrustAuth(secret=SECRET)


@pytest.fixture
def gateway(auth):
    registry = TenantRegistry()
    gw = APIGateway(registry=registry, auth=auth)
    return gw


@pytest.fixture
def client(gateway):
    return TestClient(gateway.app)


@pytest.fixture
def admin_token(auth):
    return auth.generate_token(subject="admin-1", roles=["admin"], tenant_id="t-1")


@pytest.fixture
def operator_token(auth):
    return auth.generate_token(subject="op-1", roles=["operator"], tenant_id="t-1")


@pytest.fixture
def viewer_token(auth):
    return auth.generate_token(subject="viewer-1", roles=["viewer"], tenant_id="t-1")


@pytest.fixture
def tenant_id(client, admin_token):
    resp = client.post(
        "/api/v1/tenants",
        headers={"name": "Test Farm", "Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 201
    return resp.json()["id"]


@pytest.fixture
def admin_token_for_tenant(auth, tenant_id):
    """Create an admin token scoped to the created tenant."""
    return auth.generate_token(subject="admin-1", roles=["admin"], tenant_id=tenant_id)


class TestAPIAuthentication:
    def test_health_endpoint_open(self, client):
        """Health endpoint is accessible without auth."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_create_tenant_requires_auth(self, client):
        """Creating a tenant requires authentication."""
        resp = client.post("/api/v1/tenants", headers={"name": "Test"})
        assert resp.status_code == 401

    def test_create_tenant_with_valid_token(self, client, admin_token):
        """Creating a tenant with a valid admin token succeeds."""
        resp = client.post(
            "/api/v1/tenants",
            headers={"name": "Test Farm", "Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 201
        assert resp.json()["name"] == "Test Farm"

    def test_create_tenant_with_invalid_token(self, client):
        """Creating a tenant with an invalid token fails."""
        resp = client.post(
            "/api/v1/tenants",
            headers={"name": "Test", "Authorization": "Bearer invalid-token"},
        )
        assert resp.status_code == 401

    def test_list_tenants_requires_auth(self, client):
        """Listing tenants requires authentication."""
        resp = client.get("/api/v1/tenants")
        assert resp.status_code == 401

    def test_list_tenants_with_valid_token(self, client, admin_token):
        """Listing tenants with a valid token succeeds."""
        resp = client.get(
            "/api/v1/tenants",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200

    def test_create_farm_requires_auth(self, client, tenant_id):
        """Creating a farm requires authentication."""
        resp = client.post(
            f"/api/v1/tenants/{tenant_id}/farms",
            json={"name": "Field", "location": "loc", "crop_type": "corn", "area_hectares": 10.0},
        )
        assert resp.status_code == 401

    def test_create_farm_with_valid_token(self, client, tenant_id, admin_token_for_tenant):
        """Creating a farm with a valid token succeeds."""
        resp = client.post(
            f"/api/v1/tenants/{tenant_id}/farms",
            json={"name": "Field", "location": "loc", "crop_type": "corn", "area_hectares": 10.0},
            headers={"Authorization": f"Bearer {admin_token_for_tenant}"},
        )
        assert resp.status_code == 201

    def test_viewer_cannot_create_farm(self, client, tenant_id, viewer_token):
        """Viewer role cannot create farms."""
        resp = client.post(
            f"/api/v1/tenants/{tenant_id}/farms",
            json={"name": "Field", "location": "loc", "crop_type": "corn", "area_hectares": 10.0},
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert resp.status_code == 403

    def test_operator_can_create_farm(self, client, tenant_id, auth):
        """Operator role can create farms."""
        operator_token = auth.generate_token(
            subject="op-1", roles=["operator"], tenant_id=tenant_id
        )
        resp = client.post(
            f"/api/v1/tenants/{tenant_id}/farms",
            json={"name": "Field", "location": "loc", "crop_type": "corn", "area_hectares": 10.0},
            headers={"Authorization": f"Bearer {operator_token}"},
        )
        assert resp.status_code == 201

    def test_cross_tenant_access_denied(self, client, admin_token, tenant_id):
        """A token for tenant A cannot access tenant B's farms."""
        # Create a second tenant
        resp = client.post(
            "/api/v1/tenants",
            headers={"name": "Other Farm", "Authorization": f"Bearer {admin_token}"},
        )
        other_tenant_id = resp.json()["id"]

        # Try to access other tenant's farms with first tenant's token
        resp = client.get(
            f"/api/v1/tenants/{other_tenant_id}/farms",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 403

    def test_recommendations_require_auth(self, client, tenant_id):
        """Getting recommendations requires authentication."""
        resp = client.post(
            f"/api/v1/tenants/{tenant_id}/recommendations",
            json={
                "soil_moisture": 0.5,
                "temperature": 25.0,
                "crop_height": 0.5,
                "nutrient_level": 0.5,
                "pest_pressure": 0.0,
            },
        )
        assert resp.status_code == 401

    def test_recommendations_with_valid_token(self, client, tenant_id, auth):
        """Getting recommendations with a valid token succeeds."""
        token = auth.generate_token(subject="user-1", roles=["viewer"], tenant_id=tenant_id)
        resp = client.post(
            f"/api/v1/tenants/{tenant_id}/recommendations",
            json={
                "soil_moisture": 0.5,
                "temperature": 25.0,
                "crop_height": 0.5,
                "nutrient_level": 0.5,
                "pest_pressure": 0.0,
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_actuator_commands_require_auth(self, client, tenant_id):
        """Sending actuator commands requires authentication."""
        resp = client.post(
            f"/api/v1/tenants/{tenant_id}/farms/some-farm/commands",
            json={"actuator_type": "valve", "action": "open"},
        )
        assert resp.status_code == 401

    def test_revoked_token_rejected(self, client, auth):
        """A revoked token is rejected."""
        token = auth.generate_token(subject="user-1", roles=["admin"])
        auth.revoke_token(token)
        resp = client.get(
            "/api/v1/tenants",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 401


class TestAPIAuditLogging:
    def test_api_request_logged(self, client, auth, admin_token):
        """API requests are logged to the audit logger."""
        from src.decision_support.audit import AuditEventType, AuditLogger

        audit = AuditLogger()
        auth._audit_logger = audit

        client.get(
            "/api/v1/tenants",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        events = audit.get_events(event_type=AuditEventType.API_REQUEST)
        assert len(events) >= 1

    def test_auth_failure_logged(self, client, auth):
        """Failed authentication attempts are logged."""
        from src.decision_support.audit import AuditEventType, AuditLogger

        audit = AuditLogger()
        auth._audit_logger = audit

        client.get(
            "/api/v1/tenants",
            headers={"Authorization": "Bearer invalid-token"},
        )
        events = audit.get_events(event_type=AuditEventType.AUTH_FAILURE)
        assert len(events) >= 1

    def test_request_has_correlation_id(self, client, auth, admin_token):
        """API requests include correlation IDs."""
        resp = client.get(
            "/api/v1/tenants",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert "X-Request-ID" in resp.headers


class TestMultiTenantIsolation:
    def test_tenant_cannot_see_other_tenant_farms(self, client, auth):
        """Farms from one tenant are not visible to another."""
        # Create admin tokens for two tenants
        admin_t1 = auth.generate_token(subject="admin-1", roles=["admin"], tenant_id="t-1")
        admin_t2 = auth.generate_token(subject="admin-2", roles=["admin"], tenant_id="t-2")

        # Create tenants
        r1 = client.post(
            "/api/v1/tenants", headers={"name": "Farm A", "Authorization": f"Bearer {admin_t1}"}
        )
        r2 = client.post(
            "/api/v1/tenants", headers={"name": "Farm B", "Authorization": f"Bearer {admin_t2}"}
        )
        t1_id = r1.json()["id"]
        r2.json()["id"]

        # Create farm in tenant 1
        client.post(
            f"/api/v1/tenants/{t1_id}/farms",
            json={"name": "Field A", "location": "loc", "crop_type": "corn", "area_hectares": 10.0},
            headers={"Authorization": f"Bearer {admin_t1}"},
        )

        # Tenant 2 should not see tenant 1's farms
        resp = client.get(
            f"/api/v1/tenants/{t1_id}/farms",
            headers={"Authorization": f"Bearer {admin_t2}"},
        )
        assert resp.status_code == 403

    def test_tenant_deletion_cascades(self, client, auth):
        """Deleting a tenant removes associated farms and data."""
        # Create a global admin token (no tenant_id for creating tenants)
        global_admin = auth.generate_token(subject="admin-1", roles=["admin"])
        r = client.post(
            "/api/v1/tenants", headers={"name": "Farm A", "Authorization": f"Bearer {global_admin}"}
        )
        t_id = r.json()["id"]

        # Create a tenant-scoped token for the new tenant
        tenant_admin = auth.generate_token(subject="admin-1", roles=["admin"], tenant_id=t_id)

        # Create a farm
        client.post(
            f"/api/v1/tenants/{t_id}/farms",
            json={"name": "Field", "location": "loc", "crop_type": "corn", "area_hectares": 10.0},
            headers={"Authorization": f"Bearer {tenant_admin}"},
        )

        # Delete tenant
        resp = client.delete(
            f"/api/v1/tenants/{t_id}",
            headers={"Authorization": f"Bearer {tenant_admin}"},
        )
        assert resp.status_code == 204

        # Farm should no longer be accessible
        resp = client.get(
            f"/api/v1/tenants/{t_id}/farms",
            headers={"Authorization": f"Bearer {tenant_admin}"},
        )
        assert resp.status_code == 404
