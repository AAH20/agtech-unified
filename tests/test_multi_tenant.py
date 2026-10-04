"""Tests for multi-tenant dashboard and alert isolation."""

import pytest
from fastapi.testclient import TestClient

from src.decision_support.alerts import (
    AlertManager,
    EscalationPolicy,
    NotificationChannel,
    Threshold,
)
from src.decision_support.dashboard import DashboardConfig, FarmDashboard
from src.decision_support.security import ZeroTrustAuth

SECRET = "test-secret-key-that-is-long-enough-for-hs256"


@pytest.fixture
def auth():
    return ZeroTrustAuth(secret=SECRET)


class TestMultiTenantDashboard:
    def test_dashboard_has_tenant_id(self):
        """Dashboard config includes tenant_id."""
        config = DashboardConfig(tenant_id="t-1")
        dashboard = FarmDashboard(config=config)
        assert dashboard.tenant_id == "t-1"

    def test_dashboard_state_includes_tenant(self):
        """Dashboard state dict includes tenant_id."""
        config = DashboardConfig(tenant_id="t-42")
        dashboard = FarmDashboard(config=config)
        state = dashboard.get_state_dict()
        assert state["tenant_id"] == "t-42"

    def test_dashboard_ws_requires_auth(self, auth):
        """WebSocket endpoint requires authentication."""
        config = DashboardConfig(tenant_id="t-1")
        dashboard = FarmDashboard(config=config, auth=auth)
        app = dashboard.create_app()
        client = TestClient(app)
        # Connecting without token should fail
        with pytest.raises(Exception):
            with client.websocket_connect("/ws"):
                pass

    def test_dashboard_ws_with_valid_token(self, auth):
        """WebSocket accepts connections with valid token."""
        config = DashboardConfig(tenant_id="t-1")
        dashboard = FarmDashboard(config=config, auth=auth)
        app = dashboard.create_app()
        client = TestClient(app)
        token = auth.generate_token(subject="user-1", roles=["viewer"], tenant_id="t-1")
        with client.websocket_connect(f"/ws?token={token}") as ws:
            data = ws.receive_json()
            assert data["type"] == "state"
            assert data["data"]["tenant_id"] == "t-1"

    def test_dashboard_ws_rejects_wrong_tenant(self, auth):
        """WebSocket rejects tokens for a different tenant."""
        config = DashboardConfig(tenant_id="t-1")
        dashboard = FarmDashboard(config=config, auth=auth)
        app = dashboard.create_app()
        client = TestClient(app)
        token = auth.generate_token(subject="user-1", roles=["viewer"], tenant_id="t-2")
        with pytest.raises(Exception):
            with client.websocket_connect(f"/ws?token={token}"):
                pass

    def test_dashboard_rest_requires_auth(self, auth):
        """REST endpoints require authentication."""
        config = DashboardConfig(tenant_id="t-1")
        dashboard = FarmDashboard(config=config, auth=auth)
        app = dashboard.create_app()
        client = TestClient(app)
        resp = client.get("/state")
        assert resp.status_code == 401

    def test_dashboard_rest_with_valid_token(self, auth):
        """REST endpoints work with valid token."""
        config = DashboardConfig(tenant_id="t-1")
        dashboard = FarmDashboard(config=config, auth=auth)
        app = dashboard.create_app()
        client = TestClient(app)
        token = auth.generate_token(subject="user-1", roles=["viewer"], tenant_id="t-1")
        resp = client.get("/state", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["tenant_id"] == "t-1"


class TestMultiTenantAlerts:
    def test_alert_manager_tenant_isolation(self):
        """AlertManager scoped to a tenant only shows that tenant's alerts."""
        manager_t1 = AlertManager(tenant_id="t-1")
        manager_t2 = AlertManager(tenant_id="t-2")

        manager_t1.add_threshold(Threshold(metric="soil_moisture", min_value=0.3, tenant_id="t-1"))
        manager_t2.add_threshold(Threshold(metric="soil_moisture", min_value=0.3, tenant_id="t-2"))

        manager_t1.check_value("soil_moisture", 0.1)
        manager_t2.check_value("soil_moisture", 0.1)

        assert manager_t1.alert_count == 1
        assert manager_t2.alert_count == 1
        assert manager_t1.get_all_alerts()[0].tenant_id == "t-1"
        assert manager_t2.get_all_alerts()[0].tenant_id == "t-2"

    def test_alert_manager_rejects_wrong_tenant_threshold(self):
        """AlertManager rejects thresholds for a different tenant."""
        manager = AlertManager(tenant_id="t-1")
        with pytest.raises(ValueError, match="tenant"):
            manager.add_threshold(Threshold(metric="soil_moisture", min_value=0.3, tenant_id="t-2"))

    def test_alert_manager_rejects_wrong_tenant_channel(self):
        """AlertManager rejects channels for a different tenant."""
        manager = AlertManager(tenant_id="t-1")
        with pytest.raises(ValueError, match="tenant"):
            manager.add_channel(
                NotificationChannel(
                    name="email", channel_type="email", target="a@b.com", tenant_id="t-2"
                )
            )

    def test_alert_deduplication(self):
        """Duplicate alerts within the dedup window are suppressed."""
        manager = AlertManager()
        manager.add_threshold(Threshold(metric="soil_moisture", min_value=0.3))

        alerts1 = manager.check_value("soil_moisture", 0.1)
        alerts2 = manager.check_value("soil_moisture", 0.1)

        assert len(alerts1) == 1
        assert len(alerts2) == 0  # Deduplicated

    def test_alert_escalation_policy(self):
        """Escalation policies can be added and are tenant-scoped."""
        manager = AlertManager(tenant_id="t-1")
        policy = EscalationPolicy(
            name="crit-escalation",
            timeout_seconds=300,
            escalation_targets=["manager@farm.com"],
            tenant_id="t-1",
        )
        manager.add_escalation_policy(policy)
        assert len(manager._escalation_policies) == 1

    def test_alert_lifecycle_audit(self):
        """Alert state changes are audited."""
        from src.decision_support.audit import AuditEventType, AuditLogger

        audit = AuditLogger()
        manager = AlertManager(audit_logger=audit)
        manager.add_threshold(Threshold(metric="soil_moisture", min_value=0.3))

        alerts = manager.check_value("soil_moisture", 0.1)
        manager.acknowledge_alert(alerts[0].id, actor="user-1")
        manager.resolve_alert(alerts[0].id, actor="user-1")

        events = audit.get_events(event_type=AuditEventType.ALERT_STATE_CHANGE)
        assert len(events) == 3  # created + acknowledged + resolved

    def test_alert_get_by_tenant(self):
        """Alerts can be filtered by tenant."""
        manager = AlertManager()
        manager.add_threshold(Threshold(metric="soil_moisture", min_value=0.3, tenant_id="t-1"))
        manager.add_threshold(Threshold(metric="temperature", max_value=35.0, tenant_id="t-2"))

        manager.check_value("soil_moisture", 0.1, tenant_id="t-1")
        manager.check_value("temperature", 40.0, tenant_id="t-2")

        t1_alerts = manager.get_all_alerts(tenant_id="t-1")
        t2_alerts = manager.get_all_alerts(tenant_id="t-2")
        assert len(t1_alerts) == 1
        assert len(t2_alerts) == 1
        assert t1_alerts[0].metric == "soil_moisture"
        assert t2_alerts[0].metric == "temperature"
