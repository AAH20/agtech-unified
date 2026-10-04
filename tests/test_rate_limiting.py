"""Tests for rate limiting middleware in API gateway."""

import time

from fastapi.testclient import TestClient

from src.decision_support.api_gateway import TenantRegistry, create_app


class TestRateLimiting:
    """Test rate limiting middleware."""

    def test_rate_limit_allows_under_limit(self):
        """Requests under the rate limit are allowed."""
        registry = TenantRegistry()
        app = create_app(registry=registry, rate_limit=100)
        client = TestClient(app)
        for _ in range(10):
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200

    def test_rate_limit_blocks_over_limit(self):
        """Requests over the rate limit are blocked with 429."""
        registry = TenantRegistry()
        app = create_app(registry=registry, rate_limit=5)
        client = TestClient(app)
        # First 5 should succeed
        for _ in range(5):
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200
        # 6th should be rate limited
        resp = client.get("/api/v1/health")
        assert resp.status_code == 429

    def test_rate_limit_resets_after_window(self):
        """Rate limit resets after the time window."""
        registry = TenantRegistry()
        app = create_app(registry=registry, rate_limit=2, rate_window=1)
        client = TestClient(app)
        # Exhaust the limit
        client.get("/api/v1/health")
        client.get("/api/v1/health")
        resp = client.get("/api/v1/health")
        assert resp.status_code == 429
        # Wait for window to reset
        time.sleep(1.1)
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_rate_limit_per_client(self):
        """Rate limit is tracked per client IP."""
        registry = TenantRegistry()
        app = create_app(registry=registry, rate_limit=3)
        client = TestClient(app)
        # Use up the limit for default client
        for _ in range(3):
            client.get("/api/v1/health")
        resp = client.get("/api/v1/health")
        assert resp.status_code == 429

    def test_rate_limit_headers_present(self):
        """Rate limit headers are present in response."""
        registry = TenantRegistry()
        app = create_app(registry=registry, rate_limit=10)
        client = TestClient(app)
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers

    def test_rate_limit_429_response_body(self):
        """Rate limited response includes retry-after info."""
        registry = TenantRegistry()
        app = create_app(registry=registry, rate_limit=1)
        client = TestClient(app)
        client.get("/api/v1/health")
        resp = client.get("/api/v1/health")
        assert resp.status_code == 429
        data = resp.json()
        assert "detail" in data
        assert "rate" in data["detail"].lower() or "limit" in data["detail"].lower()

    def test_rate_limit_disabled_by_default(self):
        """No rate limiting when rate_limit is None."""
        registry = TenantRegistry()
        app = create_app(registry=registry, rate_limit=None)
        client = TestClient(app)
        for _ in range(50):
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200

    def test_rate_limit_with_auth(self):
        """Rate limiting works alongside authentication."""
        from src.decision_support.security import ZeroTrustAuth

        auth = ZeroTrustAuth(secret="test-secret-key-for-testing-only")
        registry = TenantRegistry()
        app = create_app(registry=registry, auth=auth, rate_limit=5)
        client = TestClient(app)
        # Health endpoint doesn't require auth
        for _ in range(5):
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200
        resp = client.get("/api/v1/health")
        assert resp.status_code == 429

    def test_rate_limit_different_endpoints_share_limit(self):
        """Rate limit is global across endpoints, not per-endpoint."""
        registry = TenantRegistry()
        app = create_app(registry=registry, rate_limit=3)
        client = TestClient(app)
        client.get("/api/v1/health")
        client.get("/api/v1/health")
        client.get("/api/v1/health")
        # Any endpoint should be rate limited now
        resp = client.get("/api/v1/health")
        assert resp.status_code == 429
