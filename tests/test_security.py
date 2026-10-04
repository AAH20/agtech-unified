"""Tests for ZeroTrustAuth — JWT validation, RBAC, rate limiting."""
import time
import pytest
from src.decision_support.security import ZeroTrustAuth, AuthenticationError, AuthorizationError


class TestJWTValidation:
    def test_generate_token_creates_valid_jwt(self):
        auth = ZeroTrustAuth(secret="test-secret-key-that-is-long-enough-for-hs256")
        token = auth.generate_token(subject="sensor-001", roles=["device"])
        assert isinstance(token, str)
        assert len(token.split(".")) == 3  # header.payload.signature

    def test_validate_token_returns_payload(self):
        auth = ZeroTrustAuth(secret="test-secret-key-that-is-long-enough-for-hs256")
        token = auth.generate_token(subject="sensor-001", roles=["device"])
        payload = auth.validate_token(token)
        assert payload["sub"] == "sensor-001"
        assert "device" in payload["roles"]
        assert "exp" in payload
        assert "iat" in payload

    def test_validate_token_rejects_expired_token(self):
        auth = ZeroTrustAuth(secret="test-secret-key-that-is-long-enough-for-hs256")
        token = auth.generate_token(subject="sensor-001", roles=["device"], ttl_seconds=-1)
        with pytest.raises(AuthenticationError, match="expired"):
            auth.validate_token(token)

    def test_validate_token_rejects_invalid_signature(self):
        auth1 = ZeroTrustAuth(secret="secret-a-that-is-long-enough-for-hs256")
        auth2 = ZeroTrustAuth(secret="secret-b-that-is-long-enough-for-hs256")
        token = auth1.generate_token(subject="sensor-001", roles=["device"])
        with pytest.raises(AuthenticationError):
            auth2.validate_token(token)


class TestRoleBasedAccess:
    def test_has_role_checks_membership(self):
        auth = ZeroTrustAuth(secret="test-secret-key")
        payload = {"roles": ["admin", "operator"]}
        assert auth.has_role(payload, "admin") is True
        assert auth.has_role(payload, "viewer") is False

    def test_authorize_allows_required_role(self):
        auth = ZeroTrustAuth(secret="test-secret-key")
        payload = {"roles": ["admin"]}
        assert auth.authorize(payload, "admin") is True

    def test_authorize_denies_missing_role(self):
        auth = ZeroTrustAuth(secret="test-secret-key")
        payload = {"roles": ["viewer"]}
        assert auth.authorize(payload, "admin") is False


class TestRateLimiting:
    def test_rate_limit_allows_under_threshold(self):
        auth = ZeroTrustAuth(secret="test-secret-key")
        assert auth.check_rate_limit("client-1", max_requests=5, window_seconds=60) is True
        assert auth.check_rate_limit("client-1", max_requests=5, window_seconds=60) is True

    def test_rate_limit_blocks_over_threshold(self):
        auth = ZeroTrustAuth(secret="test-secret-key")
        for _ in range(3):
            auth.check_rate_limit("client-2", max_requests=3, window_seconds=60)
        assert auth.check_rate_limit("client-2", max_requests=3, window_seconds=60) is False


class TestAuthenticate:
    def test_authenticate_succeeds_with_valid_token_and_role(self):
        auth = ZeroTrustAuth(secret="test-secret-key-that-is-long-enough-for-hs256")
        token = auth.generate_token(subject="sensor-001", roles=["device"])
        payload = auth.authenticate(token, required_role="device")
        assert payload["sub"] == "sensor-001"

    def test_authenticate_raises_on_missing_role(self):
        auth = ZeroTrustAuth(secret="test-secret-key-that-is-long-enough-for-hs256")
        token = auth.generate_token(subject="sensor-001", roles=["viewer"])
        with pytest.raises(AuthorizationError):
            auth.authenticate(token, required_role="admin")
