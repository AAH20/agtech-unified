"""Tests for enhanced security: tenant tokens, revocation, RBAC, MFA, password policy."""

import time

import pytest

from src.decision_support.security import (
    AuthenticationError,
    PasswordPolicy,
    RoleHierarchy,
    TokenBlacklist,
    TOTPProvider,
    ZeroTrustAuth,
)

SECRET = "test-secret-key-that-is-long-enough-for-hs256"


class TestTenantAwareTokens:
    def test_generate_token_includes_tenant_id(self):
        auth = ZeroTrustAuth(secret=SECRET)
        token = auth.generate_token(subject="user-1", roles=["operator"], tenant_id="t-42")
        payload = auth.validate_token(token)
        assert payload["tenant_id"] == "t-42"

    def test_generate_token_without_tenant_id(self):
        auth = ZeroTrustAuth(secret=SECRET)
        token = auth.generate_token(subject="user-1", roles=["operator"])
        payload = auth.validate_token(token)
        assert "tenant_id" not in payload or payload.get("tenant_id") is None

    def test_authenticate_checks_tenant_access(self):
        auth = ZeroTrustAuth(secret=SECRET)
        token = auth.generate_token(subject="user-1", roles=["operator"], tenant_id="t-1")
        # Should succeed when tenant matches
        payload = auth.authenticate(token, required_role="operator", tenant_id="t-1")
        assert payload["sub"] == "user-1"

    def test_authenticate_rejects_wrong_tenant(self):
        auth = ZeroTrustAuth(secret=SECRET)
        token = auth.generate_token(subject="user-1", roles=["operator"], tenant_id="t-1")
        with pytest.raises(AuthenticationError, match="tenant"):
            auth.authenticate(token, required_role="operator", tenant_id="t-2")


class TestTokenBlacklist:
    def test_blacklist_revokes_token(self):
        auth = ZeroTrustAuth(secret=SECRET)
        blacklist = TokenBlacklist()
        auth.blacklist = blacklist
        token = auth.generate_token(subject="user-1", roles=["operator"])
        # Token works before revocation
        auth.validate_token(token)
        # Revoke it
        blacklist.revoke(token, ttl_seconds=3600)
        # Now it should fail
        with pytest.raises(AuthenticationError, match="revoked"):
            auth.validate_token(token)

    def test_blacklist_is_empty_by_default(self):
        blacklist = TokenBlacklist()
        assert blacklist.is_revoked("nonexistent-token") is False

    def test_blacklist_cleanup_removes_expired(self):
        blacklist = TokenBlacklist()
        blacklist._tokens["expired-token"] = time.time() - 100
        blacklist._tokens["valid-token"] = time.time() + 3600
        blacklist.cleanup()
        assert blacklist.is_revoked("expired-token") is False
        assert blacklist.is_revoked("valid-token") is True

    def test_authenticate_rejects_blacklisted_token(self):
        auth = ZeroTrustAuth(secret=SECRET)
        blacklist = TokenBlacklist()
        auth.blacklist = blacklist
        token = auth.generate_token(subject="user-1", roles=["operator"])
        blacklist.revoke(token)
        with pytest.raises(AuthenticationError):
            auth.authenticate(token)


class TestRoleHierarchy:
    def test_admin_inherits_operator_permissions(self):
        hierarchy = RoleHierarchy()
        assert hierarchy.has_permission("admin", "operator") is True

    def test_operator_inherits_viewer_permissions(self):
        hierarchy = RoleHierarchy()
        assert hierarchy.has_permission("operator", "viewer") is True

    def test_viewer_does_not_inherit_admin(self):
        hierarchy = RoleHierarchy()
        assert hierarchy.has_permission("viewer", "admin") is False

    def test_admin_inherits_viewer_through_chain(self):
        hierarchy = RoleHierarchy()
        assert hierarchy.has_permission("admin", "viewer") is True

    def test_custom_role_hierarchy(self):
        hierarchy = RoleHierarchy()
        hierarchy.add_role("supervisor", inherits_from=["operator"])
        assert hierarchy.has_permission("supervisor", "operator") is True
        assert hierarchy.has_permission("supervisor", "viewer") is True
        assert hierarchy.has_permission("supervisor", "admin") is False

    def test_authorize_with_hierarchy(self):
        auth = ZeroTrustAuth(secret=SECRET)
        hierarchy = RoleHierarchy()
        auth.role_hierarchy = hierarchy
        token = auth.generate_token(subject="user-1", roles=["admin"])
        # Admin should pass operator check via hierarchy
        payload = auth.authenticate(token, required_role="operator")
        assert payload["sub"] == "user-1"


class TestMFA:
    def test_totp_generates_valid_code(self):
        totp = TOTPProvider(secret="JBSWY3DPEHPK3PXP")
        code = totp.generate_code()
        assert len(code) == 6
        assert code.isdigit()

    def test_totp_verifies_correct_code(self):
        totp = TOTPProvider(secret="JBSWY3DPEHPK3PXP")
        code = totp.generate_code()
        assert totp.verify(code) is True

    def test_totp_rejects_wrong_code(self):
        totp = TOTPProvider(secret="JBSWY3DPEHPK3PXP")
        assert totp.verify("000000") is False

    def test_totp_rejects_expired_code(self):
        totp = TOTPProvider(secret="JBSWY3DPEHPK3PXP")
        # Generate code for a different time window
        code = totp.generate_code(timestamp=time.time() - 60)
        assert totp.verify(code) is False

    def test_mfa_enforcement_per_tenant(self):
        auth = ZeroTrustAuth(secret=SECRET)
        auth.mfa_required_tenants = {"t-secure"}
        token = auth.generate_token(subject="user-1", roles=["operator"], tenant_id="t-secure")
        # Should require MFA for this tenant
        with pytest.raises(AuthenticationError, match="MFA"):
            auth.authenticate(token, tenant_id="t-secure", mfa_code=None)

    def test_mfa_passes_with_correct_code(self):
        auth = ZeroTrustAuth(secret=SECRET)
        auth.mfa_required_tenants = {"t-secure"}
        totp = TOTPProvider(secret="JBSWY3DPEHPK3PXP")
        auth._totp_provider = totp
        token = auth.generate_token(subject="user-1", roles=["operator"], tenant_id="t-secure")
        code = totp.generate_code()
        payload = auth.authenticate(token, tenant_id="t-secure", mfa_code=code)
        assert payload["sub"] == "user-1"


class TestPasswordPolicy:
    def test_strong_password_passes(self):
        policy = PasswordPolicy()
        policy.validate("Str0ng!Pass#2024")  # Should not raise

    def test_short_password_fails(self):
        policy = PasswordPolicy(min_length=8)
        with pytest.raises(ValueError, match="length"):
            policy.validate("Short1!")

    def test_password_without_uppercase_fails(self):
        policy = PasswordPolicy(require_uppercase=True)
        with pytest.raises(ValueError, match="uppercase"):
            policy.validate("lowercase1!")

    def test_password_without_lowercase_fails(self):
        policy = PasswordPolicy(require_lowercase=True)
        with pytest.raises(ValueError, match="lowercase"):
            policy.validate("UPPERCASE1!")

    def test_password_without_digit_fails(self):
        policy = PasswordPolicy(require_digit=True)
        with pytest.raises(ValueError, match="digit"):
            policy.validate("NoDigits!")

    def test_password_without_special_fails(self):
        policy = PasswordPolicy(require_special=True)
        with pytest.raises(ValueError, match="special"):
            policy.validate("NoSpecial1")

    def test_password_history_prevents_reuse(self):
        policy = PasswordPolicy(history_size=3)
        policy.add_to_history("OldPass1!")
        with pytest.raises(ValueError, match="history"):
            policy.validate("OldPass1!")

    def test_common_password_rejected(self):
        policy = PasswordPolicy()
        with pytest.raises(ValueError, match="common"):
            policy.validate("password")


class TestAuthAuditLogging:
    def test_successful_auth_logged(self):
        from src.decision_support.audit import AuditEventType, AuditLogger

        audit = AuditLogger()
        auth = ZeroTrustAuth(secret=SECRET, audit_logger=audit)
        token = auth.generate_token(subject="user-1", roles=["operator"])
        auth.authenticate(token)
        events = audit.get_events(event_type=AuditEventType.AUTH_SUCCESS)
        assert len(events) == 1
        assert events[0].actor == "user-1"

    def test_failed_auth_logged(self):
        from src.decision_support.audit import AuditEventType, AuditLogger

        audit = AuditLogger()
        auth = ZeroTrustAuth(secret=SECRET, audit_logger=audit)
        token = auth.generate_token(subject="user-1", roles=["operator"])
        # Tamper with token to make it invalid
        bad_token = token[:-5] + "XXXXX"
        with pytest.raises(AuthenticationError):
            auth.authenticate(bad_token)
        events = audit.get_events(event_type=AuditEventType.AUTH_FAILURE)
        assert len(events) == 1
