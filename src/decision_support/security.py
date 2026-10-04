"""Zero-trust authentication for agricultural IoT API.
Provides JWT validation, role-based access control, rate limiting,
tenant-aware tokens, token revocation, MFA, and password policies.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import jwt

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Raised when authentication fails (invalid/expired token)."""


class AuthorizationError(Exception):
    """Raised when authorization fails (insufficient role)."""


class TokenBlacklist:
    """In-memory token blacklist with TTL-based expiration."""

    def __init__(self):
        self._tokens: Dict[str, float] = {}
        self._lock = threading.Lock()

    def revoke(self, token: str, ttl_seconds: int = 3600) -> None:
        """Add a token to the blacklist."""
        with self._lock:
            self._tokens[token] = time.time() + ttl_seconds

    def is_revoked(self, token: str) -> bool:
        """Check if a token is blacklisted."""
        with self._lock:
            return token in self._tokens

    def cleanup(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.time()
        with self._lock:
            expired = [t for t, exp in self._tokens.items() if exp < now]
            for t in expired:
                del self._tokens[t]
            return len(expired)


class RoleHierarchy:
    """Role hierarchy with inheritance (admin > operator > viewer)."""

    DEFAULT_HIERARCHY = {
        "admin": ["operator", "viewer"],
        "operator": ["viewer"],
        "viewer": [],
    }

    def __init__(self):
        self._hierarchy: Dict[str, List[str]] = dict(self.DEFAULT_HIERARCHY)

    def add_role(self, role: str, inherits_from: List[str]) -> None:
        """Add a custom role with inheritance."""
        self._hierarchy[role] = list(inherits_from)

    def get_inherited_roles(self, role: str) -> Set[str]:
        """Get all roles inherited by the given role (transitive)."""
        result: Set[str] = set()
        stack = list(self._hierarchy.get(role, []))
        while stack:
            current = stack.pop()
            if current not in result:
                result.add(current)
                stack.extend(self._hierarchy.get(current, []))
        return result

    def has_permission(self, user_role: str, required_role: str) -> bool:
        """Check if user_role has permission for required_role via hierarchy."""
        if user_role == required_role:
            return True
        return required_role in self.get_inherited_roles(user_role)


class TOTPProvider:
    """TOTP-based MFA provider (RFC 6238 compatible)."""

    def __init__(self, secret: str, digits: int = 6, interval: int = 30):
        self._secret = secret
        self._digits = digits
        self._interval = interval

    def generate_code(self, timestamp: Optional[float] = None) -> str:
        """Generate a TOTP code for the given timestamp (default: now)."""
        if timestamp is None:
            timestamp = time.time()
        counter = int(timestamp) // self._interval
        return self._hotp(counter)

    def verify(self, code: str, window: int = 1) -> bool:
        """Verify a TOTP code with a tolerance window."""
        now = time.time()
        for offset in range(-window, window + 1):
            counter = int(now) // self._interval + offset
            if hmac.compare_digest(self._hotp(counter), code):
                return True
        return False

    def _hotp(self, counter: int) -> str:
        """Compute HOTP for a counter value."""
        counter_bytes = counter.to_bytes(8, byteorder="big")
        key = self._secret.encode("utf-8")
        digest = hmac.new(key, counter_bytes, hashlib.sha1).digest()
        offset = digest[-1] & 0x0F
        binary = (
            ((digest[offset] & 0x7F) << 24)
            | ((digest[offset + 1] & 0xFF) << 16)
            | ((digest[offset + 2] & 0xFF) << 8)
            | (digest[offset + 3] & 0xFF)
        )
        otp = binary % (10**self._digits)
        return str(otp).zfill(self._digits)


class PasswordPolicy:
    """Configurable password policy with validation."""

    COMMON_PASSWORDS = {
        "password",
        "123456",
        "12345678",
        "qwerty",
        "abc123",
        "monkey",
        "1234567",
        "letmein",
        "trustno1",
        "dragon",
        "baseball",
        "iloveyou",
        "master",
        "sunshine",
        "ashley",
        "michael",
        "shadow",
        "123123",
        "654321",
        "password1",
    }

    def __init__(
        self,
        min_length: int = 8,
        require_uppercase: bool = True,
        require_lowercase: bool = True,
        require_digit: bool = True,
        require_special: bool = True,
        history_size: int = 5,
    ):
        self.min_length = min_length
        self.require_uppercase = require_uppercase
        self.require_lowercase = require_lowercase
        self.require_digit = require_digit
        self.require_special = require_special
        self.history_size = history_size
        self._history: List[str] = []

    def validate(self, password: str) -> None:
        """Validate a password against the policy. Raises ValueError on failure."""
        if len(password) < self.min_length:
            raise ValueError(f"Password length must be at least {self.min_length} characters")
        if password.lower() in self.COMMON_PASSWORDS:
            raise ValueError("Password is too common")
        if self.require_uppercase and not any(c.isupper() for c in password):
            raise ValueError("Password must contain at least one uppercase letter")
        if self.require_lowercase and not any(c.islower() for c in password):
            raise ValueError("Password must contain at least one lowercase letter")
        if self.require_digit and not any(c.isdigit() for c in password):
            raise ValueError("Password must contain at least one digit")
        if self.require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            raise ValueError("Password must contain at least one special character")
        if self._history and password in self._history:
            raise ValueError("Password has been used in recent history")

    def add_to_history(self, password: str) -> None:
        """Add a password to the history after successful change."""
        self._history.append(password)
        if len(self._history) > self.history_size:
            self._history.pop(0)


class ZeroTrustAuth:
    """Zero-trust authentication with JWT, RBAC, rate limiting, and MFA."""

    def __init__(
        self,
        secret: str,
        algorithm: str = "HS256",
        audit_logger: Any = None,
    ):
        self._secret = secret
        self._algorithm = algorithm
        self._rate_limits: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self.blacklist = TokenBlacklist()
        self.role_hierarchy = RoleHierarchy()
        self.mfa_required_tenants: Set[str] = set()
        self._totp_provider: Optional[TOTPProvider] = None
        self._audit_logger = audit_logger

    def generate_token(
        self,
        subject: str,
        roles: List[str],
        ttl_seconds: int = 3600,
        tenant_id: Optional[str] = None,
    ) -> str:
        """Generate a JWT token for the given subject, roles, and optional tenant."""
        now = time.time()
        payload: Dict[str, Any] = {
            "sub": subject,
            "roles": roles,
            "iat": now,
            "exp": now + ttl_seconds,
        }
        if tenant_id is not None:
            payload["tenant_id"] = tenant_id
        return jwt.encode(payload, self._secret, algorithm=self._algorithm)

    def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate a JWT token and return its payload.

        Raises:
            AuthenticationError: If the token is invalid, expired, or revoked.
        """
        if self.blacklist.is_revoked(token):
            raise AuthenticationError("Token has been revoked")
        try:
            payload = jwt.decode(
                token,
                self._secret,
                algorithms=[self._algorithm],
            )
            return payload
        except jwt.ExpiredSignatureError as exc:
            raise AuthenticationError("Token has expired") from exc
        except jwt.InvalidTokenError as exc:
            raise AuthenticationError(f"Invalid token: {exc}") from exc

    def has_role(self, payload: Dict[str, Any], role: str) -> bool:
        """Check if the payload contains the given role."""
        roles = payload.get("roles", [])
        return role in roles

    def authorize(self, payload: Dict[str, Any], required_role: str) -> bool:
        """Check if the payload authorizes the required role via hierarchy."""
        roles = payload.get("roles", [])
        for user_role in roles:
            if self.role_hierarchy.has_permission(user_role, required_role):
                return True
        return False

    def check_rate_limit(
        self,
        client_id: str,
        max_requests: int,
        window_seconds: float,
    ) -> bool:
        """Check if a client is within its rate limit.

        Uses a sliding window. Returns True if the request is allowed,
        False if the client has exceeded the limit.
        """
        now = time.time()
        with self._lock:
            timestamps = self._rate_limits[client_id]
            # Remove expired entries
            cutoff = now - window_seconds
            self._rate_limits[client_id] = [t for t in timestamps if t > cutoff]
            timestamps = self._rate_limits[client_id]
            if len(timestamps) >= max_requests:
                return False
            timestamps.append(now)
            return True

    def authenticate(
        self,
        token: str,
        required_role: Optional[str] = None,
        tenant_id: Optional[str] = None,
        mfa_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Authenticate a token with optional role, tenant, and MFA checks.

        Raises:
            AuthenticationError: If the token is invalid or tenant mismatch.
            AuthorizationError: If the required role is missing.
        """
        try:
            payload = self.validate_token(token)
        except AuthenticationError:
            if self._audit_logger:
                from src.decision_support.audit import AuditEventType

                self._audit_logger.log(
                    event_type=AuditEventType.AUTH_FAILURE,
                    action="authenticate",
                    resource="auth",
                    actor="unknown",
                    tenant_id=tenant_id,
                )
            raise

        # Tenant check
        if tenant_id is not None:
            token_tenant = payload.get("tenant_id")
            if token_tenant is not None and token_tenant != tenant_id:
                raise AuthenticationError(f"Token not valid for tenant {tenant_id}")

        # Role check
        if required_role is not None and not self.authorize(payload, required_role):
            raise AuthorizationError(f"Role '{required_role}' required but not present")

        # MFA check
        if tenant_id and tenant_id in self.mfa_required_tenants:
            if mfa_code is None:
                raise AuthenticationError("MFA code required for this tenant")
            if self._totp_provider is None or not self._totp_provider.verify(mfa_code):
                raise AuthenticationError("Invalid MFA code")

        # Audit logging
        if self._audit_logger:
            from src.decision_support.audit import AuditEventType

            self._audit_logger.log(
                event_type=AuditEventType.AUTH_SUCCESS,
                action="authenticate",
                resource="auth",
                actor=payload.get("sub", "unknown"),
                tenant_id=tenant_id or payload.get("tenant_id"),
            )

        return payload

    def revoke_token(self, token: str, ttl_seconds: int = 3600) -> None:
        """Revoke a token by adding it to the blacklist."""
        self.blacklist.revoke(token, ttl_seconds=ttl_seconds)
