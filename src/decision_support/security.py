"""Zero-trust authentication for agricultural IoT API.

Provides JWT validation, role-based access control, and rate limiting.
"""
from __future__ import annotations

import time
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

import jwt


class AuthenticationError(Exception):
    """Raised when authentication fails (invalid/expired token)."""


class AuthorizationError(Exception):
    """Raised when authorization fails (insufficient role)."""


class ZeroTrustAuth:
    """Zero-trust authentication with JWT, RBAC, and rate limiting."""

    def __init__(self, secret: str, algorithm: str = "HS256"):
        self._secret = secret
        self._algorithm = algorithm
        self._rate_limits: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def generate_token(
        self,
        subject: str,
        roles: List[str],
        ttl_seconds: int = 3600,
    ) -> str:
        """Generate a JWT token for the given subject and roles."""
        now = time.time()
        payload = {
            "sub": subject,
            "roles": roles,
            "iat": now,
            "exp": now + ttl_seconds,
        }
        return jwt.encode(payload, self._secret, algorithm=self._algorithm)

    def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate a JWT token and return its payload.

        Raises:
            AuthenticationError: If the token is invalid, expired, or malformed.
        """
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
        """Check if the payload authorizes the required role."""
        return self.has_role(payload, required_role)

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
    ) -> Dict[str, Any]:
        """Authenticate a token and optionally check role authorization.

        Raises:
            AuthenticationError: If the token is invalid.
            AuthorizationError: If the required role is missing.
        """
        payload = self.validate_token(token)
        if required_role is not None and not self.authorize(payload, required_role):
            raise AuthorizationError(
                f"Role '{required_role}' required but not present"
            )
        return payload
