"""FIWARE NGSI-LD context broker integration.

Provides a thin client for the NGSI-LD API (ETSI GS CIM 009) covering
entity CRUD, subscriptions, and context source registrations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

NGSILD_VERSION = "1.6.1"


class NGSILDBroker:
    """Client for a FIWARE NGSI-LD context broker.

    Wraps the NGSI-LD API endpoints under ``/ngsi-ld/v1`` and scopes all
    requests to a tenant via the ``NGSILD-Tenant`` header.
    """

    def __init__(
        self,
        base_url: str,
        tenant: Optional[str] = None,
        timeout: float = 10.0,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.tenant = tenant
        self.timeout = timeout
        self.session = session or requests.Session()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base_url}/ngsi-ld/v1{path}"

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/ld+json",
            "Accept": "application/ld+json",
        }
        if self.tenant:
            headers["NGSILD-Tenant"] = self.tenant
        return headers

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def create_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Create an entity via POST /entities.

        Returns the response body (containing the assigned id) or an
        empty dict when the broker returns no content.
        """
        resp = self.session.post(
            self._url("/entities"),
            json=entity,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        if resp.content:
            return resp.json()
        return {}

    def get_entity(self, entity_id: str) -> Dict[str, Any]:
        """Retrieve an entity by id via GET /entities/{id}."""
        resp = self.session.get(
            self._url(f"/entities/{entity_id}"),
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def update_entity(self, entity_id: str, attributes: Dict[str, Any]) -> None:
        """Update entity attributes via PATCH /entities/{id}/attrs."""
        resp = self.session.patch(
            self._url(f"/entities/{entity_id}/attrs"),
            json=attributes,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity via DELETE /entities/{id}.

        Returns True on success (204), False when the entity does not
        exist (404).
        """
        resp = self.session.delete(
            self._url(f"/entities/{entity_id}"),
            headers=self._headers(),
            timeout=self.timeout,
        )
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return True

    def list_entities(
        self,
        entity_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List entities via GET /entities with optional type filter."""
        params: Dict[str, Any] = {"limit": limit}
        if entity_type:
            params["type"] = entity_type
        resp = self.session.get(
            self._url("/entities"),
            params=params,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def create_subscription(self, subscription: Dict[str, Any]) -> Dict[str, Any]:
        """Create a subscription via POST /subscriptions."""
        resp = self.session.post(
            self._url("/subscriptions"),
            json=subscription,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        if resp.content:
            return resp.json()
        return {}

    def delete_subscription(self, subscription_id: str) -> bool:
        """Delete a subscription via DELETE /subscriptions/{id}.

        Returns True on success (204), False when not found (404).
        """
        resp = self.session.delete(
            self._url(f"/subscriptions/{subscription_id}"),
            headers=self._headers(),
            timeout=self.timeout,
        )
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return True

    # ------------------------------------------------------------------
    # Context source registration
    # ------------------------------------------------------------------

    def register_context(self, registration: Dict[str, Any]) -> Dict[str, Any]:
        """Register a context source via POST /csourceRegistrations."""
        resp = self.session.post(
            self._url("/csourceRegistrations"),
            json=registration,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        if resp.content:
            return resp.json()
        return {}
