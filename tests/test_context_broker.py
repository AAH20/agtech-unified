"""Tests for FIWARE NGSI-LD context broker integration."""

from unittest.mock import MagicMock

import pytest

from src.iot.context_broker import NGSILDBroker


def make_response(status_code=200, json_data=None):
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.ok = 200 <= status_code < 300
    resp.content = b"{}" if json_data is not None else b""
    return resp


@pytest.fixture
def mock_session():
    return MagicMock()


@pytest.fixture
def broker(mock_session):
    return NGSILDBroker(
        base_url="http://broker.example:1026",
        tenant="agtech",
        session=mock_session,
    )


class TestEntityCrud:
    """Entity create/read/update/delete against the NGSI-LD API."""

    def test_create_entity_posts_to_entities_endpoint(self, broker, mock_session):
        """create_entity POSTs the entity JSON to /ngsi-ld/v1/entities."""
        mock_session.post.return_value = make_response(201, {"id": "urn:1"})
        result = broker.create_entity({"id": "urn:1", "type": "Crop"})

        args, kwargs = mock_session.post.call_args
        assert args[0] == "http://broker.example:1026/ngsi-ld/v1/entities"
        assert kwargs["json"] == {"id": "urn:1", "type": "Crop"}
        assert result["id"] == "urn:1"

    def test_create_entity_sends_tenant_header(self, broker, mock_session):
        """Tenant (NGSILD-Tenant header) is sent so data is scoped per tenant."""
        mock_session.post.return_value = make_response(201, {})
        broker.create_entity({"id": "urn:2", "type": "Soil"})

        headers = mock_session.post.call_args.kwargs["headers"]
        assert headers["NGSILD-Tenant"] == "agtech"

    def test_get_entity_fetches_by_id(self, broker, mock_session):
        """get_entity GETs /entities/{id} and returns the parsed body."""
        mock_session.get.return_value = make_response(200, {"id": "urn:1", "type": "Crop"})
        entity = broker.get_entity("urn:1")

        args, _ = mock_session.get.call_args
        assert args[0] == "http://broker.example:1026/ngsi-ld/v1/entities/urn:1"
        assert entity["type"] == "Crop"

    def test_update_entity_patches_attributes(self, broker, mock_session):
        """update_entity PATCHes the entity's attrs endpoint."""
        mock_session.patch.return_value = make_response(204)
        broker.update_entity("urn:1", {"height": {"type": "Property", "value": 1.5}})

        args, kwargs = mock_session.patch.call_args
        assert args[0] == "http://broker.example:1026/ngsi-ld/v1/entities/urn:1/attrs"
        assert kwargs["json"]["height"]["value"] == 1.5

    def test_delete_entity_true_on_success(self, broker, mock_session):
        """delete_entity returns True when the broker responds 204."""
        mock_session.delete.return_value = make_response(204)
        assert broker.delete_entity("urn:1") is True

    def test_delete_entity_false_on_404(self, broker, mock_session):
        """delete_entity returns False when the entity does not exist."""
        mock_session.delete.return_value = make_response(404)
        assert broker.delete_entity("urn:missing") is False

    def test_list_entities_filters_by_type(self, broker, mock_session):
        """list_entities passes the type query parameter."""
        mock_session.get.return_value = make_response(200, [{"id": "urn:1"}])
        entities = broker.list_entities(entity_type="Crop", limit=5)

        _, kwargs = mock_session.get.call_args
        assert kwargs["params"]["type"] == "Crop"
        assert kwargs["params"]["limit"] == 5
        assert len(entities) == 1


class TestSubscriptions:
    """Subscription management against the NGSI-LD API."""

    def test_create_subscription_posts_to_subscriptions(self, broker, mock_session):
        """create_subscription POSTs to /ngsi-ld/v1/subscriptions."""
        mock_session.post.return_value = make_response(201, {"id": "urn:sub:1"})
        sub = {
            "type": "Subscription",
            "entities": [{"type": "Crop"}],
            "notification": {"endpoint": {"uri": "http://consumer/hook"}},
        }
        result = broker.create_subscription(sub)

        args, kwargs = mock_session.post.call_args
        assert args[0] == "http://broker.example:1026/ngsi-ld/v1/subscriptions"
        assert kwargs["json"]["type"] == "Subscription"
        assert result["id"] == "urn:sub:1"

    def test_delete_subscription_true_on_success(self, broker, mock_session):
        """delete_subscription returns True on 204."""
        mock_session.delete.return_value = make_response(204)
        assert broker.delete_subscription("urn:sub:1") is True


class TestContextRegistration:
    """Context source registration for NGSI-LD @context resolution."""

    def test_register_context_posts_registration(self, broker, mock_session):
        """register_context POSTs a ContextSourceRegistration."""
        mock_session.post.return_value = make_response(201, {"id": "urn:reg:1"})
        reg = {
            "type": "ContextSourceRegistration",
            "information": [{"entities": [{"type": "Crop"}]}],
            "endpoint": "http://models.example/contexts",
        }
        result = broker.register_context(reg)

        args, kwargs = mock_session.post.call_args
        assert args[0] == ("http://broker.example:1026/ngsi-ld/v1/csourceRegistrations")
        assert kwargs["json"]["type"] == "ContextSourceRegistration"
        assert result["id"] == "urn:reg:1"
