"""Tests for audit logging and API access tracking."""

import json
import threading

from src.decision_support.audit import AuditEventType, AuditLogger


class TestAuditLogging:
    def test_log_creates_event_with_fields(self):
        """log() records an event with the supplied fields."""
        logger = AuditLogger(actor="user-1")
        event = logger.log(
            event_type=AuditEventType.API_REQUEST,
            action="GET",
            resource="/farms",
            tenant_id="t-1",
            source_ip="10.0.0.1",
        )
        assert event.event_type == AuditEventType.API_REQUEST
        assert event.action == "GET"
        assert event.resource == "/farms"
        assert event.actor == "user-1"
        assert event.tenant_id == "t-1"
        assert event.source_ip == "10.0.0.1"
        assert logger.event_count == 1

    def test_event_ids_are_unique(self):
        """Each event gets a distinct id."""
        logger = AuditLogger()
        e1 = logger.log(AuditEventType.USER_ACTION, "update", "/farm/1")
        e2 = logger.log(AuditEventType.USER_ACTION, "update", "/farm/1")
        assert e1.event_id != e2.event_id

    def test_log_api_request_convenience(self):
        """log_api_request maps method/endpoint onto the event."""
        logger = AuditLogger()
        event = logger.log_api_request(
            method="post",
            endpoint="/farms/42/irrigate",
            tenant_id="t-9",
            actor="agronomist-3",
            request_id="req-1",
            status="200",
            latency_ms=12.5,
        )
        assert event.event_type == AuditEventType.API_REQUEST
        assert event.action == "POST"
        assert event.resource == "/farms/42/irrigate"
        assert event.status == "200"
        assert event.latency_ms == 12.5
        assert event.request_id == "req-1"

    def test_default_actor_is_system(self):
        """Events without an explicit actor default to the logger actor."""
        logger = AuditLogger()
        event = logger.log(AuditEventType.AUTH_FAILURE, "login", "/auth")
        assert event.actor == "system"


class TestAuditQueries:
    def test_filter_by_event_type(self):
        """get_events filters by event type."""
        logger = AuditLogger()
        logger.log(AuditEventType.API_REQUEST, "GET", "/a")
        logger.log(AuditEventType.AUTH_FAILURE, "login", "/auth")
        logger.log(AuditEventType.API_REQUEST, "POST", "/b")
        api_events = logger.get_events(event_type=AuditEventType.API_REQUEST)
        assert len(api_events) == 2
        assert all(e.event_type == AuditEventType.API_REQUEST for e in api_events)

    def test_filter_by_tenant(self):
        """get_events filters by tenant id."""
        logger = AuditLogger()
        logger.log(AuditEventType.API_REQUEST, "GET", "/a", tenant_id="t-1")
        logger.log(AuditEventType.API_REQUEST, "GET", "/b", tenant_id="t-2")
        logger.log(AuditEventType.API_REQUEST, "GET", "/c", tenant_id="t-1")
        assert len(logger.get_events(tenant_id="t-1")) == 2
        assert len(logger.get_events(tenant_id="t-2")) == 1

    def test_filter_by_actor(self):
        """get_events filters by actor."""
        logger = AuditLogger()
        logger.log(AuditEventType.USER_ACTION, "update", "/a", actor="alice")
        logger.log(AuditEventType.USER_ACTION, "update", "/b", actor="bob")
        assert len(logger.get_events(actor="alice")) == 1

    def test_request_trace_groups_events(self):
        """get_request_trace returns all events for a request id."""
        logger = AuditLogger()
        logger.log(AuditEventType.API_REQUEST, "GET", "/a", request_id="r-1")
        logger.log(AuditEventType.RECOMMENDATION, "recommend", "/a", request_id="r-1")
        logger.log(AuditEventType.API_REQUEST, "GET", "/b", request_id="r-2")
        trace = logger.get_request_trace("r-1")
        assert len(trace) == 2
        assert all(e.request_id == "r-1" for e in trace)

    def test_limit_returns_most_recent(self):
        """limit returns the latest N events."""
        logger = AuditLogger()
        for i in range(5):
            logger.log(AuditEventType.API_REQUEST, "GET", f"/{i}", timestamp=float(i))
        limited = logger.get_events(limit=2)
        assert len(limited) == 2
        assert [e.resource for e in limited] == ["/3", "/4"]


class TestTamperEvidence:
    def test_chain_verifies_when_intact(self):
        """A freshly logged sequence verifies."""
        logger = AuditLogger()
        for i in range(5):
            logger.log(AuditEventType.API_REQUEST, "GET", f"/r{i}")
        assert logger.verify_chain() is True

    def test_chain_detects_tampering(self):
        """Mutating a recorded event breaks the chain."""
        logger = AuditLogger()
        for i in range(3):
            logger.log(AuditEventType.API_REQUEST, "GET", f"/r{i}")
        # Tamper with the first event's recorded action.
        logger._events[0].action = "DELETE"
        assert logger.verify_chain() is False

    def test_chain_detects_deletion(self):
        """Removing a middle event breaks the chain."""
        logger = AuditLogger()
        for i in range(4):
            logger.log(AuditEventType.API_REQUEST, "GET", f"/r{i}")
        del logger._events[1]
        assert logger.verify_chain() is False


class TestExportAndConcurrency:
    def test_export_json_round_trips(self):
        """export_json produces a parseable array of event dicts."""
        logger = AuditLogger()
        logger.log(AuditEventType.API_REQUEST, "GET", "/a", tenant_id="t-1")
        logger.log(AuditEventType.AUTH_SUCCESS, "login", "/auth", actor="alice")
        data = json.loads(logger.export_json())
        assert len(data) == 2
        assert data[0]["resource"] == "/a"
        assert data[1]["actor"] == "alice"
        assert all("event_id" in e for e in data)

    def test_clear_resets_state(self):
        """clear() empties events and resets the chain."""
        logger = AuditLogger()
        logger.log(AuditEventType.API_REQUEST, "GET", "/a")
        logger.clear()
        assert logger.event_count == 0
        assert logger.verify_chain() is True

    def test_thread_safe_concurrent_logging(self):
        """Concurrent logging from many threads records all events."""
        logger = AuditLogger()

        def worker(n):
            for i in range(10):
                logger.log(
                    AuditEventType.API_REQUEST,
                    "GET",
                    f"/t{n}/{i}",
                    request_id=f"r-{n}-{i}",
                )

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert logger.event_count == 80
        assert logger.verify_chain() is True
