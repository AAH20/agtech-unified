"""Tests for audit persistence, date filtering, and export formats."""

import json
import os
import tempfile
import time

from src.decision_support.audit import AuditEventType, AuditLogger


class TestAuditPersistence:
    def test_save_to_jsonl(self):
        """AuditLogger can save events to a JSONL file."""
        logger = AuditLogger()
        logger.log(AuditEventType.API_REQUEST, "GET", "/test", tenant_id="t-1")
        logger.log(AuditEventType.AUTH_SUCCESS, "login", "/auth", actor="user-1")

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            logger.save_to_file(path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2
            event = json.loads(lines[0])
            assert event["event_type"] == "api_request"
            assert event["resource"] == "/test"
        finally:
            os.unlink(path)

    def test_load_from_jsonl(self):
        """AuditLogger can load events from a JSONL file."""
        logger = AuditLogger()
        logger.log(AuditEventType.API_REQUEST, "GET", "/test", tenant_id="t-1")

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            logger.save_to_file(path)
            new_logger = AuditLogger()
            new_logger.load_from_file(path)
            assert new_logger.event_count == 1
            events = new_logger.get_events()
            assert events[0].resource == "/test"
        finally:
            os.unlink(path)

    def test_persistence_round_trip(self):
        """Events survive a save/load round trip."""
        logger = AuditLogger()
        logger.log(AuditEventType.API_REQUEST, "GET", "/a", tenant_id="t-1", actor="user-1")
        logger.log(AuditEventType.AUTH_FAILURE, "login", "/auth", actor="user-2")

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            logger.save_to_file(path)
            loaded = AuditLogger()
            loaded.load_from_file(path)
            assert loaded.event_count == 2
            assert loaded.verify_chain() is True
        finally:
            os.unlink(path)


class TestAuditDateFiltering:
    def test_filter_by_date_range(self):
        """get_events supports date range filtering."""
        logger = AuditLogger()
        now = time.time()
        logger.log(AuditEventType.API_REQUEST, "GET", "/a", timestamp=now - 100)
        logger.log(AuditEventType.API_REQUEST, "GET", "/b", timestamp=now - 50)
        logger.log(AuditEventType.API_REQUEST, "GET", "/c", timestamp=now)

        events = logger.get_events(start_time=now - 60, end_time=now + 10)
        assert len(events) == 2
        resources = {e.resource for e in events}
        assert resources == {"/b", "/c"}

    def test_filter_by_start_time_only(self):
        """get_events supports start_time only."""
        logger = AuditLogger()
        now = time.time()
        logger.log(AuditEventType.API_REQUEST, "GET", "/a", timestamp=now - 100)
        logger.log(AuditEventType.API_REQUEST, "GET", "/b", timestamp=now)

        events = logger.get_events(start_time=now - 50)
        assert len(events) == 1
        assert events[0].resource == "/b"

    def test_filter_by_end_time_only(self):
        """get_events supports end_time only."""
        logger = AuditLogger()
        now = time.time()
        logger.log(AuditEventType.API_REQUEST, "GET", "/a", timestamp=now - 100)
        logger.log(AuditEventType.API_REQUEST, "GET", "/b", timestamp=now)

        events = logger.get_events(end_time=now - 50)
        assert len(events) == 1
        assert events[0].resource == "/a"


class TestAuditExportFormats:
    def test_export_csv(self):
        """AuditLogger can export events as CSV."""
        logger = AuditLogger()
        logger.log(AuditEventType.API_REQUEST, "GET", "/test", tenant_id="t-1", actor="user-1")
        csv_output = logger.export_csv()
        lines = csv_output.strip().split("\n")
        assert len(lines) == 2  # header + 1 event
        assert "event_id" in lines[0]
        assert "timestamp" in lines[0]
        assert "/test" in lines[1]

    def test_export_json(self):
        """AuditLogger can export events as JSON."""
        logger = AuditLogger()
        logger.log(AuditEventType.API_REQUEST, "GET", "/test")
        json_output = logger.export_json()
        data = json.loads(json_output)
        assert len(data) == 1
        assert data[0]["resource"] == "/test"


class TestAuditArchival:
    def test_archive_old_events(self):
        """Old events can be archived to a file."""
        logger = AuditLogger()
        now = time.time()
        logger.log(AuditEventType.API_REQUEST, "GET", "/old", timestamp=now - 10000)
        logger.log(AuditEventType.API_REQUEST, "GET", "/new", timestamp=now)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            logger.archive_old_events(path, max_age_seconds=1000)
            # Old event should be archived, new one remains
            assert logger.event_count == 1
            events = logger.get_events()
            assert events[0].resource == "/new"
            # Archived file should have the old event
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            assert json.loads(lines[0])["resource"] == "/old"
        finally:
            os.unlink(path)

    def test_chain_verification_after_archive(self):
        """Chain verification still works after archival."""
        logger = AuditLogger()
        now = time.time()
        logger.log(AuditEventType.API_REQUEST, "GET", "/old", timestamp=now - 10000)
        logger.log(AuditEventType.API_REQUEST, "GET", "/new", timestamp=now)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            logger.archive_old_events(path, max_age_seconds=1000)
            assert logger.verify_chain() is True
        finally:
            os.unlink(path)
