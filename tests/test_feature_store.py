"""Tests for feature store: persistence, versioning, retrieval of computed features."""

from __future__ import annotations

import pytest

from src.decision_support.feature_store import (
    FeatureRecord,
    FeatureStore,
    FeatureVersion,
)


class TestFeatureRecord:
    """FeatureRecord dataclass tests."""

    def test_create_record(self):
        rec = FeatureRecord(
            feature_name="gdd",
            value=12.5,
            entity_id="farm-1",
            tenant_id="t1",
            version="1.0",
            timestamp=1000.0,
        )
        assert rec.feature_name == "gdd"
        assert rec.value == 12.5
        assert rec.entity_id == "farm-1"
        assert rec.tenant_id == "t1"
        assert rec.version == "1.0"
        assert rec.timestamp == 1000.0

    def test_create_record_defaults(self):
        rec = FeatureRecord(
            feature_name="vpd",
            value=1.2,
            entity_id="farm-1",
        )
        assert rec.tenant_id is None
        assert rec.version == "1.0"
        assert rec.timestamp is not None


class TestFeatureVersion:
    """FeatureVersion dataclass tests."""

    def test_create_version(self):
        fv = FeatureVersion(
            version="1.0",
            created_at=1000.0,
            feature_count=3,
        )
        assert fv.version == "1.0"
        assert fv.created_at == 1000.0
        assert fv.feature_count == 3

    def test_version_comparison(self):
        v1 = FeatureVersion(version="1.0", created_at=100.0, feature_count=1)
        v2 = FeatureVersion(version="2.0", created_at=200.0, feature_count=1)
        assert v1.version != v2.version


class TestFeatureStore:
    """FeatureStore core tests."""

    def test_put_and_get(self):
        store = FeatureStore()
        store.put("gdd", 12.5, entity_id="farm-1")
        val = store.get("gdd", entity_id="farm-1")
        assert val == 12.5

    def test_get_nonexistent_returns_none(self):
        store = FeatureStore()
        assert store.get("nonexistent", entity_id="farm-1") is None

    def test_get_latest_by_entity(self):
        store = FeatureStore()
        store.put("gdd", 10.0, entity_id="farm-1")
        store.put("gdd", 20.0, entity_id="farm-1")
        val = store.get("gdd", entity_id="farm-1")
        assert val == 20.0

    def test_put_with_version(self):
        store = FeatureStore()
        store.put("vpd", 1.5, entity_id="farm-1", version="2.0")
        val = store.get("vpd", entity_id="farm-1", version="2.0")
        assert val == 1.5

    def test_get_specific_version(self):
        store = FeatureStore()
        store.put("vpd", 1.0, entity_id="farm-1", version="1.0")
        store.put("vpd", 2.0, entity_id="farm-1", version="2.0")
        assert store.get("vpd", entity_id="farm-1", version="1.0") == 1.0
        assert store.get("vpd", entity_id="farm-1", version="2.0") == 2.0

    def test_get_all_features_for_entity(self):
        store = FeatureStore()
        store.put("gdd", 10.0, entity_id="farm-1")
        store.put("vpd", 1.5, entity_id="farm-1")
        store.put("trend", 0.3, entity_id="farm-1")
        features = store.get_all("farm-1")
        assert features["gdd"] == 10.0
        assert features["vpd"] == 1.5
        assert features["trend"] == 0.3

    def test_get_all_features_empty(self):
        store = FeatureStore()
        assert store.get_all("nonexistent") == {}

    def test_entity_isolation(self):
        store = FeatureStore()
        store.put("gdd", 10.0, entity_id="farm-1", tenant_id="t1")
        store.put("gdd", 20.0, entity_id="farm-2", tenant_id="t2")
        assert store.get("gdd", entity_id="farm-1", tenant_id="t1") == 10.0
        assert store.get("gdd", entity_id="farm-2", tenant_id="t2") == 20.0

    def test_tenant_isolation_enforced(self):
        store = FeatureStore()
        store.put("gdd", 10.0, entity_id="farm-1", tenant_id="t1")
        # Accessing with wrong tenant should not find it
        assert store.get("gdd", entity_id="farm-1", tenant_id="t2") is None

    def test_list_features(self):
        store = FeatureStore()
        store.put("gdd", 10.0, entity_id="farm-1")
        store.put("vpd", 1.5, entity_id="farm-1")
        store.put("trend", 0.3, entity_id="farm-2")
        features = store.list_features()
        assert "gdd" in features
        assert "vpd" in features
        assert "trend" in features

    def test_list_features_by_entity(self):
        store = FeatureStore()
        store.put("gdd", 10.0, entity_id="farm-1")
        store.put("vpd", 1.5, entity_id="farm-2")
        features = store.list_features(entity_id="farm-1")
        assert "gdd" in features
        assert "vpd" not in features

    def test_delete_feature(self):
        store = FeatureStore()
        store.put("gdd", 10.0, entity_id="farm-1")
        store.delete("gdd", entity_id="farm-1")
        assert store.get("gdd", entity_id="farm-1") is None

    def test_delete_nonexistent_no_error(self):
        store = FeatureStore()
        store.delete("nonexistent", entity_id="farm-1")  # Should not raise

    def test_clear_all(self):
        store = FeatureStore()
        store.put("gdd", 10.0, entity_id="farm-1")
        store.put("vpd", 1.5, entity_id="farm-2")
        store.clear()
        assert store.get_all("farm-1") == {}
        assert store.get_all("farm-2") == {}

    def test_count_features(self):
        store = FeatureStore()
        store.put("gdd", 10.0, entity_id="farm-1")
        store.put("vpd", 1.5, entity_id="farm-1")
        store.put("trend", 0.3, entity_id="farm-2")
        assert store.count() == 3

    def test_count_empty(self):
        store = FeatureStore()
        assert store.count() == 0


class TestFeatureStorePersistence:
    """FeatureStore persistence tests."""

    def test_save_and_load(self, tmp_path):
        store = FeatureStore()
        store.put("gdd", 10.0, entity_id="farm-1", tenant_id="t1")
        store.put("vpd", 1.5, entity_id="farm-2", tenant_id="t2")
        filepath = str(tmp_path / "features.json")
        store.save(filepath)

        new_store = FeatureStore()
        new_store.load(filepath)
        assert new_store.get("gdd", entity_id="farm-1", tenant_id="t1") == 10.0
        assert new_store.get("vpd", entity_id="farm-2", tenant_id="t2") == 1.5

    def test_load_nonexistent_file(self, tmp_path):
        store = FeatureStore()
        store.load(str(tmp_path / "nonexistent.json"))  # Should not raise
        assert store.count() == 0

    def test_save_creates_file(self, tmp_path):
        store = FeatureStore()
        store.put("gdd", 10.0, entity_id="farm-1")
        filepath = str(tmp_path / "features.json")
        store.save(filepath)
        import os

        assert os.path.exists(filepath)


class TestFeatureStoreVersioning:
    """FeatureStore versioning tests."""

    def test_put_creates_version(self):
        store = FeatureStore()
        store.put("gdd", 10.0, entity_id="farm-1", version="1.0")
        versions = store.list_versions("gdd", entity_id="farm-1")
        assert "1.0" in versions

    def test_multiple_versions(self):
        store = FeatureStore()
        store.put("gdd", 10.0, entity_id="farm-1", version="1.0")
        store.put("gdd", 20.0, entity_id="farm-1", version="2.0")
        versions = store.list_versions("gdd", entity_id="farm-1")
        assert len(versions) == 2
        assert "1.0" in versions
        assert "2.0" in versions

    def test_get_version_history(self):
        store = FeatureStore()
        store.put("gdd", 10.0, entity_id="farm-1", version="1.0")
        store.put("gdd", 20.0, entity_id="farm-1", version="2.0")
        history = store.get_version_history("gdd", entity_id="farm-1")
        assert len(history) == 2
        values = [h["value"] for h in history]
        assert 10.0 in values
        assert 20.0 in values

    def test_rollback_to_version(self):
        store = FeatureStore()
        store.put("gdd", 10.0, entity_id="farm-1", version="1.0")
        store.put("gdd", 20.0, entity_id="farm-1", version="2.0")
        store.rollback("gdd", entity_id="farm-1", to_version="1.0")
        val = store.get("gdd", entity_id="farm-1")
        assert val == 10.0

    def test_rollback_nonexistent_version_raises(self):
        store = FeatureStore()
        store.put("gdd", 10.0, entity_id="farm-1", version="1.0")
        with pytest.raises(ValueError):
            store.rollback("gdd", entity_id="farm-1", to_version="99.0")


class TestFeatureStoreBatchOperations:
    """FeatureStore batch operations tests."""

    def test_put_batch(self):
        store = FeatureStore()
        features = {"gdd": 10.0, "vpd": 1.5, "trend": 0.3}
        store.put_batch(features, entity_id="farm-1")
        assert store.get("gdd", entity_id="farm-1") == 10.0
        assert store.get("vpd", entity_id="farm-1") == 1.5
        assert store.get("trend", entity_id="farm-1") == 0.3

    def test_get_batch(self):
        store = FeatureStore()
        store.put("gdd", 10.0, entity_id="farm-1")
        store.put("vpd", 1.5, entity_id="farm-1")
        store.put("trend", 0.3, entity_id="farm-1")
        result = store.get_batch(["gdd", "vpd"], entity_id="farm-1")
        assert result == {"gdd": 10.0, "vpd": 1.5}

    def test_get_batch_with_missing(self):
        store = FeatureStore()
        store.put("gdd", 10.0, entity_id="farm-1")
        result = store.get_batch(["gdd", "nonexistent"], entity_id="farm-1")
        assert result == {"gdd": 10.0}

    def test_get_entities(self):
        store = FeatureStore()
        store.put("gdd", 10.0, entity_id="farm-1")
        store.put("vpd", 1.5, entity_id="farm-2")
        store.put("trend", 0.3, entity_id="farm-3")
        entities = store.get_entities()
        assert "farm-1" in entities
        assert "farm-2" in entities
        assert "farm-3" in entities

    def test_get_entities_by_tenant(self):
        store = FeatureStore()
        store.put("gdd", 10.0, entity_id="farm-1", tenant_id="t1")
        store.put("vpd", 1.5, entity_id="farm-2", tenant_id="t2")
        t1_entities = store.get_entities(tenant_id="t1")
        assert "farm-1" in t1_entities
        assert "farm-2" not in t1_entities


class TestFeatureStoreIntegrationWithFeatureEngine:
    """Integration tests between FeatureStore and FeatureEngine."""

    def test_store_computed_features(self):
        from src.decision_support.ml_models import FeatureEngine

        store = FeatureStore()
        engine = FeatureEngine()
        temps = [15.0, 20.0, 25.0, 22.0, 18.0]
        gdd = engine.compute_growing_degree_days(temps)
        store.put("gdd", gdd, entity_id="farm-1")
        assert store.get("gdd", entity_id="farm-1") == gdd

    def test_store_vpd(self):
        from src.decision_support.ml_models import FeatureEngine

        store = FeatureStore()
        engine = FeatureEngine()
        vpd = engine.compute_vpd(30.0, 60.0)
        store.put("vpd", vpd, entity_id="farm-1")
        assert store.get("vpd", entity_id="farm-1") == vpd

    def test_store_trend(self):
        from src.decision_support.ml_models import FeatureEngine

        store = FeatureStore()
        engine = FeatureEngine()
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        trend = engine.compute_trend(values)
        store.put("trend", trend, entity_id="farm-1")
        assert store.get("trend", entity_id="farm-1") == trend
