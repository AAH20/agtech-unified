"""Feature store for persistence, versioning, and retrieval of computed features."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

__all__ = ["FeatureRecord", "FeatureVersion", "FeatureStore"]


@dataclass
class FeatureRecord:
    """A single computed feature value."""

    feature_name: str
    value: float
    entity_id: str
    tenant_id: Optional[str] = None
    version: str = "1.0"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "value": self.value,
            "entity_id": self.entity_id,
            "tenant_id": self.tenant_id,
            "version": self.version,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeatureRecord:
        """Create from dictionary."""
        return cls(
            feature_name=data["feature_name"],
            value=data["value"],
            entity_id=data["entity_id"],
            tenant_id=data.get("tenant_id"),
            version=data.get("version", "1.0"),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class FeatureVersion:
    """Version metadata for a feature."""

    version: str
    created_at: float
    feature_count: int = 0
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "feature_count": self.feature_count,
            "description": self.description,
        }


class FeatureStore:
    """In-memory feature store with persistence and versioning.

    Supports:
    - Put/get individual features
    - Batch operations
    - Versioning per feature per entity
    - Tenant isolation
    - JSON persistence
    """

    def __init__(self):
        # Nested dict: entity_id -> feature_name -> version -> FeatureRecord
        self._store: Dict[str, Dict[str, Dict[str, FeatureRecord]]] = {}
        # Version metadata: (entity_id, feature_name) -> List[FeatureVersion]
        self._versions: Dict[str, Dict[str, List[FeatureVersion]]] = {}
        self._lock = threading.Lock()

    def put(
        self,
        feature_name: str,
        value: float,
        entity_id: str,
        tenant_id: Optional[str] = None,
        version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeatureRecord:
        """Store a feature value.

        Args:
            feature_name: Name of the feature (e.g., "gdd", "vpd").
            value: Computed feature value.
            entity_id: Entity identifier (e.g., farm ID).
            tenant_id: Optional tenant for isolation.
            version: Version string for this feature value.
            metadata: Optional metadata dict.

        Returns:
            The stored FeatureRecord.
        """
        record = FeatureRecord(
            feature_name=feature_name,
            value=value,
            entity_id=entity_id,
            tenant_id=tenant_id,
            version=version,
            metadata=metadata or {},
        )
        with self._lock:
            if entity_id not in self._store:
                self._store[entity_id] = {}
            if feature_name not in self._store[entity_id]:
                self._store[entity_id][feature_name] = {}
            self._store[entity_id][feature_name][version] = record

            # Track version metadata
            if entity_id not in self._versions:
                self._versions[entity_id] = {}
            if feature_name not in self._versions[entity_id]:
                self._versions[entity_id][feature_name] = []
            version_meta = FeatureVersion(
                version=version,
                created_at=record.timestamp,
                feature_count=len(self._store[entity_id][feature_name]),
            )
            self._versions[entity_id][feature_name].append(version_meta)

        logger.debug(
            "Stored feature %s=%s for entity %s (version %s)",
            feature_name,
            value,
            entity_id,
            version,
        )
        return record

    def get(
        self,
        feature_name: str,
        entity_id: str,
        tenant_id: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Optional[float]:
        """Retrieve a feature value.

        Args:
            feature_name: Name of the feature.
            entity_id: Entity identifier.
            tenant_id: Optional tenant for isolation check.
            version: Specific version to retrieve. If None, returns latest.

        Returns:
            Feature value or None if not found.
        """
        with self._lock:
            if entity_id not in self._store:
                return None
            if feature_name not in self._store[entity_id]:
                return None

            versions = self._store[entity_id][feature_name]
            if version is not None:
                record = versions.get(version)
            else:
                # Return latest version (sorted by version string)
                sorted_versions = sorted(versions.keys())
                if not sorted_versions:
                    return None
                record = versions[sorted_versions[-1]]

            if record is None:
                return None

            # Tenant isolation check
            if tenant_id is not None and record.tenant_id is not None:
                if record.tenant_id != tenant_id:
                    return None

            return record.value

    def get_record(
        self,
        feature_name: str,
        entity_id: str,
        tenant_id: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Optional[FeatureRecord]:
        """Retrieve the full FeatureRecord.

        Args:
            feature_name: Name of the feature.
            entity_id: Entity identifier.
            tenant_id: Optional tenant for isolation check.
            version: Specific version to retrieve. If None, returns latest.

        Returns:
            FeatureRecord or None if not found.
        """
        with self._lock:
            if entity_id not in self._store:
                return None
            if feature_name not in self._store[entity_id]:
                return None

            versions = self._store[entity_id][feature_name]
            if version is not None:
                record = versions.get(version)
            else:
                sorted_versions = sorted(versions.keys())
                if not sorted_versions:
                    return None
                record = versions[sorted_versions[-1]]

            if record is None:
                return None

            if tenant_id is not None and record.tenant_id is not None:
                if record.tenant_id != tenant_id:
                    return None

            return record

    def get_all(self, entity_id: str, tenant_id: Optional[str] = None) -> Dict[str, float]:
        """Get all latest feature values for an entity.

        Args:
            entity_id: Entity identifier.
            tenant_id: Optional tenant for isolation.

        Returns:
            Dict mapping feature names to their latest values.
        """
        with self._lock:
            if entity_id not in self._store:
                return {}
            result = {}
            for feature_name, versions in self._store[entity_id].items():
                sorted_versions = sorted(versions.keys())
                if not sorted_versions:
                    continue
                record = versions[sorted_versions[-1]]
                if tenant_id is not None and record.tenant_id is not None:
                    if record.tenant_id != tenant_id:
                        continue
                result[feature_name] = record.value
            return result

    def put_batch(
        self,
        features: Dict[str, float],
        entity_id: str,
        tenant_id: Optional[str] = None,
        version: str = "1.0",
    ) -> List[FeatureRecord]:
        """Store multiple features at once.

        Args:
            features: Dict mapping feature names to values.
            entity_id: Entity identifier.
            tenant_id: Optional tenant for isolation.
            version: Version string for all features.

        Returns:
            List of stored FeatureRecords.
        """
        records = []
        for name, value in features.items():
            record = self.put(name, value, entity_id, tenant_id=tenant_id, version=version)
            records.append(record)
        return records

    def get_batch(
        self,
        feature_names: List[str],
        entity_id: str,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Retrieve multiple features at once.

        Args:
            feature_names: List of feature names to retrieve.
            entity_id: Entity identifier.
            tenant_id: Optional tenant for isolation.

        Returns:
            Dict mapping found feature names to values.
        """
        result = {}
        for name in feature_names:
            val = self.get(name, entity_id, tenant_id=tenant_id)
            if val is not None:
                result[name] = val
        return result

    def delete(
        self,
        feature_name: str,
        entity_id: str,
        version: Optional[str] = None,
    ) -> bool:
        """Delete a feature.

        Args:
            feature_name: Name of the feature.
            entity_id: Entity identifier.
            version: Specific version to delete. If None, deletes all versions.

        Returns:
            True if anything was deleted.
        """
        with self._lock:
            if entity_id not in self._store:
                return False
            if feature_name not in self._store[entity_id]:
                return False

            if version is not None:
                if version in self._store[entity_id][feature_name]:
                    del self._store[entity_id][feature_name][version]
                    return True
                return False
            else:
                del self._store[entity_id][feature_name]
                return True

    def clear(self) -> None:
        """Clear all stored features."""
        with self._lock:
            self._store.clear()
            self._versions.clear()

    def count(self) -> int:
        """Count total feature records across all entities."""
        with self._lock:
            total = 0
            for entity_features in self._store.values():
                for versions in entity_features.values():
                    total += len(versions)
            return total

    def list_features(
        self,
        entity_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> List[str]:
        """List feature names.

        Args:
            entity_id: Optional entity filter.
            tenant_id: Optional tenant filter.

        Returns:
            List of feature names.
        """
        with self._lock:
            names = set()
            entities = [entity_id] if entity_id else list(self._store.keys())
            for eid in entities:
                if eid not in self._store:
                    continue
                for feature_name, versions in self._store[eid].items():
                    if tenant_id is not None:
                        # Check if any version belongs to this tenant
                        has_tenant = any(v.tenant_id == tenant_id for v in versions.values())
                        if not has_tenant:
                            continue
                    names.add(feature_name)
            return sorted(names)

    def get_entities(self, tenant_id: Optional[str] = None) -> List[str]:
        """List entity IDs.

        Args:
            tenant_id: Optional tenant filter.

        Returns:
            List of entity IDs.
        """
        with self._lock:
            if tenant_id is None:
                return sorted(self._store.keys())
            entities = set()
            for eid, features in self._store.items():
                for versions in features.values():
                    for record in versions.values():
                        if record.tenant_id == tenant_id:
                            entities.add(eid)
                            break
                    else:
                        continue
                    break
            return sorted(entities)

    def list_versions(
        self,
        feature_name: str,
        entity_id: str,
    ) -> List[str]:
        """List versions for a feature.

        Args:
            feature_name: Name of the feature.
            entity_id: Entity identifier.

        Returns:
            List of version strings.
        """
        with self._lock:
            if entity_id not in self._store:
                return []
            if feature_name not in self._store[entity_id]:
                return []
            return sorted(self._store[entity_id][feature_name].keys())

    def get_version_history(
        self,
        feature_name: str,
        entity_id: str,
    ) -> List[Dict[str, Any]]:
        """Get version history for a feature.

        Args:
            feature_name: Name of the feature.
            entity_id: Entity identifier.

        Returns:
            List of dicts with version info.
        """
        with self._lock:
            if entity_id not in self._store:
                return []
            if feature_name not in self._store[entity_id]:
                return []
            history = []
            for ver, record in sorted(self._store[entity_id][feature_name].items()):
                history.append(
                    {
                        "version": ver,
                        "value": record.value,
                        "timestamp": record.timestamp,
                        "tenant_id": record.tenant_id,
                    }
                )
            return history

    def rollback(
        self,
        feature_name: str,
        entity_id: str,
        to_version: str,
    ) -> bool:
        """Rollback a feature to a previous version.

        Args:
            feature_name: Name of the feature.
            entity_id: Entity identifier.
            to_version: Version to rollback to.

        Returns:
            True if rollback succeeded.
        """
        with self._lock:
            if entity_id not in self._store:
                raise ValueError(f"Entity '{entity_id}' not found")
            if feature_name not in self._store[entity_id]:
                raise ValueError(f"Feature '{feature_name}' not found for entity '{entity_id}'")
            versions = self._store[entity_id][feature_name]
            if to_version not in versions:
                raise ValueError(f"Version '{to_version}' not found for feature '{feature_name}'")
            # Remove all versions after the rollback point
            to_remove = [v for v in versions.keys() if v > to_version]
            for v in to_remove:
                del versions[v]
            return True

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist feature store to JSON file.

        Args:
            filepath: Path to save file.
        """
        with self._lock:
            data = {
                "store": {},
                "versions": {},
            }
            for eid, features in self._store.items():
                data["store"][eid] = {}
                for fname, versions in features.items():
                    data["store"][eid][fname] = {}
                    for ver, record in versions.items():
                        data["store"][eid][fname][ver] = record.to_dict()
            for eid, features in self._versions.items():
                data["versions"][eid] = {}
                for fname, vers_list in features.items():
                    data["versions"][eid][fname] = [v.to_dict() for v in vers_list]

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved feature store to %s", filepath)

    def load(self, filepath: Union[str, Path]) -> None:
        """Load feature store from JSON file.

        Args:
            filepath: Path to load file.
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning("Feature store file not found: %s", filepath)
            return

        with open(path) as f:
            data = json.load(f)

        with self._lock:
            self._store.clear()
            self._versions.clear()

            for eid, features in data.get("store", {}).items():
                self._store[eid] = {}
                for fname, versions in features.items():
                    self._store[eid][fname] = {}
                    for ver, record_data in versions.items():
                        self._store[eid][fname][ver] = FeatureRecord.from_dict(record_data)

            for eid, features in data.get("versions", {}).items():
                self._versions[eid] = {}
                for fname, vers_list in features.items():
                    self._versions[eid][fname] = [
                        FeatureVersion(
                            version=v["version"],
                            created_at=v["created_at"],
                            feature_count=v.get("feature_count", 0),
                            description=v.get("description"),
                        )
                        for v in vers_list
                    ]

        logger.info("Loaded feature store from %s", filepath)
