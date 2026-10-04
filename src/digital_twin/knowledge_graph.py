"""Agricultural knowledge graph with entity management and queries.

Neo4j-compatible interface backed by in-memory graph storage.
Provides entity CRUD, relationship mapping, graph queries, and
serialization for agricultural knowledge representation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """A knowledge graph entity (node)."""
    id: str
    label: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """A knowledge graph relationship (edge)."""
    source: str
    target: str
    rel_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


class AgriKnowledgeGraph:
    """Agricultural knowledge graph with entity and relationship management.

    Provides a Neo4j-like interface for managing agricultural entities
    (fields, crops, sensors, etc.) and their relationships.
    """

    def __init__(self):
        self._entities: Dict[str, Entity] = {}
        self._relationships: List[Relationship] = []
        self._adj: Dict[str, List[Relationship]] = {}
        self._radj: Dict[str, List[Relationship]] = {}

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph."""
        if entity.id in self._entities:
            raise ValueError(f"Entity '{entity.id}' already exists")
        self._entities[entity.id] = entity
        self._adj[entity.id] = []
        self._radj[entity.id] = []

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID."""
        return self._entities.get(entity_id)

    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity and all its relationships."""
        if entity_id not in self._entities:
            return
        # Remove all relationships involving this entity
        self._relationships = [
            r for r in self._relationships
            if r.source != entity_id and r.target != entity_id
        ]
        # Clean adjacency lists
        self._adj.pop(entity_id, None)
        self._radj.pop(entity_id, None)
        for adj_list in self._adj.values():
            adj_list[:] = [r for r in adj_list if r.target != entity_id]
        for radj_list in self._radj.values():
            radj_list[:] = [r for r in radj_list if r.source != entity_id]
        del self._entities[entity_id]

    def entity_count(self) -> int:
        """Return the number of entities."""
        return len(self._entities)

    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship between two entities."""
        if relationship.source not in self._entities:
            raise ValueError(f"Source entity '{relationship.source}' not found")
        if relationship.target not in self._entities:
            raise ValueError(f"Target entity '{relationship.target}' not found")
        self._relationships.append(relationship)
        self._adj[relationship.source].append(relationship)
        self._radj[relationship.target].append(relationship)

    def get_relationships(self, entity_id: str, direction: str = "outgoing") -> List[Relationship]:
        """Get relationships for an entity.

        Args:
            entity_id: The entity ID.
            direction: 'outgoing', 'incoming', or 'both'.
        """
        if entity_id not in self._entities:
            return []
        if direction == "outgoing":
            return list(self._adj.get(entity_id, []))
        elif direction == "incoming":
            return list(self._radj.get(entity_id, []))
        else:  # both
            return list(self._adj.get(entity_id, [])) + list(self._radj.get(entity_id, []))

    def relationship_count(self) -> int:
        """Return the number of relationships."""
        return len(self._relationships)

    def query(
        self,
        entity_type: Optional[str] = None,
        label: Optional[str] = None,
        relationship: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> List[Entity]:
        """Query entities by type, label, relationship, or properties.

        All specified filters are combined with AND logic.
        """
        results: List[Entity] = []

        for entity in self._entities.values():
            if entity_type is not None and entity.entity_type != entity_type:
                continue
            if label is not None and entity.label != label:
                continue
            if properties is not None:
                if not all(entity.properties.get(k) == v for k, v in properties.items()):
                    continue
            if relationship is not None:
                rels = self._radj.get(entity.id, [])
                if not any(r.rel_type == relationship for r in rels):
                    continue
            results.append(entity)

        return results

    def get_neighbors(self, entity_id: str) -> List[str]:
        """Get IDs of directly connected entities."""
        if entity_id not in self._entities:
            return []
        neighbors: Set[str] = set()
        for r in self._adj.get(entity_id, []):
            neighbors.add(r.target)
        for r in self._radj.get(entity_id, []):
            neighbors.add(r.source)
        return sorted(neighbors)

    def path_exists(self, source: str, target: str) -> bool:
        """Check if a directed path exists between two entities (BFS)."""
        if source not in self._entities or target not in self._entities:
            return False
        if source == target:
            return True
        visited: Set[str] = set()
        queue = [source]
        visited.add(source)
        while queue:
            current = queue.pop(0)
            for r in self._adj.get(current, []):
                neighbor = r.target
                if neighbor == target:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def update_entity_properties(self, entity_id: str, properties: Dict[str, Any]) -> None:
        """Update properties of an existing entity."""
        if entity_id not in self._entities:
            raise ValueError(f"Entity '{entity_id}' not found")
        self._entities[entity_id].properties.update(properties)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the graph to a dictionary."""
        return {
            "entities": [
                {
                    "id": e.id,
                    "label": e.label,
                    "entity_type": e.entity_type,
                    "properties": dict(e.properties),
                }
                for e in self._entities.values()
            ],
            "relationships": [
                {
                    "source": r.source,
                    "target": r.target,
                    "rel_type": r.rel_type,
                    "properties": dict(r.properties),
                }
                for r in self._relationships
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgriKnowledgeGraph":
        """Deserialize a graph from a dictionary."""
        kg = cls()
        for e_data in data.get("entities", []):
            kg.add_entity(Entity(
                id=e_data["id"],
                label=e_data["label"],
                entity_type=e_data["entity_type"],
                properties=dict(e_data.get("properties", {})),
            ))
        for r_data in data.get("relationships", []):
            kg.add_relationship(Relationship(
                source=r_data["source"],
                target=r_data["target"],
                rel_type=r_data["rel_type"],
                properties=dict(r_data.get("properties", {})),
            ))
        return kg

    def clear(self) -> None:
        """Remove all entities and relationships."""
        self._entities.clear()
        self._relationships.clear()
        self._adj.clear()
        self._radj.clear()
