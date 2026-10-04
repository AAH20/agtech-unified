"""Consistent hashing for partition assignment (IOT gap: hash() is randomized).

Python's built-in hash() is salted per process (PYTHONHASHSEED), so
partition assignments change across restarts. This module provides a
consistent hash ring using MD5-based hashing with virtual nodes for
stable, balanced assignment that survives node additions and removals.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_right
from typing import Dict, List, Tuple


class ConsistentHashRing:
    """Consistent hash ring with virtual nodes.

    Maps keys to partitions and named nodes using MD5 hashing.
    Adding or removing a node remaps only ~1/N of keys, unlike
    modulo-based hashing which remaps nearly all keys.
    """

    def __init__(
        self,
        num_partitions: int = 4,
        virtual_nodes: int = 150,
    ) -> None:
        if num_partitions <= 0:
            raise ValueError("num_partitions must be positive")
        if virtual_nodes <= 0:
            raise ValueError("virtual_nodes must be positive")
        self.num_partitions = num_partitions
        self.virtual_nodes = virtual_nodes
        self._nodes: List[str] = [f"node-{i}" for i in range(num_partitions)]
        self._node_partitions: Dict[str, int] = {f"node-{i}": i for i in range(num_partitions)}
        self._ring: List[Tuple[int, str]] = []  # (hash, node_name), sorted by hash
        self._build_ring()

    def _build_ring(self) -> None:
        """Build the virtual-node ring from current nodes."""
        self._ring = []
        for node in self._nodes:
            for v in range(self.virtual_nodes):
                key = f"{node}:{v}"
                h = self._hash(key)
                self._ring.append((h, node))
        self._ring.sort(key=lambda x: x[0])

    @staticmethod
    def _hash(key: str) -> int:
        """MD5-based hash (deterministic across processes)."""
        return int(hashlib.md5(key.encode(), usedforsecurity=False).hexdigest(), 16)

    def _find_node(self, key: str) -> str:
        """Find the node responsible for a key (clockwise on the ring)."""
        if not self._ring:
            raise RuntimeError("hash ring is empty")
        h = self._hash(key)
        idx = bisect_right(self._ring, (h, "\xff"))
        if idx >= len(self._ring):
            idx = 0
        return self._ring[idx][1]

    def get_partition(self, key: str) -> int:
        """Get the partition number for a key."""
        node = self._find_node(key)
        return self._node_partitions[node]

    def get_node(self, key: str) -> str:
        """Get the node name responsible for a key."""
        return self._find_node(key)

    def add_node(self, name: str) -> None:
        """Add a node to the ring. Remaps only ~1/N of keys."""
        if name in self._nodes:
            raise ValueError(f"Node '{name}' already exists")
        partition = len(self._nodes) % self.num_partitions
        self._nodes.append(name)
        self._node_partitions[name] = partition
        self._build_ring()

    def remove_node(self, name: str) -> None:
        """Remove a node from the ring. Remaps only its own keys."""
        if name not in self._nodes:
            raise ValueError(f"Node '{name}' does not exist")
        self._nodes.remove(name)
        del self._node_partitions[name]
        self._build_ring()

    def get_nodes(self) -> List[str]:
        """Return all node names in the ring."""
        return list(self._nodes)
