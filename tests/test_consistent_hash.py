"""Tests for consistent hashing partition assignment (IOT gap: hash() is randomized).

Python's built-in hash() is salted per process (PYTHONHASHSEED), so
partition assignments change across restarts. ConsistentHashRing uses
MD5-based hashing with virtual nodes for stable, balanced assignment.
"""

import pytest

from src.iot.consistent_hash import ConsistentHashRing


class TestConsistentHashRing:
    """Consistent hashing ring: assignment, stability, rebalancing."""

    def test_assignment_in_range(self):
        """Assigned partition is always in [0, num_partitions)."""
        ring = ConsistentHashRing(num_partitions=4)
        for i in range(100):
            p = ring.get_partition(f"sensor-{i}")
            assert 0 <= p < 4

    def test_consistent_assignment(self):
        """The same key always maps to the same partition."""
        ring = ConsistentHashRing(num_partitions=4)
        p1 = ring.get_partition("sensor-001")
        p2 = ring.get_partition("sensor-001")
        assert p1 == p2

    def test_stable_across_instances(self):
        """Two independently built rings assign identically."""
        ring1 = ConsistentHashRing(num_partitions=4)
        ring2 = ConsistentHashRing(num_partitions=4)
        for i in range(50):
            key = f"sensor-{i}"
            assert ring1.get_partition(key) == ring2.get_partition(key)

    def test_distribution(self):
        """Keys are distributed across multiple partitions."""
        ring = ConsistentHashRing(num_partitions=4)
        partitions = {ring.get_partition(f"sensor-{i}") for i in range(100)}
        assert len(partitions) > 1

    def test_add_node_minimal_remap(self):
        """Adding a node remaps only a fraction of keys."""
        ring = ConsistentHashRing(num_partitions=4)
        before = {f"sensor-{i}": ring.get_partition(f"sensor-{i}") for i in range(200)}
        ring.add_node("node-5")
        after = {f"sensor-{i}": ring.get_partition(f"sensor-{i}") for i in range(200)}
        remapped = sum(1 for k in before if before[k] != after[k])
        # Consistent hashing should remap roughly 1/N of keys, not all
        assert remapped < 100  # Less than half

    def test_remove_node_minimal_remap(self):
        """Removing a node remaps only its own keys."""
        ring = ConsistentHashRing(num_partitions=4)
        before = {f"sensor-{i}": ring.get_partition(f"sensor-{i}") for i in range(200)}
        ring.remove_node("node-0")
        after = {f"sensor-{i}": ring.get_partition(f"sensor-{i}") for i in range(200)}
        remapped = sum(1 for k in before if before[k] != after[k])
        assert remapped < 100

    def test_virtual_nodes_improve_balance(self):
        """More virtual nodes yield better distribution."""
        ring_few = ConsistentHashRing(num_partitions=4, virtual_nodes=1)
        ring_many = ConsistentHashRing(num_partitions=4, virtual_nodes=100)
        keys = [f"sensor-{i}" for i in range(200)]
        counts_few = [0] * 4
        counts_many = [0] * 4
        for k in keys:
            counts_few[ring_few.get_partition(k)] += 1
            counts_many[ring_many.get_partition(k)] += 1
        # More virtual nodes should give a lower max/min ratio
        assert max(counts_many) <= max(counts_few)

    def test_invalid_num_partitions(self):
        """num_partitions must be positive."""
        with pytest.raises(ValueError):
            ConsistentHashRing(num_partitions=0)
        with pytest.raises(ValueError):
            ConsistentHashRing(num_partitions=-1)

    def test_invalid_virtual_nodes(self):
        """virtual_nodes must be positive."""
        with pytest.raises(ValueError):
            ConsistentHashRing(virtual_nodes=0)

    def test_get_node(self):
        """get_node returns the node responsible for a key."""
        ring = ConsistentHashRing(num_partitions=4)
        node = ring.get_node("sensor-001")
        assert node is not None
        assert isinstance(node, str)

    def test_nodes_listed(self):
        """All configured nodes are listed."""
        ring = ConsistentHashRing(num_partitions=4)
        nodes = ring.get_nodes()
        assert len(nodes) == 4
        assert all(isinstance(n, str) for n in nodes)
