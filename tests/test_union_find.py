"""Tests for Union-Find data structure used in VRP route tracking."""

import pytest

from src.integration.union_find import UnionFind


class TestUnionFindBasic:
    """Core union-find operations."""

    def test_add_and_find(self):
        uf = UnionFind()
        uf.add("A")
        assert uf.find("A") == "A"

    def test_union_connects_elements(self):
        uf = UnionFind()
        uf.union("A", "B")
        assert uf.connected("A", "B")

    def test_union_returns_true_for_new_connection(self):
        uf = UnionFind()
        assert uf.union("A", "B") is True

    def test_union_returns_false_when_already_connected(self):
        uf = UnionFind()
        uf.union("A", "B")
        assert uf.union("A", "B") is False

    def test_transitive_connectivity(self):
        uf = UnionFind()
        uf.union("A", "B")
        uf.union("B", "C")
        assert uf.connected("A", "C")

    def test_disconnected_elements(self):
        uf = UnionFind()
        uf.add("A")
        uf.add("B")
        assert not uf.connected("A", "B")

    def test_single_element_is_own_parent(self):
        uf = UnionFind()
        uf.add("X")
        assert uf.find("X") == "X"

    def test_union_creates_connection_without_explicit_add(self):
        uf = UnionFind()
        uf.union("A", "B")
        assert uf.connected("A", "B")
        assert uf.find("A") == uf.find("B")


class TestUnionFindComponents:
    """Component tracking for VRP route grouping."""

    def test_single_component(self):
        uf = UnionFind()
        uf.union("A", "B")
        uf.union("B", "C")
        components = uf.get_components()
        assert len(components) == 1
        assert set(components.keys()) == {uf.find("A")}

    def test_multiple_components(self):
        uf = UnionFind()
        uf.union("A", "B")
        uf.union("C", "D")
        components = uf.get_components()
        assert len(components) == 2

    def test_component_count(self):
        uf = UnionFind()
        uf.add("A")
        uf.add("B")
        uf.add("C")
        assert uf.component_count() == 3
        uf.union("A", "B")
        assert uf.component_count() == 2

    def test_component_membership(self):
        uf = UnionFind()
        uf.union("A", "B")
        uf.union("C", "D")
        components = uf.get_components()
        all_members = set()
        for members in components.values():
            all_members.update(members)
        assert all_members == {"A", "B", "C", "D"}

    def test_empty_union_find(self):
        uf = UnionFind()
        assert uf.component_count() == 0
        assert uf.get_components() == {}


class TestUnionFindPathCompression:
    """Path compression keeps trees flat for near-O(1) amortized lookups."""

    def test_path_compression_flattens_tree(self):
        uf = UnionFind()
        # Create a chain: A -> B -> C -> D -> E
        for i in range(5):
            uf.add(i)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(2, 3)
        uf.union(3, 4)
        # After find, all nodes should point directly to root
        root = uf.find(0)
        for i in range(5):
            assert uf.find(i) == root

    def test_large_chain_stays_efficient(self):
        uf = UnionFind()
        n = 100
        for i in range(n):
            uf.union(i, i + 1)
        # All should be connected
        for i in range(n + 1):
            assert uf.connected(0, i)


class TestUnionFindVRPUseCase:
    """VRP route tracking: customers grouped by vehicle route."""

    def test_vrp_route_grouping(self):
        """Simulate assigning customers to routes and tracking groups."""
        uf = UnionFind()
        # Route 1: customers 0, 1, 2
        uf.union(0, 1)
        uf.union(1, 2)
        # Route 2: customers 3, 4
        uf.union(3, 4)
        # Route 3: customer 5 alone
        uf.add(5)

        assert uf.connected(0, 2)
        assert uf.connected(3, 4)
        assert not uf.connected(0, 3)
        assert not uf.connected(0, 5)
        assert uf.component_count() == 3

    def test_vrp_merge_routes(self):
        """Merging two routes into one."""
        uf = UnionFind()
        uf.union(0, 1)
        uf.union(2, 3)
        assert uf.component_count() == 2
        # Merge routes by connecting customer 1 and 2
        uf.union(1, 2)
        assert uf.component_count() == 1
        assert uf.connected(0, 3)

    def test_vrp_customer_in_route(self):
        """Check if a customer belongs to a specific route."""
        uf = UnionFind()
        route_a = [0, 1, 2]
        route_b = [3, 4, 5]
        for c in route_a[1:]:
            uf.union(route_a[0], c)
        for c in route_b[1:]:
            uf.union(route_b[0], c)

        root_a = uf.find(0)
        assert all(uf.find(c) == root_a for c in route_a)
        assert all(uf.find(c) != root_a for c in route_b)


class TestUnionFindEdgeCases:
    """Edge cases and error handling."""

    def test_self_union(self):
        uf = UnionFind()
        uf.add("A")
        assert uf.union("A", "A") is False
        assert uf.connected("A", "A")

    def test_repeated_unions(self):
        uf = UnionFind()
        for _ in range(10):
            uf.union("A", "B")
        assert uf.connected("A", "B")
        assert uf.component_count() == 1

    def test_mixed_types(self):
        uf = UnionFind()
        uf.union(1, "A")
        uf.union("A", (2, 3))
        assert uf.connected(1, (2, 3))

    def test_hashable_elements_only(self):
        uf = UnionFind()
        with pytest.raises(TypeError):
            uf.add([1, 2, 3])  # lists are unhashable
