"""Union-Find (Disjoint Set Union) data structure for VRP route tracking.

Provides near-O(1) amortized merge and find operations using path compression
and union by rank. Used to efficiently track which customers belong to the
same vehicle route in VRP solvers.
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Set


class UnionFind:
    """Union-Find with path compression and union by rank.

    Supports any hashable element type. Amortized time per operation is
    O(α(n)) — effectively constant for all practical purposes.

    Example:
        uf = UnionFind()
        uf.union("A", "B")
        uf.union("B", "C")
        assert uf.connected("A", "C")
        assert uf.component_count() == 1
    """

    def __init__(self) -> None:
        self._parent: Dict[Hashable, Hashable] = {}
        self._rank: Dict[Hashable, int] = {}

    def add(self, element: Hashable) -> None:
        """Add a new element as a singleton set."""
        if element not in self._parent:
            self._parent[element] = element
            self._rank[element] = 0

    def find(self, element: Hashable) -> Hashable:
        """Find the root representative of the element's set.

        Uses path compression: every node on the path points directly to root.
        """
        if element not in self._parent:
            self.add(element)
            return element
        # Path compression
        if self._parent[element] != element:
            self._parent[element] = self.find(self._parent[element])
        return self._parent[element]

    def union(self, a: Hashable, b: Hashable) -> bool:
        """Merge the sets containing a and b.

        Returns True if a merge occurred, False if already in the same set.
        Uses union by rank to keep trees shallow.
        """
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False
        # Union by rank: attach smaller tree under larger tree
        if self._rank[root_a] < self._rank[root_b]:
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a
        if self._rank[root_a] == self._rank[root_b]:
            self._rank[root_a] += 1
        return True

    def connected(self, a: Hashable, b: Hashable) -> bool:
        """Check if two elements are in the same set."""
        return self.find(a) == self.find(b)

    def component_count(self) -> int:
        """Return the number of disjoint sets."""
        return len({self.find(e) for e in self._parent})

    def get_components(self) -> Dict[Hashable, List[Hashable]]:
        """Return all components as {root: [members]} mapping."""
        components: Dict[Hashable, List[Hashable]] = {}
        for element in self._parent:
            root = self.find(element)
            components.setdefault(root, []).append(element)
        return components

    def get_members(self, element: Hashable) -> Set[Hashable]:
        """Return all elements in the same set as the given element."""
        root = self.find(element)
        return {e for e in self._parent if self.find(e) == root}

    def __len__(self) -> int:
        """Total number of elements."""
        return len(self._parent)

    def __contains__(self, element: Hashable) -> bool:
        """Check if element exists in the structure."""
        return element in self._parent
