"""Module dependency graph for onboarding modules.

Provides a directed graph of module dependencies with topological
ordering for installation sequencing and cycle detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.onboarding.modules import ModuleRegistry


class ModuleDependencyGraph:
    """Directed graph of module dependencies.

    Each edge A -> B means "A depends on B" (B must be installed
    before A). Supports topological ordering and cycle detection.
    """

    def __init__(self) -> None:
        """Initialize an empty dependency graph."""
        self._graph: dict[str, list[str]] = {}

    def add_dependency(self, module: str, depends_on: str) -> None:
        """Add a dependency edge: module depends on depends_on.

        Args:
            module: The dependent module.
            depends_on: The prerequisite module.
        """
        if module not in self._graph:
            self._graph[module] = []
        if depends_on not in self._graph[module]:
            self._graph[module].append(depends_on)
        # Ensure the prerequisite is also in the graph
        if depends_on not in self._graph:
            self._graph[depends_on] = []

    def get_dependencies(self, module: str) -> list[str]:
        """Return direct dependencies of a module.

        Args:
            module: Module name.

        Returns:
            List of direct dependency module names.
        """
        return list(self._graph.get(module, []))

    def get_all_dependencies(self, module: str) -> list[str]:
        """Return all transitive dependencies of a module.

        Args:
            module: Module name.

        Returns:
            List of all transitive dependency module names.
        """
        visited: set[str] = set()
        stack = list(self._graph.get(module, []))
        while stack:
            dep = stack.pop()
            if dep in visited:
                continue
            visited.add(dep)
            stack.extend(self._graph.get(dep, []))
        return list(visited)

    def get_install_order(self, modules: list[str]) -> list[str]:
        """Return modules in dependency-respecting install order.

        Uses Kahn's algorithm for topological sorting. Raises
        ValueError if a cycle is detected.

        Args:
            modules: List of module names to order.

        Returns:
            Topologically sorted list of module names.

        Raises:
            ValueError: If the graph contains a cycle.
        """
        if not modules:
            return []

        # Build in-degree map for the requested modules
        in_degree: dict[str, int] = {m: 0 for m in modules}
        for m in modules:
            for dep in self._graph.get(m, []):
                if dep in in_degree:
                    in_degree[dep] = in_degree.get(dep, 0)  # ensure dep is tracked
                    in_degree[m] = in_degree.get(m, 0) + 1

        # Recompute in-degree correctly
        in_degree = {m: 0 for m in modules}
        for m in modules:
            for dep in self._graph.get(m, []):
                if dep in modules:
                    in_degree[m] += 1

        # Kahn's algorithm
        queue = [m for m in modules if in_degree[m] == 0]
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            # Find all modules that depend on this node
            for m in modules:
                if node in self._graph.get(m, []):
                    in_degree[m] -= 1
                    if in_degree[m] == 0:
                        queue.append(m)

        if len(result) != len(modules):
            raise ValueError("Dependency graph contains a cycle")

        return result

    def validate(self) -> None:
        """Validate the graph has no cycles.

        Raises:
            ValueError: If a cycle is detected.
        """
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def _dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for dep in self._graph.get(node, []):
                if dep not in visited:
                    if _dfs(dep):
                        return True
                elif dep in rec_stack:
                    return True
            rec_stack.discard(node)
            return False

        for node in self._graph:
            if node not in visited:
                if _dfs(node):
                    raise ValueError("Dependency graph contains a cycle")

    @classmethod
    def from_registry(cls, registry: ModuleRegistry) -> ModuleDependencyGraph:
        """Build a dependency graph from a ModuleRegistry.

        Creates a graph where higher-tier modules depend on
        lower-tier modules. Enterprise modules depend on SMB
        modules, and all modules depend on the base modules
        (basics, sensors, alerts).

        Args:
            registry: A ModuleRegistry instance.

        Returns:
            A populated ModuleDependencyGraph.
        """
        graph = cls()

        # Base modules that everything depends on
        base_modules = ["basics", "sensors", "alerts"]

        # SMB modules depend on base modules
        smb_modules = registry.get_modules("smb")
        for mod in smb_modules:
            if mod not in base_modules:
                for base in base_modules:
                    graph.add_dependency(mod, base)

        # Enterprise modules depend on SMB modules
        enterprise_modules = registry.get_modules("enterprise")
        for mod in enterprise_modules:
            if mod not in smb_modules:
                for smb_mod in smb_modules:
                    graph.add_dependency(mod, smb_mod)

        return graph
