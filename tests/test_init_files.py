"""Tests for __init__.py files: docstrings, __all__, and re-exports."""

import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _get_all_init_packages():
    """Return all package __init__.py paths under src/ only."""
    packages = []
    for path in (ROOT / "src").rglob("__init__.py"):
        if "__pycache__" not in str(path):
            packages.append(path)
    return sorted(packages)


def test_all_init_files_have_docstrings():
    """Every __init__.py must have a non-empty module docstring."""
    for init_path in _get_all_init_packages():
        rel = init_path.relative_to(ROOT)
        content = init_path.read_text().strip()
        assert content, f"{rel} is empty"
        # Must start with a docstring (triple-quoted)
        assert content.startswith('"""') or content.startswith("'''"), (
            f"{rel} has no module docstring"
        )
        # Extract docstring
        if content.startswith('"""'):
            end = content.index('"""', 3)
            docstring = content[3:end].strip()
        else:
            end = content.index("'''", 3)
            docstring = content[3:end].strip()
        assert len(docstring) > 10, f"{rel} docstring too short: {docstring!r}"


def test_all_init_files_define_all():
    """Every __init__.py must define __all__."""
    for init_path in _get_all_init_packages():
        rel = init_path.relative_to(ROOT)
        content = init_path.read_text()
        assert "__all__" in content, f"{rel} missing __all__"


def test_all_init_files_have_reexports():
    """Every __init__.py must re-export at least one symbol."""
    for init_path in _get_all_init_packages():
        rel = init_path.relative_to(ROOT)
        content = init_path.read_text()
        # Check for import statements (re-exports)
        has_import = any(
            line.strip().startswith(("from ", "import ")) for line in content.splitlines()
        )
        assert has_import, f"{rel} has no re-exports"


def test_src_init_docstring():
    """src/__init__.py must describe the overall package."""
    content = (ROOT / "src" / "__init__.py").read_text()
    assert "agtech" in content.lower() or "agriculture" in content.lower()


def test_optimization_init_reexports():
    """optimization/__init__.py must re-export TSP, VRP, GPU solvers."""
    from src import optimization

    assert hasattr(optimization, "TSPSolver")
    assert hasattr(optimization, "VRPSolver")
    assert hasattr(optimization, "GPUTSPSolver")
    assert hasattr(optimization, "GPUVRPSolver")


def test_genomics_init_reexports():
    """genomics/__init__.py must re-export CRISPR and protein classes."""
    from src import genomics

    assert hasattr(genomics, "CRISPRDesigner")
    assert hasattr(genomics, "GuideRNA")


def test_iot_init_reexports():
    """iot/__init__.py must re-export key IoT classes."""
    from src import iot

    assert hasattr(iot, "MQTTClient")
    assert hasattr(iot, "SensorPlacement")
    assert hasattr(iot, "NGSILDBroker")


def test_decision_support_init_reexports():
    """decision_support/__init__.py must re-export key classes."""
    from src import decision_support

    assert hasattr(decision_support, "DecisionEngine")
    assert hasattr(decision_support, "ZeroTrustAuth")
    assert hasattr(decision_support, "APIGateway")


def test_multi_agent_init_reexports():
    """multi_agent/__init__.py must re-export swarm/consensus classes."""
    from src import multi_agent

    assert hasattr(multi_agent, "SwarmCoordinator")
    assert hasattr(multi_agent, "ByzantineConsensus")
    assert hasattr(multi_agent, "TaskAllocator")


def test_digital_twin_init_reexports():
    """digital_twin/__init__.py must re-export simulator and knowledge graph."""
    from src import digital_twin

    assert hasattr(digital_twin, "DigitalTwin")
    assert hasattr(digital_twin, "AgriKnowledgeGraph")
    assert hasattr(digital_twin, "AGROVOCOntology")


def test_path_planning_init_reexports():
    """path_planning/__init__.py must re-export coverage planner."""
    from src import path_planning

    assert hasattr(path_planning, "CoveragePlanner")
    assert hasattr(path_planning, "ObstacleField")


def test_integration_init_reexports():
    """integration/__init__.py must re-export event bus and farm state."""
    from src import integration

    assert hasattr(integration, "EventBus")
    assert hasattr(integration, "FarmState")
    assert hasattr(integration, "UnifiedOptimizer")


def test_onboarding_init_reexports():
    """onboarding/__init__.py must re-export module registry."""
    from src import onboarding

    assert hasattr(onboarding, "ModuleRegistry")
    assert hasattr(onboarding, "OrganizationProfiler")


def test_all_lists_are_valid():
    """All __all__ lists must contain only strings."""
    for init_path in _get_all_init_packages():
        rel = init_path.relative_to(ROOT)
        content = init_path.read_text()
        if "__all__" not in content:
            continue
        # Parse __all__ from the module
        module_name = str(init_path.parent.relative_to(ROOT)).replace("/", ".")
        if module_name == ".":
            module_name = "tests"
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(mod, "__all__"):
            assert isinstance(mod.__all__, list), f"{rel}: __all__ not a list"
            for item in mod.__all__:
                assert isinstance(item, str), f"{rel}: __all__ item {item!r} not a string"
