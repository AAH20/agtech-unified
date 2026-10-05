"""Shared fixtures for AgTech Unified test suite."""

from __future__ import annotations

import pytest

from src.decision_support.recommender import DecisionEngine, FarmState
from src.digital_twin.simulator import DigitalTwin, SimulationState
from src.genomics.crispr import CRISPRDesigner
from src.genomics.protein import ProteinAnalyzer
from src.onboarding.modules import ModuleRegistry
from src.onboarding.sizing import OrganizationProfiler


def pytest_collection_modifyitems(config, items):
    """Skip GPU-marked tests when torch is not installed."""
    try:
        import torch  # noqa: F401

        return
    except ImportError:
        skip_gpu = pytest.mark.skip(reason="torch not installed — GPU tests skipped")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Engine / Analyzer fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def decision_engine() -> DecisionEngine:
    """Default rule-based decision engine."""
    return DecisionEngine()


@pytest.fixture
def digital_twin() -> DigitalTwin:
    """Default logistic-growth digital twin."""
    return DigitalTwin()


@pytest.fixture
def crispr_designer() -> CRISPRDesigner:
    """Default CRISPR designer (NGG PAM, 20nt guide)."""
    return CRISPRDesigner()


@pytest.fixture
def protein_analyzer() -> ProteinAnalyzer:
    """Default protein analyzer."""
    return ProteinAnalyzer()


@pytest.fixture
def module_registry() -> ModuleRegistry:
    """Default module registry."""
    return ModuleRegistry()


@pytest.fixture
def org_profiler() -> OrganizationProfiler:
    """Default organization profiler."""
    return OrganizationProfiler()


# ---------------------------------------------------------------------------
# FarmState fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def healthy_farm_state() -> FarmState:
    """Healthy farm state — no urgent recommendations expected."""
    return FarmState(
        soil_moisture=0.6,
        temperature=25.0,
        crop_height=0.5,
        nutrient_level=0.7,
        pest_pressure=0.1,
    )


@pytest.fixture
def drought_farm_state() -> FarmState:
    """Drought conditions — low soil moisture."""
    return FarmState(
        soil_moisture=0.1,
        temperature=30.0,
        crop_height=0.3,
        nutrient_level=0.5,
        pest_pressure=0.2,
    )


@pytest.fixture
def extreme_farm_state() -> FarmState:
    """Extreme farm state — all values at boundaries."""
    return FarmState(
        soil_moisture=0.0,
        temperature=50.0,
        crop_height=0.0,
        nutrient_level=0.0,
        pest_pressure=1.0,
    )


@pytest.fixture
def all_zeros_farm_state() -> FarmState:
    """All-zero farm state."""
    return FarmState(
        soil_moisture=0.0,
        temperature=0.0,
        crop_height=0.0,
        nutrient_level=0.0,
        pest_pressure=0.0,
    )


# ---------------------------------------------------------------------------
# SimulationState fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def optimal_sim_state() -> SimulationState:
    """Optimal growing conditions."""
    return SimulationState(
        soil_moisture=0.5,
        temperature=25.0,
        crop_height=0.1,
        nutrient_level=0.5,
    )


@pytest.fixture
def drought_sim_state() -> SimulationState:
    """Drought conditions for simulation."""
    return SimulationState(
        soil_moisture=0.05,
        temperature=25.0,
        crop_height=0.1,
        nutrient_level=0.5,
    )


@pytest.fixture
def extreme_heat_sim_state() -> SimulationState:
    """Extreme heat — beyond 45°C threshold."""
    return SimulationState(
        soil_moisture=0.8,
        temperature=50.0,
        crop_height=0.5,
        nutrient_level=0.8,
    )


# ---------------------------------------------------------------------------
# DNA / Protein sequence fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_dna_sequence() -> str:
    """Valid DNA sequence with PAM site."""
    return "ATCGATCGATCGATCGATCGATCGGG"


@pytest.fixture
def gc_rich_dna_sequence() -> str:
    """GC-rich DNA sequence with PAM site."""
    return "GCGCGCGCGCGCGCGCGCGCGGG"


@pytest.fixture
def no_pam_dna_sequence() -> str:
    """DNA sequence without any PAM site."""
    return "ATATATATATATATATATAT"


@pytest.fixture
def valid_protein_sequence() -> str:
    """Valid protein sequence with all 20 amino acids."""
    return "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture
def single_aa_protein_sequence() -> str:
    """Single amino acid protein sequence."""
    return "A"


@pytest.fixture
def all_same_protein_sequence() -> str:
    """Protein sequence with all same amino acid."""
    return "AAAAAAAAAA"


@pytest.fixture
def invalid_protein_sequence() -> str:
    """Protein sequence with invalid amino acid."""
    return "ACDEFGHIKLMNPQRSTVWYZ"
