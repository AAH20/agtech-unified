"""Multi-twin synchronization for agricultural digital twin.

Provides TwinSynchronizer to keep multiple digital twin instances
consistent by periodically aligning their states toward the mean.

Also provides HierarchicalSynchronizer for multi-level consensus
across field-level, farm-level, and regional digital twins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from src.digital_twin.simulator import DigitalTwin, SimulationState

logger = logging.getLogger(__name__)


class TwinLevel(Enum):
    """Hierarchy level for a digital twin."""

    FIELD = "field"
    FARM = "farm"
    REGIONAL = "regional"


@dataclass
class TwinRegistration:
    """Registration info for a twin in the hierarchy."""

    twin: DigitalTwin
    level: TwinLevel
    weight: float = 1.0
    parent_id: Optional[str] = None


@dataclass
class SyncResult:
    """Result of a synchronization operation."""

    sync_count: int
    mean_state: SimulationState
    max_difference: float


class TwinSynchronizer:
    """Synchronizes multiple digital twin instances.

    Periodically aligns all registered twins toward the mean state,
    with configurable convergence strength.
    """

    def __init__(self, convergence_factor: float = 0.5):
        """
        Args:
            convergence_factor: How strongly twins move toward mean (0-1).
                0 = no sync, 1 = full convergence to mean.
        """
        self.convergence_factor = max(0.0, min(1.0, convergence_factor))
        self.twins: Dict[str, DigitalTwin] = {}

    def register_twin(self, twin_id: str, twin: DigitalTwin) -> None:
        """Register a digital twin for synchronization."""
        self.twins[twin_id] = twin

    def unregister_twin(self, twin_id: str) -> None:
        """Remove a twin from synchronization."""
        self.twins.pop(twin_id, None)

    def get_twin_state(self, twin_id: str) -> Optional[SimulationState]:
        """Get the current simulation state of a registered twin."""
        twin = self.twins.get(twin_id)
        if twin is None:
            return None
        state = twin.get_current_state()
        if state is not None:
            return state
        # If no state ingested yet, return a default state
        return SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.0,
            nutrient_level=0.5,
        )

    def sync_all(self, initial_states: Optional[Dict[str, SimulationState]] = None) -> SyncResult:
        """Synchronize all registered twins toward the mean state.

        Args:
            initial_states: Optional dict of twin_id -> SimulationState.
                If provided, these states are used instead of the twins'
                current states.

        Returns:
            SyncResult with the computed mean state and sync statistics.

        Raises:
            ValueError: If no twins are registered.
        """
        if not self.twins:
            raise ValueError("No twins registered")

        # Gather current states
        states: Dict[str, SimulationState] = {}
        for twin_id in self.twins:
            if initial_states and twin_id in initial_states:
                states[twin_id] = initial_states[twin_id]
            else:
                state = self.get_twin_state(twin_id)
                if state is not None:
                    states[twin_id] = state

        if not states:
            raise ValueError("No twin states available for synchronization")

        # Compute mean state
        mean_state = self._compute_mean_state(list(states.values()))

        # Update each twin toward the mean
        max_diff = 0.0
        for twin_id, twin in self.twins.items():
            if twin_id not in states:
                continue
            current = states[twin_id]
            new_state = self._interpolate_state(current, mean_state, self.convergence_factor)
            # Update the twin's internal state
            twin.ingest_sensor_data(_to_farm_state(new_state))
            # Track max difference from mean
            diff = self._state_difference(new_state, mean_state)
            max_diff = max(max_diff, diff)

        return SyncResult(
            sync_count=len(self.twins),
            mean_state=mean_state,
            max_difference=max_diff,
        )

    def _compute_mean_state(self, states: list) -> SimulationState:
        """Compute the mean state across all provided states."""
        if not states:
            raise ValueError("Cannot compute mean of empty state list")

        n = len(states)
        mean_moisture = sum(s.soil_moisture for s in states) / n
        mean_temp = sum(s.temperature for s in states) / n
        mean_height = sum(s.crop_height for s in states) / n
        mean_nutrient = sum(s.nutrient_level for s in states) / n
        mean_pest = sum(getattr(s, "pest_pressure", 0.0) for s in states) / n
        mean_disease = sum(getattr(s, "disease_pressure", 0.0) for s in states) / n
        mean_weed = sum(getattr(s, "weed_pressure", 0.0) for s in states) / n

        return SimulationState(
            soil_moisture=mean_moisture,
            temperature=mean_temp,
            crop_height=mean_height,
            nutrient_level=mean_nutrient,
            pest_pressure=mean_pest,
            disease_pressure=mean_disease,
            weed_pressure=mean_weed,
        )

    def _interpolate_state(
        self, current: SimulationState, target: SimulationState, factor: float
    ) -> SimulationState:
        """Move current state toward target by the given factor."""
        return SimulationState(
            soil_moisture=current.soil_moisture
            + factor * (target.soil_moisture - current.soil_moisture),
            temperature=current.temperature + factor * (target.temperature - current.temperature),
            crop_height=current.crop_height + factor * (target.crop_height - current.crop_height),
            nutrient_level=current.nutrient_level
            + factor * (target.nutrient_level - current.nutrient_level),
            pest_pressure=getattr(current, "pest_pressure", 0.0)
            + factor
            * (getattr(target, "pest_pressure", 0.0) - getattr(current, "pest_pressure", 0.0)),
            disease_pressure=getattr(current, "disease_pressure", 0.0)
            + factor
            * (
                getattr(target, "disease_pressure", 0.0) - getattr(current, "disease_pressure", 0.0)
            ),
            weed_pressure=getattr(current, "weed_pressure", 0.0)
            + factor
            * (getattr(target, "weed_pressure", 0.0) - getattr(current, "weed_pressure", 0.0)),
        )

    def _state_difference(self, a: SimulationState, b: SimulationState) -> float:
        """Compute the maximum absolute difference between two states."""
        diffs = [
            abs(a.soil_moisture - b.soil_moisture),
            abs(a.temperature - b.temperature),
            abs(a.crop_height - b.crop_height),
            abs(a.nutrient_level - b.nutrient_level),
            abs(getattr(a, "pest_pressure", 0.0) - getattr(b, "pest_pressure", 0.0)),
            abs(getattr(a, "disease_pressure", 0.0) - getattr(b, "disease_pressure", 0.0)),
            abs(getattr(a, "weed_pressure", 0.0) - getattr(b, "weed_pressure", 0.0)),
        ]
        return max(diffs)


def _to_farm_state(state: SimulationState):
    """Convert SimulationState to FarmState for ingestion."""
    from src.integration.farm_state import FarmState

    return FarmState(
        soil_moisture=state.soil_moisture,
        temperature=state.temperature,
        crop_height=state.crop_height,
        nutrient_level=state.nutrient_level,
        pest_pressure=getattr(state, "pest_pressure", 0.0),
    )


@dataclass
class HierarchicalSyncResult:
    """Result of a hierarchical synchronization operation."""

    level_results: Dict[str, SyncResult]
    regional_mean: SimulationState
    convergence_rounds: int


class HierarchicalSynchronizer:
    """Hierarchical multi-twin synchronization with weighted consensus.

    Supports three levels: field, farm, and regional. Field twins sync
    within their farm group, farm twins sync within the regional group,
    and the regional twin aggregates all farm means.

    Uses weighted averaging where each twin's influence is proportional
    to its weight (e.g., field area in hectares).
    """

    def __init__(self, convergence_factor: float = 0.5):
        """
        Args:
            convergence_factor: How strongly twins move toward mean (0-1).
        """
        self.convergence_factor = max(0.0, min(1.0, convergence_factor))
        self._registrations: Dict[str, TwinRegistration] = {}

    def register_twin(
        self,
        twin_id: str,
        twin: DigitalTwin,
        level: TwinLevel,
        weight: float = 1.0,
        parent_id: Optional[str] = None,
    ) -> None:
        """Register a digital twin in the hierarchy.

        Args:
            twin_id: Unique identifier for this twin.
            twin: The DigitalTwin instance.
            level: Hierarchy level (field, farm, or regional).
            weight: Weight for weighted averaging (e.g., field area).
            parent_id: Parent twin_id (farm_id for field twins).
        """
        if weight <= 0:
            raise ValueError("Weight must be positive")
        self._registrations[twin_id] = TwinRegistration(
            twin=twin, level=level, weight=weight, parent_id=parent_id
        )

    def unregister_twin(self, twin_id: str) -> None:
        """Remove a twin from the hierarchy."""
        self._registrations.pop(twin_id, None)

    def get_twins_at_level(self, level: TwinLevel) -> List[str]:
        """Get all twin IDs at a given hierarchy level."""
        return [tid for tid, reg in self._registrations.items() if reg.level == level]

    def get_children(self, parent_id: str) -> List[str]:
        """Get all child twin IDs for a given parent."""
        return [tid for tid, reg in self._registrations.items() if reg.parent_id == parent_id]

    def sync_hierarchical(self) -> HierarchicalSyncResult:
        """Perform hierarchical synchronization.

        1. Field twins sync within their farm groups (weighted mean).
        2. Farm twins sync toward the regional mean (weighted).
        3. Regional twin updates to the weighted mean of all farm means.

        Returns:
            HierarchicalSyncResult with per-level results and regional mean.
        """
        if not self._registrations:
            raise ValueError("No twins registered")

        level_results: Dict[str, SyncResult] = {}

        # Step 1: Field-level sync within each farm group
        farm_ids = self.get_twins_at_level(TwinLevel.FARM)
        field_ids = self.get_twins_at_level(TwinLevel.FIELD)

        for farm_id in farm_ids:
            children = self.get_children(farm_id)
            if not children:
                continue
            group_sync = TwinSynchronizer(convergence_factor=self.convergence_factor)
            for child_id in children:
                reg = self._registrations[child_id]
                group_sync.register_twin(child_id, reg.twin)
            result = group_sync.sync_all()
            level_results[f"farm:{farm_id}"] = result

        # Step 2: Farm-level sync toward regional mean
        if farm_ids:
            farm_sync = TwinSynchronizer(convergence_factor=self.convergence_factor)
            for farm_id in farm_ids:
                reg = self._registrations[farm_id]
                farm_sync.register_twin(farm_id, reg.twin)
            farm_result = farm_sync.sync_all()
            level_results["regional"] = farm_result
            regional_mean = farm_result.mean_state
        else:
            # No farm twins — compute regional mean from fields
            all_states = []
            for tid in field_ids:
                reg = self._registrations[tid]
                state = self._get_state(tid)
                if state is not None:
                    all_states.append((reg.weight, state))
            regional_mean = self._weighted_mean(all_states)

        # Step 3: Update regional twin if one exists
        regional_ids = self.get_twins_at_level(TwinLevel.REGIONAL)
        for reg_id in regional_ids:
            reg = self._registrations[reg_id]
            current = self._get_state(reg_id)
            if current is not None:
                new_state = self._interpolate_state(current, regional_mean, self.convergence_factor)
                reg.twin.ingest_sensor_data(_to_farm_state(new_state))

        return HierarchicalSyncResult(
            level_results=level_results,
            regional_mean=regional_mean,
            convergence_rounds=1,
        )

    def merge_states(
        self, states: Dict[str, SimulationState], weights: Optional[Dict[str, float]] = None
    ) -> SimulationState:
        """Merge multiple twin states using weighted averaging.

        Args:
            states: Dict of twin_id -> SimulationState.
            weights: Optional dict of twin_id -> weight. Uses registration
                weights if not provided.

        Returns:
            Merged SimulationState.
        """
        if not states:
            raise ValueError("No states to merge")

        weighted_states: List[tuple] = []
        for tid, state in states.items():
            if weights and tid in weights:
                w = weights[tid]
            elif tid in self._registrations:
                w = self._registrations[tid].weight
            else:
                w = 1.0
            weighted_states.append((w, state))

        return self._weighted_mean(weighted_states)

    def _get_state(self, twin_id: str) -> Optional[SimulationState]:
        """Get current state of a registered twin."""
        reg = self._registrations.get(twin_id)
        if reg is None:
            return None
        state = reg.twin.get_current_state()
        if state is not None:
            return state
        return SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.0,
            nutrient_level=0.5,
        )

    def _weighted_mean(self, weighted_states: List[tuple]) -> SimulationState:
        """Compute weighted mean of states.

        Args:
            weighted_states: List of (weight, SimulationState) tuples.

        Returns:
            Weighted mean SimulationState.
        """
        if not weighted_states:
            raise ValueError("Cannot compute mean of empty state list")

        total_weight = sum(w for w, _ in weighted_states)
        if total_weight <= 0:
            raise ValueError("Total weight must be positive")

        mean_moisture = sum(w * s.soil_moisture for w, s in weighted_states) / total_weight
        mean_temp = sum(w * s.temperature for w, s in weighted_states) / total_weight
        mean_height = sum(w * s.crop_height for w, s in weighted_states) / total_weight
        mean_nutrient = sum(w * s.nutrient_level for w, s in weighted_states) / total_weight
        mean_pest = (
            sum(w * getattr(s, "pest_pressure", 0.0) for w, s in weighted_states) / total_weight
        )
        mean_disease = (
            sum(w * getattr(s, "disease_pressure", 0.0) for w, s in weighted_states) / total_weight
        )
        mean_weed = (
            sum(w * getattr(s, "weed_pressure", 0.0) for w, s in weighted_states) / total_weight
        )

        return SimulationState(
            soil_moisture=mean_moisture,
            temperature=mean_temp,
            crop_height=mean_height,
            nutrient_level=mean_nutrient,
            pest_pressure=mean_pest,
            disease_pressure=mean_disease,
            weed_pressure=mean_weed,
        )

    def _interpolate_state(
        self, current: SimulationState, target: SimulationState, factor: float
    ) -> SimulationState:
        """Move current state toward target by the given factor."""
        return SimulationState(
            soil_moisture=current.soil_moisture
            + factor * (target.soil_moisture - current.soil_moisture),
            temperature=current.temperature + factor * (target.temperature - current.temperature),
            crop_height=current.crop_height + factor * (target.crop_height - current.crop_height),
            nutrient_level=current.nutrient_level
            + factor * (target.nutrient_level - current.nutrient_level),
            pest_pressure=getattr(current, "pest_pressure", 0.0)
            + factor
            * (getattr(target, "pest_pressure", 0.0) - getattr(current, "pest_pressure", 0.0)),
            disease_pressure=getattr(current, "disease_pressure", 0.0)
            + factor
            * (
                getattr(target, "disease_pressure", 0.0) - getattr(current, "disease_pressure", 0.0)
            ),
            weed_pressure=getattr(current, "weed_pressure", 0.0)
            + factor
            * (getattr(target, "weed_pressure", 0.0) - getattr(current, "weed_pressure", 0.0)),
        )
