"""Root growth model for agricultural digital twin (ROOT-001, SIM-003).

Simulates dynamic root zone expansion (logistic depth growth) and root
water uptake distributed by soil depth, with per-layer root length density
and moisture-dependent uptake.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class RootLayerState:
    """State of a single soil layer in the root zone."""

    depth_top: float  # m
    depth_bottom: float  # m
    root_length_density: float  # cm/cm³
    soil_moisture: float  # m³/m³


@dataclass
class RootGrowthState:
    """State of the root system at a point in time."""

    root_depth: float  # m
    layers: List[RootLayerState] = field(default_factory=list)
    total_root_length: float = 0.0  # cm/cm² (integrated over profile)


@dataclass
class RootGrowthResult:
    """Result of a root growth simulation."""

    days_simulated: int
    final_state: RootGrowthState
    history: List[RootGrowthState]
    total_water_uptake: float  # mm over the simulation
    max_root_depth: float  # m


class RootGrowthModel:
    """Dynamic root growth and root water uptake model.

    Root depth follows logistic expansion toward a crop-specific maximum.
    Root length density decays exponentially with depth. Water uptake is
    computed per soil layer from root length density, available moisture
    (above wilting point), and layer thickness.
    """

    def __init__(
        self,
        max_root_depth: float = 1.5,
        initial_root_depth: float = 0.1,
        root_growth_rate: float = 0.08,
        n_layers: int = 5,
        wilting_point: float = 0.10,
        field_capacity: float = 0.30,
        root_decay_coefficient: float = 3.0,
    ):
        if max_root_depth <= 0:
            raise ValueError("max_root_depth must be positive")
        if initial_root_depth <= 0 or initial_root_depth > max_root_depth:
            raise ValueError("initial_root_depth must be in (0, max_root_depth]")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.max_root_depth = max_root_depth
        self.initial_root_depth = initial_root_depth
        self.root_growth_rate = root_growth_rate
        self.n_layers = n_layers
        self.wilting_point = wilting_point
        self.field_capacity = field_capacity
        self.root_decay_coefficient = root_decay_coefficient

    def root_depth_at_day(self, day: int) -> float:
        """Logistic root depth at a given day.

        d(t) = d_max / (1 + ((d_max - d0) / d0) * exp(-r * t))
        """
        if day < 0:
            raise ValueError("Day must be non-negative")
        d0 = self.initial_root_depth
        d_max = self.max_root_depth
        ratio = (d_max - d0) / d0
        return d_max / (1.0 + ratio * math.exp(-self.root_growth_rate * day))

    def root_distribution(self, depth: float) -> float:
        """Relative root length density at a given depth (0-1).

        Exponential decay: RLD(d) = exp(-k * d), normalized to 1 at surface.
        """
        if depth < 0:
            raise ValueError("Depth must be non-negative")
        return math.exp(-self.root_decay_coefficient * depth)

    def layer_water_uptake(self, layer: RootLayerState, demand: float) -> float:
        """Water uptake (mm) from a single layer given crop demand.

        Uptake scales with root length density, available moisture above
        wilting point, and layer thickness. Capped at demand.
        """
        if demand < 0:
            raise ValueError("Demand must be non-negative")
        if layer.root_length_density <= 0.0:
            return 0.0
        thickness = max(0.0, layer.depth_bottom - layer.depth_top)
        if thickness <= 0.0:
            return 0.0

        available = max(0.0, layer.soil_moisture - self.wilting_point)
        # Uptake capacity: root density × available water × thickness (mm).
        uptake = layer.root_length_density * available * thickness * 1000.0
        return min(demand, uptake)

    def profile_water_uptake(self, layers: List[RootLayerState], demand: float) -> float:
        """Total water uptake (mm) across a soil profile given demand.

        Demand is distributed to layers proportionally to their uptake
        capacity, then each layer takes its share (capped at its capacity).
        """
        if demand < 0:
            raise ValueError("Demand must be non-negative")
        if not layers or demand == 0.0:
            return 0.0

        capacities = [self.layer_water_uptake(l, demand) for l in layers]
        total_capacity = sum(capacities)
        if total_capacity <= 0.0:
            return 0.0

        # Scale demand across layers by relative capacity.
        scale = min(1.0, demand / total_capacity)
        return sum(c * scale for c in capacities)

    def layer_uptake(self, layer: RootLayerState, demand: float) -> float:
        """Alias for layer_water_uptake (per-layer share of demand)."""
        return self.layer_water_uptake(layer, demand)

    def simulate(
        self,
        days: int,
        daily_water_demand: float,
        initial_moisture: float = 0.30,
    ) -> RootGrowthResult:
        """Simulate root growth and water uptake over time.

        Args:
            days: Number of days to simulate.
            daily_water_demand: Crop water demand (mm/day).
            initial_moisture: Starting soil moisture (m³/m³), uniform.

        Returns:
            RootGrowthResult with history and aggregate uptake.
        """
        if days < 0:
            raise ValueError("Days must be non-negative")
        if daily_water_demand < 0:
            raise ValueError("Daily water demand must be non-negative")

        history: List[RootGrowthState] = []
        total_uptake = 0.0
        max_depth_seen = self.initial_root_depth

        for day in range(days):
            depth = self.root_depth_at_day(day)
            max_depth_seen = max(max_depth_seen, depth)
            layers = self._build_layers(depth, initial_moisture)
            state = RootGrowthState(
                root_depth=depth,
                layers=layers,
                total_root_length=self._total_root_length(layers),
            )
            history.append(state)
            total_uptake += self.profile_water_uptake(layers, daily_water_demand)

        final_depth = self.root_depth_at_day(days)
        final_layers = self._build_layers(final_depth, initial_moisture)
        final_state = RootGrowthState(
            root_depth=final_depth,
            layers=final_layers,
            total_root_length=self._total_root_length(final_layers),
        )

        return RootGrowthResult(
            days_simulated=days,
            final_state=final_state,
            history=history,
            total_water_uptake=total_uptake,
            max_root_depth=max_depth_seen,
        )

    def _build_layers(self, root_depth: float, moisture: float) -> List[RootLayerState]:
        """Build n_layers covering [0, root_depth] with decaying root density."""
        thickness = root_depth / self.n_layers
        layers: List[RootLayerState] = []
        for i in range(self.n_layers):
            mid_depth = (i + 0.5) * thickness
            rld = self.root_distribution(mid_depth)
            layers.append(
                RootLayerState(
                    depth_top=i * thickness,
                    depth_bottom=(i + 1) * thickness,
                    root_length_density=rld,
                    soil_moisture=moisture,
                )
            )
        return layers

    def _total_root_length(self, layers: List[RootLayerState]) -> float:
        """Integrated root length density over the profile (cm/cm²)."""
        return sum(l.root_length_density * (l.depth_bottom - l.depth_top) * 100.0 for l in layers)
