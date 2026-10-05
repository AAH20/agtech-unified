"""Multi-agent module: swarm coordination, consensus, task allocation, and fault tolerance."""

from src.multi_agent.coalition import (
    AgentProfile,
    Coalition,
    CoalitionFormation,
    CoalitionStatus,
    TaskRequirement,
)
from src.multi_agent.collision_avoidance import CollisionAvoidance, Position, Velocity
from src.multi_agent.consensus import ByzantineConsensus, ConsensusResult, ConsensusStatus
from src.multi_agent.dwa import DWAConfig, DynamicWindowApproach, Trajectory
from src.multi_agent.fault_tolerance import AgentInfo, LeaderElection, TaskInfo, TaskReassignment
from src.multi_agent.flocking import FlockingBehavior
from src.multi_agent.swarm import Agent, AgentStatus, SwarmCoordinator, Task, TaskStatus
from src.multi_agent.task_allocation import AllocationInstance, AllocationResult, TaskAllocator
from src.multi_agent.token_ring import TokenRing, TokenRingError, TokenRingState

__all__ = [
    "AgentStatus",
    "TaskStatus",
    "Agent",
    "Task",
    "SwarmCoordinator",
    "ConsensusStatus",
    "ConsensusResult",
    "ByzantineConsensus",
    "AllocationInstance",
    "AllocationResult",
    "TaskAllocator",
    "Position",
    "Velocity",
    "CollisionAvoidance",
    "AgentInfo",
    "TaskInfo",
    "TaskReassignment",
    "LeaderElection",
    "FlockingBehavior",
    "DWAConfig",
    "DynamicWindowApproach",
    "Trajectory",
    "AgentProfile",
    "Coalition",
    "CoalitionFormation",
    "CoalitionStatus",
    "TaskRequirement",
    "TokenRing",
    "TokenRingError",
    "TokenRingState",
]
