"""Test multi-agent coalition formation for multi-capability tasks."""

from src.multi_agent.coalition import (
    AgentProfile,
    Coalition,
    CoalitionFormation,
    CoalitionStatus,
    TaskRequirement,
)

# ── MA-CF-001: Agent profile and task requirement ─────────────────────


def test_agent_profile_creation():
    """AgentProfile stores agent capabilities."""
    profile = AgentProfile(agent_id="A1", capabilities=["spray", "scan"])
    assert profile.agent_id == "A1"
    assert "spray" in profile.capabilities
    assert "scan" in profile.capabilities


def test_task_requirement_creation():
    """TaskRequirement stores required capabilities."""
    req = TaskRequirement(task_id="T1", required_capabilities=["spray", "harvest"])
    assert req.task_id == "T1"
    assert "spray" in req.required_capabilities


# ── MA-CF-002: Coalition formation ────────────────────────────────────


def test_form_coalition_single_agent_sufficient():
    """Single agent with all required capabilities forms coalition."""
    agents = [AgentProfile("A1", ["spray", "scan"])]
    tasks = [TaskRequirement("T1", ["spray", "scan"])]
    cf = CoalitionFormation(agents=agents, tasks=tasks)
    coalitions = cf.form_coalitions()
    assert len(coalitions) == 1
    assert "A1" in coalitions[0].members


def test_form_coalition_multiple_agents_needed():
    """Multiple agents form coalition when no single agent has all capabilities."""
    agents = [
        AgentProfile("A1", ["spray"]),
        AgentProfile("A2", ["scan"]),
    ]
    tasks = [TaskRequirement("T1", ["spray", "scan"])]
    cf = CoalitionFormation(agents=agents, tasks=tasks)
    coalitions = cf.form_coalitions()
    assert len(coalitions) == 1
    members = coalitions[0].members
    assert "A1" in members
    assert "A2" in members


def test_form_coalition_no_capable_agents():
    """No coalition formed when no agents have required capabilities."""
    agents = [AgentProfile("A1", ["harvest"])]
    tasks = [TaskRequirement("T1", ["spray", "scan"])]
    cf = CoalitionFormation(agents=agents, tasks=tasks)
    coalitions = cf.form_coalitions()
    assert len(coalitions) == 0


def test_form_coalition_empty_agents():
    """No coalitions formed with no agents."""
    tasks = [TaskRequirement("T1", ["spray"])]
    cf = CoalitionFormation(agents=[], tasks=tasks)
    coalitions = cf.form_coalitions()
    assert len(coalitions) == 0


def test_form_coalition_empty_tasks():
    """No coalitions formed with no tasks."""
    agents = [AgentProfile("A1", ["spray"])]
    cf = CoalitionFormation(agents=agents, tasks=[])
    coalitions = cf.form_coalitions()
    assert len(coalitions) == 0


# ── MA-CF-003: Coalition capability coverage ──────────────────────────


def test_coalition_covers_all_required_capabilities():
    """Formed coalition covers all task requirements."""
    agents = [
        AgentProfile("A1", ["spray"]),
        AgentProfile("A2", ["scan"]),
        AgentProfile("A3", ["harvest"]),
    ]
    tasks = [TaskRequirement("T1", ["spray", "scan"])]
    cf = CoalitionFormation(agents=agents, tasks=tasks)
    coalitions = cf.form_coalitions()
    assert len(coalitions) == 1
    coalition = coalitions[0]
    covered = coalition.get_covered_capabilities()
    assert "spray" in covered
    assert "scan" in covered


def test_coalition_does_not_include_unnecessary_agents():
    """Coalition only includes agents that contribute required capabilities."""
    agents = [
        AgentProfile("A1", ["spray"]),
        AgentProfile("A2", ["scan"]),
        AgentProfile("A3", ["harvest"]),
    ]
    tasks = [TaskRequirement("T1", ["spray", "scan"])]
    cf = CoalitionFormation(agents=agents, tasks=tasks)
    coalitions = cf.form_coalitions()
    coalition = coalitions[0]
    # A3 has harvest which is not needed
    assert "A3" not in coalition.members


# ── MA-CF-004: Multiple task coalitions ───────────────────────────────


def test_form_multiple_coalitions():
    """Multiple tasks can form separate coalitions."""
    agents = [
        AgentProfile("A1", ["spray"]),
        AgentProfile("A2", ["scan"]),
        AgentProfile("A3", ["harvest"]),
        AgentProfile("A4", ["plant"]),
    ]
    tasks = [
        TaskRequirement("T1", ["spray", "scan"]),
        TaskRequirement("T2", ["harvest", "plant"]),
    ]
    cf = CoalitionFormation(agents=agents, tasks=tasks)
    coalitions = cf.form_coalitions()
    assert len(coalitions) == 2


def test_form_coalitions_shared_agent():
    """Agent can only be in one coalition at a time."""
    agents = [
        AgentProfile("A1", ["spray", "harvest"]),
        AgentProfile("A2", ["scan"]),
    ]
    tasks = [
        TaskRequirement("T1", ["spray", "scan"]),
        TaskRequirement("T2", ["harvest"]),
    ]
    cf = CoalitionFormation(agents=agents, tasks=tasks)
    coalitions = cf.form_coalitions()
    # A1 can only be in one coalition
    all_members = []
    for c in coalitions:
        all_members.extend(c.members)
    # No duplicate memberships
    assert len(all_members) == len(set(all_members))


# ── MA-CF-005: Coalition status ───────────────────────────────────────


def test_coalition_status_active():
    """Coalition status is ACTIVE after formation."""
    agents = [AgentProfile("A1", ["spray"])]
    tasks = [TaskRequirement("T1", ["spray"])]
    cf = CoalitionFormation(agents=agents, tasks=tasks)
    coalitions = cf.form_coalitions()
    assert coalitions[0].status == CoalitionStatus.ACTIVE


def test_coalition_status_dissolved():
    """Coalition can be dissolved."""
    agents = [AgentProfile("A1", ["spray"])]
    tasks = [TaskRequirement("T1", ["spray"])]
    cf = CoalitionFormation(agents=agents, tasks=tasks)
    coalitions = cf.form_coalitions()
    coalition = coalitions[0]
    coalition.dissolve()
    assert coalition.status == CoalitionStatus.DISSOLVED


# ── MA-CF-006: Coalition merge and split ─────────────────────────────


def test_merge_coalitions():
    """Two coalitions can be merged into one."""
    c1 = Coalition(coalition_id="C1", members=["A1"])
    c2 = Coalition(coalition_id="C2", members=["A2"])
    merged = c1.merge(c2)
    assert "A1" in merged.members
    assert "A2" in merged.members


def test_split_coalition():
    """Coalition can be split, removing a member."""
    agents = [
        AgentProfile("A1", ["spray"]),
        AgentProfile("A2", ["scan"]),
    ]
    tasks = [TaskRequirement("T1", ["spray", "scan"])]
    cf = CoalitionFormation(agents=agents, tasks=tasks)
    coalitions = cf.form_coalitions()
    coalition = coalitions[0]
    coalition.split("A2")
    assert "A2" not in coalition.members
    assert "A1" in coalition.members


# ── MA-CF-007: Coalition with agent failure ───────────────────────────


def test_handle_agent_failure_in_coalition():
    """Failed agent is removed from coalition."""
    agents = [
        AgentProfile("A1", ["spray"]),
        AgentProfile("A2", ["scan"]),
    ]
    tasks = [TaskRequirement("T1", ["spray", "scan"])]
    cf = CoalitionFormation(agents=agents, tasks=tasks)
    coalitions = cf.form_coalitions()
    coalition = coalitions[0]
    coalition.handle_agent_failure("A1")
    assert "A1" not in coalition.members
    assert "A2" in coalition.members


def test_handle_agent_failure_dissolves_coalition():
    """Coalition is dissolved when all members fail."""
    agents = [AgentProfile("A1", ["spray"])]
    tasks = [TaskRequirement("T1", ["spray"])]
    cf = CoalitionFormation(agents=agents, tasks=tasks)
    coalitions = cf.form_coalitions()
    coalition = coalitions[0]
    coalition.handle_agent_failure("A1")
    assert coalition.status == CoalitionStatus.DISSOLVED


# ── MA-CF-008: Coalition utility and cost ─────────────────────────────


def test_coalition_utility():
    """Coalition utility is based on capability coverage."""
    coalition = Coalition(coalition_id="C1", members=["A1", "A2"])
    coalition._capabilities = {"spray", "scan"}
    utility = coalition.get_utility(["spray", "scan"])
    assert utility > 0


def test_coalition_cost():
    """Coalition cost is the sum of member costs."""
    agents = [
        AgentProfile("A1", ["spray"]),
        AgentProfile("A2", ["scan"]),
    ]
    tasks = [TaskRequirement("T1", ["spray", "scan"])]
    cf = CoalitionFormation(agents=agents, tasks=tasks)
    coalitions = cf.form_coalitions()
    coalition = coalitions[0]
    cost = coalition.get_cost()
    assert cost >= 0


# ── MA-CF-009: Coalition formation with priorities ────────────────────


def test_form_coalitions_with_priorities():
    """Higher priority tasks get coalitions first."""
    agents = [
        AgentProfile("A1", ["spray"]),
        AgentProfile("A2", ["scan"]),
    ]
    tasks = [
        TaskRequirement("T1", ["spray"], priority=1),
        TaskRequirement("T2", ["scan"], priority=2),
    ]
    cf = CoalitionFormation(agents=agents, tasks=tasks)
    coalitions = cf.form_coalitions()
    # T2 has higher priority, should get coalition first
    assert len(coalitions) == 2


# ── MA-CF-010: Coalition formation efficiency ─────────────────────────


def test_minimal_coalition_size():
    """Coalition formation minimizes number of agents per coalition."""
    agents = [
        AgentProfile("A1", ["spray", "scan"]),
        AgentProfile("A2", ["spray"]),
        AgentProfile("A3", ["scan"]),
    ]
    tasks = [TaskRequirement("T1", ["spray", "scan"])]
    cf = CoalitionFormation(agents=agents, tasks=tasks)
    coalitions = cf.form_coalitions()
    # A1 alone can cover both capabilities
    assert len(coalitions) == 1
    assert coalitions[0].members == ["A1"]


def test_coalition_formation_no_redundant_agents():
    """Coalition does not include agents with no useful capabilities."""
    agents = [
        AgentProfile("A1", ["spray"]),
        AgentProfile("A2", ["harvest"]),
        AgentProfile("A3", ["scan"]),
    ]
    tasks = [TaskRequirement("T1", ["spray", "scan"])]
    cf = CoalitionFormation(agents=agents, tasks=tasks)
    coalitions = cf.form_coalitions()
    coalition = coalitions[0]
    # A2 has harvest which is not needed
    assert "A2" not in coalition.members
