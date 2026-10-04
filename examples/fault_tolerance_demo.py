"""Fault tolerance demo: task reassignment and leader election."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.multi_agent.fault_tolerance import AgentInfo, TaskInfo, TaskReassignment, LeaderElection


def main():
    agents = [
        AgentInfo(id="robot-1", status="active"),
        AgentInfo(id="robot-2", status="active"),
        AgentInfo(id="robot-3", status="failed"),
    ]
    tasks = [
        TaskInfo(id="weed-1", assigned_to="robot-3"),
    ]
    reassigner = TaskReassignment(agents)
    result = reassigner.reassign(tasks)
    print(f"Reassigned: {result.reassignments}")

    election = LeaderElection(agents)
    leader = election.elect()
    print(f"Leader: {leader}")


if __name__ == "__main__":
    main()
