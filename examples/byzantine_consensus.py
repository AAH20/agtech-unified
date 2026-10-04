"""Byzantine consensus example: distributed decision-making with faulty nodes."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.multi_agent.consensus import ByzantineConsensus


def main():
    consensus = ByzantineConsensus(num_nodes=4, max_faults=1)
    result = consensus.propose(value="harvest-field-A")
    print(f"Consensus Status: {result.status}")
    print(f"Consensus Value: {result.value}")
    print(f"Votes: {result.votes}")


if __name__ == "__main__":
    main()
