# ADR 0002: PBFT Consensus Simulation

## Status

Accepted

## Context

The multi-agent swarm coordination module needs a consensus mechanism for distributed decision-making. We need to choose between:
- Practical Byzantine Fault Tolerance (PBFT)
- Raft (crash fault tolerance only)
- Proof-of-Work (too slow for real-time coordination)

## Decision

We will implement a **PBFT consensus simulation** that models the three-phase commit protocol (pre-prepare, prepare, commit) with configurable Byzantine node behavior.

### Rationale

1. **Byzantine fault tolerance**: Agricultural IoT devices may be compromised or malfunction in unpredictable ways
2. **Real-time performance**: PBFT has lower latency than PoW and is suitable for real-time swarm coordination
3. **Simulation-first**: We simulate the protocol rather than implementing a full distributed system, keeping the codebase manageable

### Consequences

- **Positive**: Models real-world Byzantine failures, deterministic testing, educational value
- **Negative**: Simulation does not cover network partitions, clock synchronization, or real message passing
- **Mitigation**: Document limitations and provide hooks for real network implementation

## Alternatives Considered

1. **Raft**: Simpler but does not handle Byzantine faults
2. **PoW**: Too slow for real-time coordination
3. **Full PBFT implementation**: Too complex for the initial release

## References

- `src/multi_agent/consensus.py` — ByzantineConsensus implementation
- `tests/test_consensus.py` — Consensus tests
