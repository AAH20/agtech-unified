# ADR 0001: In-Memory Backends

## Status

Accepted

## Context

The agtech-unified project needs to support multiple deployment scenarios:
- Development/testing with no external dependencies
- Production with real databases and message brokers
- Edge deployment with limited resources

We need to decide whether to use real backends (PostgreSQL, Kafka, MQTT) or in-memory implementations for the initial release.

## Decision

We will use **in-memory backends** as the default implementation, with interfaces that allow swapping to real backends later.

### Rationale

1. **Simplicity**: New users can run the entire system with `pip install -e .` and no external services
2. **Testing**: In-memory backends make tests fast and deterministic
3. **Edge deployment**: Resource-constrained devices (Jetson, Raspberry Pi) may not run full database clusters
4. **Modularity**: The interface-based design allows gradual migration to real backends

### Consequences

- **Positive**: Zero-config development, fast tests, works on edge devices
- **Negative**: Not suitable for production multi-node deployments without real backends
- **Mitigation**: Document the path to production backends in the deployment guide

## Alternatives Considered

1. **Require real backends**: Would increase setup complexity and exclude edge deployments
2. **Use SQLite as default**: Better than in-memory but still limited for time-series and message streaming
3. **Use Docker Compose for all deployments**: Too heavy for edge devices

## References

- `src/iot/data_pipeline.py` — MQTTClient, KafkaStream, TimescaleDBStorage interfaces
- `docs/deployment.md` — Production deployment guide
