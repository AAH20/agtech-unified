# ADR 0003: NGSI-LD Integration

## Status

Accepted

## Context

The IoT module needs to integrate with FIWARE ecosystem for interoperability with existing agricultural platforms. We need to choose an integration approach.

## Decision

We will implement **NGSI-LD context broker integration** using the FIWARE NGSI-LD API standard.

### Rationale

1. **Industry standard**: NGSI-LD is the FIWARE standard for context information management
2. **Agricultural adoption**: Many agricultural platforms (FIWARE, ADAPT) use NGSI-LD
3. **Semantic interoperability**: NGSI-LD supports linked data and semantic annotations

### Consequences

- **Positive**: Interoperability with FIWARE ecosystem, semantic data modeling
- **Negative**: Requires understanding of NGSI-LD data model, additional complexity
- **Mitigation**: Provide clear examples and documentation

## Alternatives Considered

1. **Custom JSON API**: Simpler but not interoperable
2. **MQTT only**: Lacks semantic modeling
3. **GraphQL**: Not widely adopted in agricultural IoT

## References

- `src/iot/context_broker.py` — NGSILDBroker implementation
- `src/iot/smart_models.py` — FIWARE smart data models
- `tests/test_context_broker.py` — Context broker tests
