# ADR 0004: FastAPI for API Gateway

## Status

Accepted

## Context

The decision support module needs a REST API gateway to expose the platform's capabilities to external clients. We need to choose a web framework.

## Decision

We will use **FastAPI** as the API gateway framework.

### Rationale

1. **Async support**: Native async/await for handling concurrent IoT connections
2. **Automatic docs**: OpenAPI/Swagger documentation generated automatically
3. **Type safety**: Pydantic models for request/response validation
4. **Performance**: Competitive with Node.js and Go for I/O-bound workloads
5. **Python ecosystem**: Consistent with the rest of the codebase

### Consequences

- **Positive**: Fast development, automatic documentation, type safety
- **Negative**: Requires understanding of async Python, newer framework with smaller community than Flask
- **Mitigation**: Comprehensive examples and documentation

## Alternatives Considered

1. **Flask**: Synchronous by default, no automatic docs
2. **Django**: Too heavy for a microservice API gateway
3. **aiohttp**: Lower-level, no automatic docs

## References

- `src/decision_support/api_gateway.py` — APIGateway implementation
- `tests/test_api_gateway.py` — API gateway tests
