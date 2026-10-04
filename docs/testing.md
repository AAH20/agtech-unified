# Testing Guide

## Overview

This guide covers the testing strategy for agtech-unified, including test organization, writing tests, and running them.

## Test Structure

```
tests/
├── test_*.py              # Unit and integration tests
├── conftest.py            # Shared fixtures
└── __init__.py            # Test package marker
```

## Running Tests

### All Tests

```bash
pytest
```

### Specific Module

```bash
pytest tests/test_tsp.py
```

### With Coverage

```bash
pytest --cov=src --cov-report=html
```

### With Verbose Output

```bash
pytest -v
```

## Writing Tests

### Unit Tests

Test individual functions and classes in isolation:

```python
def test_tsp_solver_basic():
    from src.optimization.tsp import TSPInstance, TSPSolver
    cities = ["A", "B", "C"]
    dist = [[0, 10, 15], [10, 0, 20], [15, 20, 0]]
    result = TSPSolver("christofides").solve(TSPInstance(cities, dist))
    assert result.cost > 0
    assert len(result.tour) == 3
```

### Integration Tests

Test interactions between modules:

```python
def test_unified_optimizer_tsp():
    from src.integration.unified_optimizer import UnifiedOptimizer, SolverType, OptimizationProblem
    optimizer = UnifiedOptimizer()
    problem = OptimizationProblem(
        solver_type=SolverType.TSP,
        cities=["A", "B", "C"],
        distance_matrix=[[0, 10, 15], [10, 0, 20], [15, 20, 0]],
    )
    result = optimizer.solve(problem)
    assert result.cost > 0
```

### Property-Based Tests

Use Hypothesis for property-based testing:

```python
from hypothesis import given, strategies as st

@given(st.lists(st.integers(min_value=0, max_value=100), min_size=3, max_size=10))
def test_tsp_tour_length(cities):
    from src.optimization.tsp import TSPInstance, TSPSolver
    n = len(cities)
    dist = [[abs(cities[i] - cities[j]) for j in range(n)] for i in range(n)]
    result = TSPSolver("nearest_neighbor").solve(TSPInstance([str(c) for c in cities], dist))
    assert len(result.tour) == n
```

## Test Fixtures

Use pytest fixtures for shared setup:

```python
import pytest

@pytest.fixture
def sample_farm_state():
    from src.decision_support.recommender import FarmState
    return FarmState(
        soil_moisture=0.5,
        temperature=25.0,
        crop_height=0.5,
        nutrient_level=0.7,
        pest_pressure=0.2,
    )

def test_decision_engine(sample_farm_state):
    from src.decision_support.recommender import DecisionEngine
    engine = DecisionEngine("rule_based")
    result = engine.recommend(sample_farm_state)
    assert result.priority_score >= 0
```

## Mocking External Services

Mock external services (MQTT, Kafka, databases) in tests:

```python
from unittest.mock import MagicMock, patch

def test_mqtt_client_publish():
    from src.iot.data_pipeline import MQTTClient
    client = MQTTClient(broker_url="localhost")
    with patch.object(client, "publish") as mock_publish:
        client.publish("topic", "message")
        mock_publish.assert_called_once_with("topic", "message")
```

## Coverage Goals

- **Target**: 90%+ code coverage
- **Critical paths**: 100% coverage
- **Minimum**: 80% coverage

## Continuous Integration

Tests run automatically on:
- Every push to `main`
- Every pull request
- Nightly builds

## Best Practices

1. **Test behavior, not implementation**: Assert on outputs, not internal state
2. **Use descriptive names**: `test_tsp_solver_returns_valid_tour` not `test_tsp_1`
3. **Keep tests fast**: Unit tests should complete in <1s
4. **Isolate tests**: Each test should be independent
5. **Use fixtures**: Share setup code with pytest fixtures
6. **Test edge cases**: Empty inputs, invalid inputs, boundary values
