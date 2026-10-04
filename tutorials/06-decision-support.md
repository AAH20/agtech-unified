# Tutorial 06: Decision Support

## Overview

Learn how to use the decision support system for agricultural recommendations.

## Basic Recommendations

```python
from src.decision_support.recommender import FarmState, DecisionEngine

# Current farm state
state = FarmState(
    soil_moisture=0.15,
    temperature=40.0,
    crop_height=0.05,
    nutrient_level=0.1,
    pest_pressure=0.8,
)

# Get recommendations
engine = DecisionEngine(strategy="rule_based")
result = engine.recommend(state)

for rec in result.recommendations:
    print(f"[{rec.priority}] {rec.action}: {rec.reason}")
```

## Alert Management

```python
from src.decision_support.alerts import AlertManager, Threshold

manager = AlertManager()
manager.add_threshold(Threshold(
    metric="soil_moisture",
    operator="<",
    value=0.2,
    severity="critical",
))

alerts = manager.check(state)
for alert in alerts:
    print(f"ALERT: {alert.message}")
```

## Next Steps

- [Tutorial 07: API Gateway](07-api-gateway.md)
- [Tutorial 08: Genomics](08-genomics.md)
