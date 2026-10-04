# Tutorial 05: Digital Twin Simulation

## Overview

Learn how to create and run a digital twin simulation of your farm.

## Basic Simulation

```python
from src.digital_twin.simulator import SimulationState, DigitalTwin

# Initial state
state = SimulationState(
    soil_moisture=0.6,
    temperature=25.0,
    crop_height=0.1,
    nutrient_level=0.8,
)

# Run simulation
twin = DigitalTwin(model="logistic_growth")
result = twin.simulate(state, days=30)

print(f"Final height: {result.final_state.crop_height:.2f}m")
print(f"Days simulated: {result.days_simulated}")
```

## Knowledge Graph

```python
from src.digital_twin.knowledge_graph import AgriKnowledgeGraph

kg = AgriKnowledgeGraph()
kg.add_entity("Field-A", type="Field", area=10.5)
kg.add_entity("Crop-Wheat", type="Crop", variety="Winter Wheat")
kg.add_relationship("Field-A", "grows", "Crop-Wheat")

results = kg.query("Field-A")
print(f"Field-A relationships: {results}")
```

## Next Steps

- [Tutorial 06: Decision Support](06-decision-support.md)
- [Tutorial 07: API Gateway](07-api-gateway.md)
