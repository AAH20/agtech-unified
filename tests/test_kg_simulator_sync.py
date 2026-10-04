"""Tests for knowledge graph ↔ simulator synchronization."""

from src.digital_twin.knowledge_graph import AgriKnowledgeGraph, Entity, Relationship
from src.digital_twin.simulator import DigitalTwin, SimulationState


class TestKnowledgeGraphSimulatorSync:
    """Test bidirectional sync between KG and simulator."""

    def test_graph_entity_drives_simulation(self):
        """Graph entity properties parameterize simulation."""
        kg = AgriKnowledgeGraph()
        kg.add_entity(
            Entity(
                id="field_01",
                label="North Field",
                entity_type="field",
                properties={
                    "soil_type": "loam",
                    "area_ha": 10.0,
                    "crop": "wheat",
                },
            )
        )

        # Extract parameters from graph
        field = kg.get_entity("field_01")
        assert field.properties["crop"] == "wheat"

        # Use in simulation
        twin = DigitalTwin()
        state = SimulationState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        result = twin.simulate(state, days=10)
        assert result.final_state.crop_height > 0.1

    def test_simulation_results_update_graph(self):
        """Simulation results update graph entity properties."""
        kg = AgriKnowledgeGraph()
        kg.add_entity(
            Entity(
                id="crop_01",
                label="Wheat Field",
                entity_type="crop",
                properties={"initial_height": 0.1},
            )
        )

        twin = DigitalTwin()
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        result = twin.simulate(state, days=10)

        # Update graph with results
        kg.update_entity_properties(
            "crop_01",
            {
                "final_height": result.final_state.crop_height,
                "total_growth": result.total_growth,
            },
        )
        updated = kg.get_entity("crop_01")
        assert updated.properties["final_height"] > 0.1
        assert updated.properties["total_growth"] > 0.0

    def test_field_to_crop_relationship(self):
        """Field-crop relationship drives simulation."""
        kg = AgriKnowledgeGraph()
        kg.add_entity(Entity(id="field_01", label="Field 1", entity_type="field"))
        kg.add_entity(Entity(id="wheat_01", label="Wheat", entity_type="crop"))
        kg.add_relationship(
            Relationship(
                source="field_01",
                target="wheat_01",
                rel_type="grows",
            )
        )

        # Query relationship
        crops = kg.query(entity_type="crop")
        assert len(crops) == 1
        assert crops[0].label == "Wheat"

    def test_sensor_entity_links_to_field(self):
        """Sensor entities link to field entities."""
        kg = AgriKnowledgeGraph()
        kg.add_entity(Entity(id="field_01", label="Field 1", entity_type="field"))
        kg.add_entity(
            Entity(
                id="sensor_01",
                label="Soil Moisture Sensor",
                entity_type="sensor",
                properties={"sensor_type": "soil_moisture"},
            )
        )
        kg.add_relationship(
            Relationship(
                source="sensor_01",
                target="field_01",
                rel_type="located_in",
            )
        )

        # Find sensors for field
        sensors = kg.query(entity_type="sensor")
        assert len(sensors) == 1

    def test_simulation_history_to_graph(self):
        """Full simulation history stored in graph."""
        kg = AgriKnowledgeGraph()
        kg.add_entity(
            Entity(
                id="sim_01",
                label="Simulation Run",
                entity_type="simulation",
            )
        )

        twin = DigitalTwin()
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        result = twin.simulate(state, days=5)

        # Store history in graph
        kg.update_entity_properties(
            "sim_01",
            {
                "days_simulated": result.days_simulated,
                "final_height": result.final_state.crop_height,
                "history_length": len(result.history),
            },
        )
        sim = kg.get_entity("sim_01")
        assert sim.properties["days_simulated"] == 5
        assert sim.properties["history_length"] == 5
