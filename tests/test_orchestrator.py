"""Tests for unified digital twin orchestrator."""

from src.digital_twin.event_engine import Event, EventType
from src.digital_twin.orchestrator import DigitalTwinOrchestrator, OrchestratorConfig
from src.digital_twin.shared_state import DigitalTwinState


class TestOrchestratorConfig:
    """Test orchestrator configuration."""

    def test_default_config(self):
        """Default configuration values."""
        config = OrchestratorConfig()
        assert config.enable_water_balance is True
        assert config.enable_nutrient_cycling is True
        assert config.enable_event_driven is True
        assert config.enable_sensor_ingestion is True

    def test_custom_config(self):
        """Custom configuration."""
        config = OrchestratorConfig(
            enable_water_balance=False,
            enable_nutrient_cycling=False,
        )
        assert config.enable_water_balance is False
        assert config.enable_nutrient_cycling is False


class TestDigitalTwinOrchestrator:
    """Test unified orchestrator facade."""

    def test_create_orchestrator(self):
        """Create orchestrator with default config."""
        orch = DigitalTwinOrchestrator()
        assert orch is not None
        assert orch.state is not None

    def test_create_with_config(self):
        """Create orchestrator with custom config."""
        config = OrchestratorConfig(enable_water_balance=False)
        orch = DigitalTwinOrchestrator(config=config)
        assert orch.config.enable_water_balance is False

    def test_get_state(self):
        """Get current state."""
        orch = DigitalTwinOrchestrator()
        state = orch.get_state()
        assert isinstance(state, DigitalTwinState)

    def test_set_state(self):
        """Set state."""
        orch = DigitalTwinOrchestrator()
        new_state = DigitalTwinState(
            soil_moisture=0.4,
            temperature=28.0,
            crop_height=0.8,
        )
        orch.set_state(new_state)
        assert orch.get_state().soil_moisture == 0.4

    def test_step(self):
        """Advance simulation by one day."""
        orch = DigitalTwinOrchestrator()
        orch.get_state().crop_height
        orch.step()
        # State should have changed
        assert orch.get_state() is not None

    def test_step_n(self):
        """Advance simulation by N days."""
        orch = DigitalTwinOrchestrator()
        orch.step_n(5)
        assert orch.get_state() is not None

    def test_simulate(self):
        """Run full simulation."""
        orch = DigitalTwinOrchestrator()
        result = orch.simulate(days=10)
        assert result is not None
        assert result.get("days_simulated") == 10

    def test_simulate_with_events(self):
        """Run simulation with events."""
        orch = DigitalTwinOrchestrator()
        events = [
            Event(event_type=EventType.IRRIGATION, day=3, payload={"amount_mm": 20.0}),
            Event(event_type=EventType.FERTILIZATION, day=5, payload={"nitrogen": 30.0}),
        ]
        result = orch.simulate(days=10, events=events)
        assert result is not None

    def test_add_sensor_reading(self):
        """Add sensor reading through orchestrator."""
        orch = DigitalTwinOrchestrator()
        reading = SensorReading(
            sensor_id="soil_01",
            sensor_type=SensorType.SOIL_MOISTURE,
            value=0.35,
            timestamp=1696156800.0,
        )
        orch.add_sensor_reading(reading)
        # State should be reconciled
        assert orch.get_state().soil_moisture == 0.35

    def test_get_submodule(self):
        """Get submodule by name."""
        orch = DigitalTwinOrchestrator()
        # Trigger lazy initialization
        orch.step()
        sim = orch.get_submodule("simulator")
        assert sim is not None

    def test_get_water_balance(self):
        """Get water balance model."""
        orch = DigitalTwinOrchestrator()
        wb = orch.get_submodule("water_balance")
        assert wb is not None

    def test_get_nutrient_cycling(self):
        """Get nutrient cycling model."""
        orch = DigitalTwinOrchestrator()
        nc = orch.get_submodule("nutrient_cycling")
        assert nc is not None

    def test_get_knowledge_graph(self):
        """Get knowledge graph."""
        orch = DigitalTwinOrchestrator()
        kg = orch.get_submodule("knowledge_graph")
        assert kg is not None

    def test_get_ontology(self):
        """Get ontology."""
        orch = DigitalTwinOrchestrator()
        ont = orch.get_submodule("ontology")
        assert ont is not None

    def test_get_event_bus(self):
        """Get event bus."""
        orch = DigitalTwinOrchestrator()
        bus = orch.get_submodule("event_bus")
        assert bus is not None

    def test_publish_event(self):
        """Publish event through orchestrator."""
        orch = DigitalTwinOrchestrator()
        event = Event(
            event_type=EventType.IRRIGATION,
            day=0,
            payload={"amount_mm": 20.0},
        )
        orch.publish_event(event)

    def test_subscribe_to_events(self):
        """Subscribe to events through orchestrator."""
        orch = DigitalTwinOrchestrator()
        received = []

        def handler(event):
            received.append(event)

        orch.subscribe(EventType.SIMULATION_COMPLETED, handler)
        orch.simulate(days=1)
        # Event should have been published
        assert len(received) > 0

    def test_reset(self):
        """Reset orchestrator state."""
        orch = DigitalTwinOrchestrator()
        orch.step_n(5)
        orch.reset()
        assert orch.get_state().crop_height == 0.0

    def test_get_history(self):
        """Get simulation history."""
        orch = DigitalTwinOrchestrator()
        orch.step_n(5)
        history = orch.get_history()
        assert len(history) == 5

    def test_export_state(self):
        """Export state to dict."""
        orch = DigitalTwinOrchestrator()
        d = orch.export_state()
        assert "soil_moisture" in d
        assert "temperature" in d

    def test_import_state(self):
        """Import state from dict."""
        orch = DigitalTwinOrchestrator()
        d = {
            "soil_moisture": 0.4,
            "temperature": 28.0,
            "crop_height": 0.8,
            "nitrogen": 50.0,
        }
        orch.import_state(d)
        assert orch.get_state().soil_moisture == 0.4
        assert orch.get_state().nitrogen == 50.0

    def test_run_scenario(self):
        """Run a named scenario."""
        orch = DigitalTwinOrchestrator()
        result = orch.run_scenario("default", days=10)
        assert result is not None

    def test_compare_scenarios(self):
        """Compare two scenarios."""
        orch = DigitalTwinOrchestrator()
        result1 = orch.run_scenario("default", days=10)
        result2 = orch.run_scenario("high_input", days=10)
        comparison = orch.compare_scenarios(result1, result2)
        assert comparison is not None

    def test_health_check(self):
        """Health check returns status."""
        orch = DigitalTwinOrchestrator()
        health = orch.health_check()
        assert "status" in health
        assert health["status"] == "healthy"

    def test_get_metrics(self):
        """Get simulation metrics."""
        orch = DigitalTwinOrchestrator()
        orch.step_n(5)
        metrics = orch.get_metrics()
        assert "days_simulated" in metrics


# Import at bottom to avoid circular import issues in test collection
from src.digital_twin.sensor_ingestion import SensorReading, SensorType
