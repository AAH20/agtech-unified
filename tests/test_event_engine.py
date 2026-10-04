"""Tests for event-driven simulation engine."""

from src.digital_twin.event_engine import (
    DiscreteEventSimulator,
    Event,
    EventQueue,
    EventType,
    SimulationEvent,
)


class TestEvent:
    """Test event data structures."""

    def test_create_event(self):
        """Create a basic event."""
        event = Event(
            event_type=EventType.IRRIGATION,
            day=5,
            payload={"amount_mm": 20.0},
        )
        assert event.event_type == EventType.IRRIGATION
        assert event.day == 5
        assert event.payload["amount_mm"] == 20.0

    def test_event_ordering(self):
        """Events ordered by day then priority."""
        e1 = Event(event_type=EventType.IRRIGATION, day=5, priority=1)
        e2 = Event(event_type=EventType.IRRIGATION, day=3, priority=5)
        e3 = Event(event_type=EventType.IRRIGATION, day=5, priority=3)
        events = sorted([e1, e2, e3])
        assert events[0].day == 3
        assert events[1].priority == 1
        assert events[2].priority == 3

    def test_simulation_event(self):
        """SimulationEvent with handler callback."""

        def handler(state):
            state["irrigated"] = True
            return state

        event = SimulationEvent(
            event_type=EventType.IRRIGATION,
            day=5,
            handler=handler,
        )
        assert event.handler is not None
        result = event.apply({"soil_moisture": 0.2})
        assert result["irrigated"] is True


class TestEventQueue:
    """Test priority event queue."""

    def test_push_pop(self):
        """Push and pop events in order."""
        queue = EventQueue()
        queue.push(Event(event_type=EventType.IRRIGATION, day=5))
        queue.push(Event(event_type=EventType.FERTILIZATION, day=3))
        queue.push(Event(event_type=EventType.PEST_OUTBREAK, day=7))

        assert queue.size() == 3
        first = queue.pop()
        assert first.day == 3
        assert first.event_type == EventType.FERTILIZATION

    def test_peek(self):
        """Peek at next event without removing."""
        queue = EventQueue()
        queue.push(Event(event_type=EventType.IRRIGATION, day=5))
        peeked = queue.peek()
        assert peeked.day == 5
        assert queue.size() == 1

    def test_empty_queue(self):
        """Empty queue returns None on pop."""
        queue = EventQueue()
        assert queue.pop() is None
        assert queue.peek() is None
        assert queue.is_empty()

    def test_clear(self):
        """Clear all events."""
        queue = EventQueue()
        queue.push(Event(event_type=EventType.IRRIGATION, day=5))
        queue.clear()
        assert queue.is_empty()

    def test_events_for_day(self):
        """Get all events for a specific day."""
        queue = EventQueue()
        queue.push(Event(event_type=EventType.IRRIGATION, day=5))
        queue.push(Event(event_type=EventType.FERTILIZATION, day=5))
        queue.push(Event(event_type=EventType.PEST_OUTBREAK, day=7))
        day5 = queue.events_for_day(5)
        assert len(day5) == 2


class TestDiscreteEventSimulator:
    """Test discrete event simulator."""

    def test_simulate_with_events(self):
        """Simulate with event injection."""
        sim = DiscreteEventSimulator()
        sim.schedule(Event(event_type=EventType.IRRIGATION, day=3, payload={"amount_mm": 20.0}))
        sim.schedule(Event(event_type=EventType.FERTILIZATION, day=5, payload={"nitrogen": 50.0}))

        # Track state changes
        states = []

        def tracker(day, state):
            states.append((day, state.get("soil_moisture", 0)))

        result = sim.simulate(days=10, initial_state={"soil_moisture": 0.2}, callback=tracker)
        assert result["days_simulated"] == 10
        assert len(states) == 10

    def test_irrigation_event_increases_moisture(self):
        """Irrigation event increases soil moisture."""
        sim = DiscreteEventSimulator()
        sim.schedule(Event(event_type=EventType.IRRIGATION, day=0, payload={"amount_mm": 30.0}))

        final = sim.simulate(days=1, initial_state={"soil_moisture": 0.2})
        assert final["final_state"]["soil_moisture"] > 0.2

    def test_fertilization_event_increases_nutrients(self):
        """Fertilization event increases nutrient level."""
        sim = DiscreteEventSimulator()
        sim.schedule(
            Event(
                event_type=EventType.FERTILIZATION,
                day=0,
                payload={"nitrogen": 50.0, "phosphorus": 10.0, "potassium": 40.0},
            )
        )

        final = sim.simulate(days=1, initial_state={"nutrient_level": 0.3})
        assert final["final_state"]["nutrient_level"] > 0.3

    def test_pest_outbreak_event(self):
        """Pest outbreak event increases pest pressure."""
        sim = DiscreteEventSimulator()
        sim.schedule(Event(event_type=EventType.PEST_OUTBREAK, day=0, payload={"severity": 0.5}))

        final = sim.simulate(days=1, initial_state={"pest_pressure": 0.0})
        assert final["final_state"]["pest_pressure"] > 0.0

    def test_no_events(self):
        """Simulation with no events runs normally."""
        sim = DiscreteEventSimulator()
        result = sim.simulate(days=5, initial_state={"soil_moisture": 0.3})
        assert result["days_simulated"] == 5

    def test_event_history(self):
        """Events are recorded in history."""
        sim = DiscreteEventSimulator()
        sim.schedule(Event(event_type=EventType.IRRIGATION, day=2, payload={"amount_mm": 10.0}))
        sim.simulate(days=5, initial_state={"soil_moisture": 0.2})
        history = sim.get_event_history()
        assert len(history) == 1
        assert history[0].day == 2
