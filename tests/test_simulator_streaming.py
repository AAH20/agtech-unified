"""Tests for streaming simulation output."""

from src.digital_twin.simulator import DigitalTwin, SimulationState


class TestStreamingOutput:
    """Test streaming simulation updates."""

    def test_stream_simulation(self):
        """Stream simulation results."""
        twin = DigitalTwin()
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        updates = list(twin.stream_simulate(state, days=5))
        assert len(updates) == 5

    def test_stream_includes_state(self):
        """Each stream update includes state."""
        twin = DigitalTwin()
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        for update in twin.stream_simulate(state, days=3):
            assert hasattr(update, "crop_height")
            assert hasattr(update, "soil_moisture")

    def test_stream_with_callback(self):
        """Stream with callback function."""
        twin = DigitalTwin()
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        received = []

        def callback(day, state):
            received.append((day, state.crop_height))

        # Consume the generator to trigger callbacks
        for _ in twin.stream_simulate(state, days=5, callback=callback):
            pass
        assert len(received) == 5

    def test_stream_zero_days(self):
        """Stream with zero days returns empty."""
        twin = DigitalTwin()
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        updates = list(twin.stream_simulate(state, days=0))
        assert len(updates) == 0

    def test_stream_growth_monotonic(self):
        """Crop height increases monotonically in stream."""
        twin = DigitalTwin()
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        heights = [u.crop_height for u in twin.stream_simulate(state, days=10)]
        for i in range(1, len(heights)):
            assert heights[i] >= heights[i - 1]
