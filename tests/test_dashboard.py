"""Test real-time farm monitoring dashboard."""

import asyncio

from fastapi.testclient import TestClient

from src.decision_support.dashboard import (
    DashboardConfig,
    FarmDashboard,
    SensorReading,
)


def test_dashboard_initial_state():
    """Dashboard initializes with default farm state."""
    dashboard = FarmDashboard()
    state = dashboard.state
    assert state.soil_moisture == 0.5
    assert state.temperature == 25.0
    assert state.crop_height == 0.3
    assert state.nutrient_level == 0.5
    assert state.pest_pressure == 0.1


def test_dashboard_custom_config():
    """Dashboard accepts custom configuration."""
    config = DashboardConfig(
        update_interval=2.0,
        max_history=500,
        farm_name="Test Farm",
    )
    dashboard = FarmDashboard(config=config)
    assert dashboard.config.farm_name == "Test Farm"
    assert dashboard.config.update_interval == 2.0
    assert dashboard.config.max_history == 500


def test_dashboard_update_state():
    """Dashboard state can be updated."""
    dashboard = FarmDashboard()
    dashboard.update_state(soil_moisture=0.8, temperature=30.0)
    assert dashboard.state.soil_moisture == 0.8
    assert dashboard.state.temperature == 30.0
    # Unchanged fields remain
    assert dashboard.state.crop_height == 0.3


def test_dashboard_state_dict():
    """Dashboard state can be serialized to dict."""
    dashboard = FarmDashboard()
    state_dict = dashboard.get_state_dict()
    assert "soil_moisture" in state_dict
    assert "temperature" in state_dict
    assert "crop_height" in state_dict
    assert "nutrient_level" in state_dict
    assert "pest_pressure" in state_dict
    assert "timestamp" in state_dict


def test_dashboard_history():
    """Dashboard tracks state history."""
    dashboard = FarmDashboard()
    dashboard.update_state(soil_moisture=0.1)
    dashboard.update_state(soil_moisture=0.2)
    dashboard.update_state(soil_moisture=0.3)
    history = dashboard.get_history()
    assert len(history) == 3
    assert history[-1]["state"]["soil_moisture"] == 0.3


def test_dashboard_history_limit():
    """Dashboard history respects limit parameter."""
    dashboard = FarmDashboard()
    for i in range(10):
        dashboard.update_state(soil_moisture=i * 0.1)
    history = dashboard.get_history(limit=3)
    assert len(history) == 3


def test_dashboard_client_management():
    """Dashboard tracks connected WebSocket clients."""
    dashboard = FarmDashboard()
    assert dashboard.client_count == 0

    class FakeWebSocket:
        pass

    ws1, ws2 = FakeWebSocket(), FakeWebSocket()
    dashboard.add_client(ws1)
    dashboard.add_client(ws2)
    assert dashboard.client_count == 2

    dashboard.remove_client(ws1)
    assert dashboard.client_count == 1


def test_dashboard_create_app():
    """Dashboard creates a FastAPI app with endpoints."""
    dashboard = FarmDashboard()
    app = dashboard.create_app()
    client = TestClient(app)

    response = client.get("/state")
    assert response.status_code == 200
    data = response.json()
    assert "soil_moisture" in data
    assert "temperature" in data


def test_dashboard_history_endpoint():
    """Dashboard history endpoint returns historical data."""
    dashboard = FarmDashboard()
    dashboard.update_state(soil_moisture=0.1)
    dashboard.update_state(soil_moisture=0.2)

    app = dashboard.create_app()
    client = TestClient(app)
    response = client.get("/history?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


def test_dashboard_stream_data():
    """Dashboard stream_data yields state updates."""
    dashboard = FarmDashboard(config=DashboardConfig(update_interval=0.01))

    async def collect():
        dashboard._running = True
        results = []
        async for state in dashboard.stream_data():
            results.append(state)
            if len(results) >= 3:
                dashboard.stop()
                break
        return results

    results = asyncio.run(collect())
    assert len(results) == 3
    assert all("soil_moisture" in r for r in results)


def test_dashboard_alert_handler():
    """Dashboard supports alert handlers."""
    dashboard = FarmDashboard()
    handler_called = []

    def handler(alert):
        handler_called.append(alert)

    dashboard.add_alert_handler(handler)
    assert len(dashboard._alert_handlers) == 1

    dashboard.remove_alert_handler(handler)
    assert len(dashboard._alert_handlers) == 0


def test_sensor_reading():
    """SensorReading dataclass works correctly."""
    reading = SensorReading(
        timestamp=1234567890.0,
        sensor_id="sensor-1",
        metric="soil_moisture",
        value=0.45,
        unit="ratio",
    )
    assert reading.sensor_id == "sensor-1"
    assert reading.metric == "soil_moisture"
    assert reading.value == 0.45
    assert reading.unit == "ratio"
