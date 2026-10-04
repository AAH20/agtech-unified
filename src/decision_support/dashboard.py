"""Real-time farm monitoring dashboard with WebSocket support."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.decision_support.recommender import FarmState

logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    """A single sensor reading."""

    timestamp: float
    sensor_id: str
    metric: str
    value: float
    unit: str = ""


@dataclass
class DashboardConfig:
    """Configuration for the farm dashboard."""

    update_interval: float = 1.0
    max_history: int = 1000
    enable_alerts: bool = True
    farm_name: str = "Default Farm"


class FarmDashboard:
    """Real-time farm monitoring dashboard with WebSocket support."""

    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self._state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.3,
            nutrient_level=0.5,
            pest_pressure=0.1,
        )
        self._history: deque = deque(maxlen=self.config.max_history)
        self._clients: Set[WebSocket] = set()
        self._running = False
        self._alert_handlers: List[Callable] = []

    @property
    def state(self) -> FarmState:
        """Current farm state."""
        return self._state

    @property
    def client_count(self) -> int:
        """Number of connected WebSocket clients."""
        return len(self._clients)

    @property
    def is_running(self) -> bool:
        """Whether the dashboard streaming is active."""
        return self._running

    def update_state(self, **kwargs) -> None:
        """Update farm state with new values."""
        current = self._state
        self._state = FarmState(
            soil_moisture=kwargs.get("soil_moisture", current.soil_moisture),
            temperature=kwargs.get("temperature", current.temperature),
            crop_height=kwargs.get("crop_height", current.crop_height),
            nutrient_level=kwargs.get("nutrient_level", current.nutrient_level),
            pest_pressure=kwargs.get("pest_pressure", current.pest_pressure),
        )
        self._history.append(
            {
                "timestamp": time.time(),
                "state": self.get_state_dict(),
            }
        )

    def get_state_dict(self) -> Dict[str, Any]:
        """Get current state as a dictionary."""
        return {
            "soil_moisture": self._state.soil_moisture,
            "temperature": self._state.temperature,
            "crop_height": self._state.crop_height,
            "nutrient_level": self._state.nutrient_level,
            "pest_pressure": self._state.pest_pressure,
            "timestamp": time.time(),
        }

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get historical state snapshots."""
        history_list = list(self._history)
        if limit is not None:
            history_list = history_list[-limit:]
        return history_list

    def add_client(self, websocket: WebSocket) -> None:
        """Register a WebSocket client."""
        self._clients.add(websocket)

    def remove_client(self, websocket: WebSocket) -> None:
        """Unregister a WebSocket client."""
        self._clients.discard(websocket)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        disconnected = set()
        for client in self._clients:
            try:
                await client.send_json(message)
            except Exception:
                disconnected.add(client)
        for client in disconnected:
            self._clients.discard(client)

    async def stream_data(self):
        """Generator that yields state updates at configured interval."""
        while self._running:
            yield self.get_state_dict()
            await asyncio.sleep(self.config.update_interval)

    def create_app(self) -> FastAPI:
        """Create a FastAPI application with WebSocket endpoint."""
        app = FastAPI(title="Farm Dashboard", version="1.0.0")

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.add_client(websocket)
            try:
                await websocket.send_json(
                    {
                        "type": "state",
                        "data": self.get_state_dict(),
                    }
                )
                while True:
                    message = await websocket.receive_text()
                    try:
                        data = json.loads(message)
                        if data.get("action") == "update":
                            self.update_state(**data.get("state", {}))
                            await self.broadcast(
                                {
                                    "type": "state",
                                    "data": self.get_state_dict(),
                                }
                            )
                    except json.JSONDecodeError:
                        pass
            except WebSocketDisconnect:
                pass
            finally:
                self.remove_client(websocket)

        @app.get("/state")
        async def get_state():
            return self.get_state_dict()

        @app.get("/history")
        async def get_history(limit: int = 100):
            return self.get_history(limit=limit)

        return app

    async def run(self):
        """Start the dashboard streaming loop."""
        self._running = True
        try:
            async for state in self.stream_data():
                await self.broadcast({"type": "state", "data": state})
        finally:
            self._running = False

    def stop(self):
        """Stop the dashboard streaming."""
        self._running = False

    def add_alert_handler(self, handler: Callable) -> None:
        """Add a handler for alert events."""
        self._alert_handlers.append(handler)

    def remove_alert_handler(self, handler: Callable) -> None:
        """Remove an alert handler."""
        if handler in self._alert_handlers:
            self._alert_handlers.remove(handler)
