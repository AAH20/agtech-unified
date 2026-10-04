"""Unified REST API Gateway for the AgTech platform."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field

from src.decision_support.recommender import DecisionEngine
from src.integration.farm_state import FarmState

logger = logging.getLogger(__name__)


@dataclass
class Farm:
    id: str
    tenant_id: str
    name: str
    location: str
    crop_type: str
    area_hectares: float
    soil_moisture: float = 0.5
    temperature: float = 20.0
    nutrient_level: float = 0.5
    pest_pressure: float = 0.0


@dataclass
class SensorReading:
    sensor_id: str
    farm_id: str
    metric: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ActuatorCommand:
    id: str
    farm_id: str
    actuator_type: str
    action: str
    status: str = "pending"
    issued_at: float = field(default_factory=time.time)


class TenantRegistry:
    def __init__(self):
        self._tenants: Dict[str, str] = {}
        self._farms: Dict[str, Farm] = {}
        self._readings: List[SensorReading] = []
        self._commands: List[ActuatorCommand] = []

    def create_tenant(self, name: str) -> str:
        tid = str(uuid.uuid4())[:8]
        self._tenants[tid] = name
        return tid

    def get_tenant(self, tenant_id: str) -> Optional[str]:
        return self._tenants.get(tenant_id)

    def list_tenants(self) -> List[Dict[str, object]]:
        return [{"id": t, "name": n} for t, n in self._tenants.items()]

    def delete_tenant(self, tenant_id: str) -> bool:
        if tenant_id not in self._tenants:
            return False
        del self._tenants[tenant_id]
        # Cascade: remove farms belonging to this tenant
        farms_to_remove = [fid for fid, f in self._farms.items() if f.tenant_id == tenant_id]
        for fid in farms_to_remove:
            del self._farms[fid]
        # Cascade: remove readings and commands for those farms
        self._readings = [r for r in self._readings if r.farm_id not in farms_to_remove]
        self._commands = [c for c in self._commands if c.farm_id not in farms_to_remove]
        return True

    def add_farm(self, tenant_id: str, farm: Farm) -> Farm:
        if tenant_id not in self._tenants:
            raise KeyError(f"Tenant {tenant_id} not found")
        self._farms[farm.id] = farm
        return farm

    def get_farm(self, tenant_id: str, farm_id: str) -> Optional[Farm]:
        farm = self._farms.get(farm_id)
        return farm if farm and farm.tenant_id == tenant_id else None

    def list_farms(self, tenant_id: str) -> List[Farm]:
        return [f for f in self._farms.values() if f.tenant_id == tenant_id]

    def delete_farm(self, tenant_id: str, farm_id: str) -> bool:
        farm = self.get_farm(tenant_id, farm_id)
        if farm is None:
            return False
        del self._farms[farm_id]
        return True

    def add_reading(self, reading: SensorReading) -> SensorReading:
        self._readings.append(reading)
        return reading

    def list_readings(self, farm_id: str) -> List[SensorReading]:
        return [r for r in self._readings if r.farm_id == farm_id]

    def add_command(self, cmd: ActuatorCommand) -> ActuatorCommand:
        self._commands.append(cmd)
        return cmd

    def list_commands(self, farm_id: str) -> List[ActuatorCommand]:
        return [c for c in self._commands if c.farm_id == farm_id]


_registry = TenantRegistry()


class CreateFarmRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    location: str = Field(..., min_length=1, max_length=200)
    crop_type: str = Field(..., min_length=1, max_length=100)
    area_hectares: float = Field(..., gt=0.0, le=100000.0)


class FarmResponse(BaseModel):
    id: str
    tenant_id: str
    name: str
    location: str
    crop_type: str
    area_hectares: float
    soil_moisture: float
    temperature: float
    nutrient_level: float
    pest_pressure: float


class SensorReadingRequest(BaseModel):
    sensor_id: str
    metric: str
    value: float
    unit: str = ""


class SensorReadingResponse(BaseModel):
    sensor_id: str
    farm_id: str
    metric: str
    value: float
    unit: str
    timestamp: float


class ActuatorCommandRequest(BaseModel):
    actuator_type: str
    action: str


class ActuatorCommandResponse(BaseModel):
    id: str
    farm_id: str
    actuator_type: str
    action: str
    status: str
    issued_at: float


class RecommendationRequest(BaseModel):
    soil_moisture: float = Field(..., ge=0.0, le=1.0)
    temperature: float = Field(..., ge=-50.0, le=60.0)
    crop_height: float = Field(..., ge=0.0, le=10.0)
    nutrient_level: float = Field(..., ge=0.0, le=1.0)
    pest_pressure: float = Field(..., ge=0.0, le=1.0)


class RecommendationResponse(BaseModel):
    recommendations: List[str]
    priority_score: float
    algorithm: str
    actions: List[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    tenants: int


def create_app(
    registry: Optional[TenantRegistry] = None,
    auth: Any = None,
) -> FastAPI:
    app = FastAPI(title="AgTech Unified API", version="1.0.0")
    reg = registry or _registry

    def _farm_resp(f: Farm) -> FarmResponse:
        return FarmResponse(
            id=f.id,
            tenant_id=f.tenant_id,
            name=f.name,
            location=f.location,
            crop_type=f.crop_type,
            area_hectares=f.area_hectares,
            soil_moisture=f.soil_moisture,
            temperature=f.temperature,
            nutrient_level=f.nutrient_level,
            pest_pressure=f.pest_pressure,
        )

    def _authenticate(request: Request, required_role: Optional[str] = None) -> Dict[str, Any]:
        """Authenticate the request using the configured auth module."""
        if auth is None:
            return {"sub": "anonymous", "roles": ["admin"], "tenant_id": None}
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        token = auth_header[7:]
        try:
            tenant_id = request.path_params.get("tenant_id")
            return auth.authenticate(token, required_role=required_role, tenant_id=tenant_id)
        except Exception as exc:
            if "tenant" in str(exc).lower():
                raise HTTPException(status_code=403, detail=str(exc))
            raise HTTPException(status_code=401, detail=str(exc))

    def _audit(
        request: Request, payload: Dict[str, Any], status_code: int, latency_ms: float
    ) -> None:
        """Log the API request to the audit logger if available."""
        if auth and hasattr(auth, "_audit_logger") and auth._audit_logger:
            tenant_id = request.path_params.get("tenant_id") or payload.get("tenant_id")
            auth._audit_logger.log_api_request(
                method=request.method,
                endpoint=str(request.url.path),
                tenant_id=tenant_id,
                actor=payload.get("sub", "anonymous"),
                request_id=request.headers.get("X-Request-ID"),
                source_ip=request.client.host if request.client else None,
                user_agent=request.headers.get("User-Agent"),
                status=str(status_code),
                latency_ms=latency_ms,
            )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        """Add correlation ID, measure latency, and audit the request."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.monotonic()
        response = await call_next(request)
        latency_ms = (time.monotonic() - start) * 1000.0
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"
        # Audit the request
        try:
            tenant_id = request.path_params.get("tenant_id")
            actor = "anonymous"
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer ") and auth:
                try:
                    payload = auth.validate_token(auth_header[7:])
                    actor = payload.get("sub", "anonymous")
                    if not tenant_id:
                        tenant_id = payload.get("tenant_id")
                except Exception:
                    pass
            if auth and hasattr(auth, "_audit_logger") and auth._audit_logger:
                auth._audit_logger.log_api_request(
                    method=request.method,
                    endpoint=str(request.url.path),
                    tenant_id=tenant_id,
                    actor=actor,
                    request_id=request_id,
                    source_ip=request.client.host if request.client else None,
                    user_agent=request.headers.get("User-Agent"),
                    status=str(response.status_code),
                    latency_ms=latency_ms,
                )
        except Exception:
            pass  # Don't let audit failures break requests
        return response

    @app.get("/api/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="healthy", version="1.0.0", tenants=len(reg.list_tenants()))

    @app.post("/api/v1/tenants", status_code=201)
    def create_tenant(request: Request, name: str = Header(...)) -> Dict[str, str]:
        _authenticate(request, required_role="admin")
        tid = reg.create_tenant(name)
        return {"id": tid, "name": name}

    @app.get("/api/v1/tenants")
    def list_tenants(request: Request) -> List[Dict[str, object]]:
        _authenticate(request, required_role="admin")
        return reg.list_tenants()

    @app.delete("/api/v1/tenants/{tenant_id}")
    def delete_tenant(request: Request, tenant_id: str):
        _authenticate(request, required_role="admin")
        if not reg.delete_tenant(tenant_id):
            raise HTTPException(status_code=404, detail="Tenant not found")
        return Response(status_code=204)

    @app.post("/api/v1/tenants/{tenant_id}/farms", response_model=FarmResponse, status_code=201)
    def create_farm(request: Request, tenant_id: str, req: CreateFarmRequest) -> FarmResponse:
        _authenticate(request, required_role="operator")
        if reg.get_tenant(tenant_id) is None:
            raise HTTPException(404, "Tenant not found")
        farm = Farm(
            id=str(uuid.uuid4())[:8],
            tenant_id=tenant_id,
            name=req.name,
            location=req.location,
            crop_type=req.crop_type,
            area_hectares=req.area_hectares,
        )
        return _farm_resp(reg.add_farm(tenant_id, farm))

    @app.get("/api/v1/tenants/{tenant_id}/farms", response_model=List[FarmResponse])
    def list_farms(request: Request, tenant_id: str) -> List[FarmResponse]:
        _authenticate(request, required_role="viewer")
        if reg.get_tenant(tenant_id) is None:
            raise HTTPException(404, "Tenant not found")
        return [_farm_resp(f) for f in reg.list_farms(tenant_id)]

    @app.get("/api/v1/tenants/{tenant_id}/farms/{farm_id}", response_model=FarmResponse)
    def get_farm(request: Request, tenant_id: str, farm_id: str) -> FarmResponse:
        _authenticate(request, required_role="viewer")
        farm = reg.get_farm(tenant_id, farm_id)
        if farm is None:
            raise HTTPException(404, "Farm not found")
        return _farm_resp(farm)

    @app.post(
        "/api/v1/tenants/{tenant_id}/farms/{farm_id}/readings",
        response_model=SensorReadingResponse,
        status_code=201,
    )
    def add_reading(
        request: Request, tenant_id: str, farm_id: str, req: SensorReadingRequest
    ) -> SensorReadingResponse:
        _authenticate(request, required_role="operator")
        if reg.get_farm(tenant_id, farm_id) is None:
            raise HTTPException(404, "Farm not found")
        r = reg.add_reading(
            SensorReading(
                sensor_id=req.sensor_id,
                farm_id=farm_id,
                metric=req.metric,
                value=req.value,
                unit=req.unit,
            )
        )
        return SensorReadingResponse(
            sensor_id=r.sensor_id,
            farm_id=r.farm_id,
            metric=r.metric,
            value=r.value,
            unit=r.unit,
            timestamp=r.timestamp,
        )

    @app.get(
        "/api/v1/tenants/{tenant_id}/farms/{farm_id}/readings",
        response_model=List[SensorReadingResponse],
    )
    def list_readings(
        request: Request, tenant_id: str, farm_id: str
    ) -> List[SensorReadingResponse]:
        _authenticate(request, required_role="viewer")
        if reg.get_farm(tenant_id, farm_id) is None:
            raise HTTPException(404, "Farm not found")
        return [
            SensorReadingResponse(
                sensor_id=r.sensor_id,
                farm_id=r.farm_id,
                metric=r.metric,
                value=r.value,
                unit=r.unit,
                timestamp=r.timestamp,
            )
            for r in reg.list_readings(farm_id)
        ]

    @app.post(
        "/api/v1/tenants/{tenant_id}/farms/{farm_id}/commands",
        response_model=ActuatorCommandResponse,
        status_code=201,
    )
    def send_command(
        request: Request, tenant_id: str, farm_id: str, req: ActuatorCommandRequest
    ) -> ActuatorCommandResponse:
        _authenticate(request, required_role="operator")
        if reg.get_farm(tenant_id, farm_id) is None:
            raise HTTPException(404, "Farm not found")
        cmd = reg.add_command(
            ActuatorCommand(
                id=str(uuid.uuid4())[:8],
                farm_id=farm_id,
                actuator_type=req.actuator_type,
                action=req.action,
            )
        )
        return ActuatorCommandResponse(
            id=cmd.id,
            farm_id=cmd.farm_id,
            actuator_type=cmd.actuator_type,
            action=cmd.action,
            status=cmd.status,
            issued_at=cmd.issued_at,
        )

    @app.get(
        "/api/v1/tenants/{tenant_id}/farms/{farm_id}/commands",
        response_model=List[ActuatorCommandResponse],
    )
    def list_commands(
        request: Request, tenant_id: str, farm_id: str
    ) -> List[ActuatorCommandResponse]:
        _authenticate(request, required_role="viewer")
        if reg.get_farm(tenant_id, farm_id) is None:
            raise HTTPException(404, "Farm not found")
        return [
            ActuatorCommandResponse(
                id=c.id,
                farm_id=c.farm_id,
                actuator_type=c.actuator_type,
                action=c.action,
                status=c.status,
                issued_at=c.issued_at,
            )
            for c in reg.list_commands(farm_id)
        ]

    @app.post("/api/v1/tenants/{tenant_id}/recommendations", response_model=RecommendationResponse)
    def recommend(
        request: Request, tenant_id: str, req: RecommendationRequest
    ) -> RecommendationResponse:
        _authenticate(request, required_role="viewer")
        if reg.get_tenant(tenant_id) is None:
            raise HTTPException(404, "Tenant not found")
        result = DecisionEngine().recommend(
            FarmState(
                soil_moisture=req.soil_moisture,
                temperature=req.temperature,
                crop_height=req.crop_height,
                nutrient_level=req.nutrient_level,
                pest_pressure=req.pest_pressure,
            )
        )
        return RecommendationResponse(
            recommendations=result.recommendations,
            priority_score=result.priority_score,
            algorithm=result.algorithm,
            actions=result.actions,
        )

    return app


class APIGateway:
    def __init__(
        self,
        registry: Optional[TenantRegistry] = None,
        auth: Any = None,
    ) -> None:
        self.registry = registry or TenantRegistry()
        self.auth = auth
        self.app = create_app(self.registry, auth=self.auth)

    def create_tenant(self, name: str) -> str:
        return self.registry.create_tenant(name)

    def get_tenant(self, tenant_id: str) -> Optional[str]:
        return self.registry.get_tenant(tenant_id)

    def list_tenants(self) -> List[Dict[str, object]]:
        return self.registry.list_tenants()

    def delete_tenant(self, tenant_id: str) -> bool:
        return self.registry.delete_tenant(tenant_id)

    def create_farm(
        self, tenant_id: str, name: str, location: str, crop_type: str, area_hectares: float
    ) -> Farm:
        if self.registry.get_tenant(tenant_id) is None:
            raise KeyError(f"Tenant {tenant_id} not found")
        return self.registry.add_farm(
            tenant_id,
            Farm(
                id=str(uuid.uuid4())[:8],
                tenant_id=tenant_id,
                name=name,
                location=location,
                crop_type=crop_type,
                area_hectares=area_hectares,
            ),
        )

    def get_farm(self, tenant_id: str, farm_id: str) -> Optional[Farm]:
        return self.registry.get_farm(tenant_id, farm_id)

    def list_farms(self, tenant_id: str) -> List[Farm]:
        return self.registry.list_farms(tenant_id)

    def delete_farm(self, tenant_id: str, farm_id: str) -> bool:
        return self.registry.delete_farm(tenant_id, farm_id)

    def add_reading(
        self, farm_id: str, sensor_id: str, metric: str, value: float, unit: str = ""
    ) -> SensorReading:
        return self.registry.add_reading(
            SensorReading(
                sensor_id=sensor_id, farm_id=farm_id, metric=metric, value=value, unit=unit
            )
        )

    def list_readings(self, farm_id: str) -> List[SensorReading]:
        return self.registry.list_readings(farm_id)

    def send_command(self, farm_id: str, actuator_type: str, action: str) -> ActuatorCommand:
        return self.registry.add_command(
            ActuatorCommand(
                id=str(uuid.uuid4())[:8],
                farm_id=farm_id,
                actuator_type=actuator_type,
                action=action,
            )
        )

    def list_commands(self, farm_id: str) -> List[ActuatorCommand]:
        return self.registry.list_commands(farm_id)

    def recommend(
        self,
        soil_moisture: float,
        temperature: float,
        crop_height: float,
        nutrient_level: float,
        pest_pressure: float,
    ):
        return DecisionEngine().recommend(
            FarmState(
                soil_moisture=soil_moisture,
                temperature=temperature,
                crop_height=crop_height,
                nutrient_level=nutrient_level,
                pest_pressure=pest_pressure,
            )
        )


app = create_app()
