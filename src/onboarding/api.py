"""Onboarding REST API for agtech-unified.

Provides FastAPI endpoints for tier recommendation, module listing,
setup status, and session management.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.onboarding.dependencies import ModuleDependencyGraph
from src.onboarding.health import SystemHealth
from src.onboarding.modules import ModuleRegistry
from src.onboarding.sizing import OrganizationProfiler
from src.onboarding.templates import ConfigTemplate
from src.onboarding.wizard import SetupWizard


class TierRecommendationRequest(BaseModel):
    """Request body for tier recommendation."""

    employees: int = Field(..., ge=0, description="Number of employees")
    revenue: float = Field(..., ge=0.0, description="Annual revenue in USD")
    fields: int = Field(default=0, ge=0, description="Number of fields/managed plots")
    sensors: int = Field(default=0, ge=0, description="Number of sensors")
    has_dedicated_it: bool = Field(default=False, description="Has dedicated IT staff")


class TierRecommendationResponse(BaseModel):
    """Response body for tier recommendation."""

    tier: str
    total_score: float
    factors: dict[str, float]
    modules: list[str]
    install_order: list[str]


class ModuleListResponse(BaseModel):
    """Response body for module listing."""

    tier: str
    modules: list[str]
    total: int


class SetupStatusResponse(BaseModel):
    """Response body for setup status."""

    org_id: str
    tier: str | None
    profile_complete: bool
    progress: float
    health_status: str
    setup_steps: list[str]
    completed_steps: list[str]


class SessionCreateRequest(BaseModel):
    """Request body for session creation."""

    employees: int = Field(..., ge=0)
    revenue: float = Field(..., ge=0.0)
    fields: int = Field(default=0, ge=0)
    sensors: int = Field(default=0, ge=0)
    has_dedicated_it: bool = Field(default=False)


class SessionCreateResponse(BaseModel):
    """Response body for session creation."""

    org_id: str
    tier: str
    profile_complete: bool
    setup_steps: list[str]


class HealthResponse(BaseModel):
    """Response body for health check."""

    status: str
    version: str


def create_onboarding_api() -> FastAPI:
    """Create and configure the onboarding FastAPI application.

    Returns:
        Configured FastAPI app instance.
    """
    app = FastAPI(title="AgTech Onboarding API", version="1.0.0")

    _registry = ModuleRegistry()
    _profiler = OrganizationProfiler()
    _templates = ConfigTemplate()
    _graph = ModuleDependencyGraph.from_registry(_registry)
    _health = SystemHealth()
    _sessions: dict[str, dict[str, Any]] = {}

    @app.get("/api/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(status="healthy", version="1.0.0")

    @app.post(
        "/api/v1/onboarding/recommend",
        response_model=TierRecommendationResponse,
    )
    def recommend_tier(req: TierRecommendationRequest) -> TierRecommendationResponse:
        """Recommend a tier based on organization attributes."""
        score_result = _profiler.score_organization(
            employees=req.employees,
            revenue=req.revenue,
            fields=req.fields,
            sensors=req.sensors,
            has_dedicated_it=req.has_dedicated_it,
        )
        tier = _profiler.recommend_tier_scored(
            employees=req.employees,
            revenue=req.revenue,
            fields=req.fields,
            sensors=req.sensors,
            has_dedicated_it=req.has_dedicated_it,
        )
        modules = _registry.get_modules(tier)
        install_order = _graph.get_install_order(modules)
        return TierRecommendationResponse(
            tier=tier,
            total_score=score_result["total_score"],
            factors=score_result["factors"],
            modules=modules,
            install_order=install_order,
        )

    @app.get("/api/v1/onboarding/modules", response_model=list[ModuleListResponse])
    def list_all_modules() -> list[ModuleListResponse]:
        """List all modules grouped by tier."""
        result = []
        for tier in _registry.all_tiers():
            modules = _registry.get_modules(tier)
            result.append(ModuleListResponse(tier=tier, modules=modules, total=len(modules)))
        return result

    @app.get(
        "/api/v1/onboarding/modules/{tier}",
        response_model=ModuleListResponse,
    )
    def get_modules_by_tier(tier: str) -> ModuleListResponse:
        """Get modules for a specific tier."""
        if tier not in _registry.all_tiers():
            raise HTTPException(status_code=404, detail=f"Unknown tier: {tier}")
        modules = _registry.get_modules(tier)
        return ModuleListResponse(tier=tier, modules=modules, total=len(modules))

    @app.get(
        "/api/v1/onboarding/status",
        response_model=SetupStatusResponse,
    )
    def get_setup_status() -> SetupStatusResponse:
        """Get the current setup status (no active session)."""
        return SetupStatusResponse(
            org_id="",
            tier=None,
            profile_complete=False,
            progress=0.0,
            health_status=_health.get_status().value,
            setup_steps=[],
            completed_steps=[],
        )

    @app.post(
        "/api/v1/onboarding/sessions",
        response_model=SessionCreateResponse,
        status_code=201,
    )
    def create_session(req: SessionCreateRequest) -> SessionCreateResponse:
        """Create a new onboarding session."""
        org_id = str(uuid.uuid4())[:8]
        wizard = SetupWizard()
        wizard.set_organization_profile(
            employees=req.employees,
            revenue=req.revenue,
            fields=req.fields,
            sensors=req.sensors,
            has_dedicated_it=req.has_dedicated_it,
        )
        tier = wizard.get_recommended_tier()
        setup_steps = wizard.get_setup_steps()
        _sessions[org_id] = {
            "tier": tier,
            "profile_complete": True,
            "setup_steps": setup_steps,
            "completed_steps": [],
        }
        return SessionCreateResponse(
            org_id=org_id,
            tier=tier,
            profile_complete=True,
            setup_steps=setup_steps,
        )

    @app.get(
        "/api/v1/onboarding/status/{org_id}",
        response_model=SetupStatusResponse,
    )
    def get_session_status(org_id: str) -> SetupStatusResponse:
        """Get setup status for a specific session."""
        if org_id not in _sessions:
            raise HTTPException(status_code=404, detail=f"Session not found: {org_id}")
        session = _sessions[org_id]
        completed = session.get("completed_steps", [])
        total_steps = len(session.get("setup_steps", []))
        progress = len(completed) / total_steps if total_steps > 0 else 0.0
        return SetupStatusResponse(
            org_id=org_id,
            tier=session["tier"],
            profile_complete=session["profile_complete"],
            progress=progress,
            health_status=_health.get_status().value,
            setup_steps=session.get("setup_steps", []),
            completed_steps=completed,
        )

    return app


app = create_onboarding_api()
