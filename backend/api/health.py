"""Health route, separate from future analysis endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Request

from backend.config import Settings
from backend.models.health import HealthResponse

router = APIRouter()


def get_settings(request: Request) -> Settings:
    """Read the settings instance attached by the application factory."""
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(status_code=500, detail="Application settings not initialized")
    return settings


@router.get("/health", response_model=HealthResponse, tags=["health"])
def get_health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Report that the API process is responding."""
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )
