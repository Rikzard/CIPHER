"""FastAPI application factory for CIPHER."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.api.health import router as health_router
from backend.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application with explicit or environment-loaded settings."""
    resolved_settings = settings or Settings.from_environment()
    application = FastAPI(title=resolved_settings.app_name, version=resolved_settings.app_version)
    application.state.settings = resolved_settings

    @application.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, _exc: Exception) -> JSONResponse:
        """Return a generic error body without leaking exception or prompt data."""
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    application.include_router(health_router)
    return application


app = create_app()
