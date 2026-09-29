"""Health endpoint response model."""

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Basic process health; this does not imply detector readiness."""

    model_config = ConfigDict(frozen=True)

    status: str
    service: str
    version: str
    environment: str
