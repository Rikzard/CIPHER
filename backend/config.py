"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigurationError(ValueError):
    """Raised when application configuration is invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings required by the backend skeleton."""

    app_name: str = "CIPHER"
    app_version: str = "0.1.0"
    environment: str = "development"

    @classmethod
    def from_environment(cls) -> "Settings":
        """Load settings from environment variables and validate them."""
        settings = cls(
            app_name=os.getenv("CIPHER_APP_NAME", "CIPHER").strip(),
            app_version=os.getenv("CIPHER_APP_VERSION", "0.1.0").strip(),
            environment=os.getenv("CIPHER_ENVIRONMENT", "development").strip(),
        )
        for name in ("app_name", "app_version", "environment"):
            if not getattr(settings, name):
                raise ConfigurationError(f"{name} must not be empty")
        return settings
