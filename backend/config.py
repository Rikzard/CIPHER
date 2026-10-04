"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MAX_DOCUMENT_SIZE_BYTES = 10 * 1024 * 1024


class ConfigurationError(ValueError):
    """Raised when application configuration is invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings required by the backend skeleton."""

    app_name: str = "CIPHER"
    app_version: str = "0.1.0"
    environment: str = "development"
    max_document_size_bytes: int = DEFAULT_MAX_DOCUMENT_SIZE_BYTES

    def __post_init__(self) -> None:
        if self.max_document_size_bytes <= 0:
            raise ConfigurationError("max_document_size_bytes must be a positive integer")

    @classmethod
    def from_environment(cls) -> "Settings":
        """Load settings from environment variables and validate them."""
        try:
            max_document_size_bytes = int(
                os.getenv("CIPHER_MAX_DOCUMENT_SIZE_BYTES", str(DEFAULT_MAX_DOCUMENT_SIZE_BYTES))
            )
        except ValueError as exc:
            raise ConfigurationError("CIPHER_MAX_DOCUMENT_SIZE_BYTES must be a positive integer") from exc
        if max_document_size_bytes <= 0:
            raise ConfigurationError("CIPHER_MAX_DOCUMENT_SIZE_BYTES must be a positive integer")
        settings = cls(
            app_name=os.getenv("CIPHER_APP_NAME", "CIPHER").strip(),
            app_version=os.getenv("CIPHER_APP_VERSION", "0.1.0").strip(),
            environment=os.getenv("CIPHER_ENVIRONMENT", "development").strip(),
            max_document_size_bytes=max_document_size_bytes,
        )
        for name in ("app_name", "app_version", "environment"):
            if not getattr(settings, name):
                raise ConfigurationError(f"{name} must not be empty")
        return settings
