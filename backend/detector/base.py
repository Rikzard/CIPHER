"""Abstract base class and port definition for detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from backend.models.contracts import DetectorResult, NormalizedContent


class BaseDetector(ABC):
    """Detector port interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the detector."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Version string for detector logic/rules/model."""
        pass

    @abstractmethod
    def analyze(self, content: NormalizedContent) -> DetectorResult:
        """Evaluate normalized content and return DetectorResult."""
        pass
