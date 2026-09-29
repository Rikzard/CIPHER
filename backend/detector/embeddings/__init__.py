"""Embedding similarity detector package boundary (deferred; unimplemented)."""

from backend.detector.base import BaseDetector
from backend.models.contracts import DetectorResult, NormalizedContent


class EmbeddingDetectorPlaceholder(BaseDetector):
    """Placeholder embedding detector returning available=False until implemented in Phase 5."""

    @property
    def name(self) -> str:
        return "embedding_detector"

    @property
    def version(self) -> str:
        return "0.0.0-deferred"

    def analyze(self, content: NormalizedContent) -> DetectorResult:
        return DetectorResult(
            detector_name=self.name,
            detector_version=self.version,
            available=False,
            score=None,
            findings=[],
            metadata={"status": "unimplemented_phase_5"},
        )


__all__ = ["EmbeddingDetectorPlaceholder"]
