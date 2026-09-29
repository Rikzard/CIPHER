"""Local semantic similarity detector based on Sentence Transformers and FAISS."""

from backend.detector.embeddings.embedding_detector import EmbeddingConfig, EmbeddingDetector
from backend.detector.embeddings.vector_store import InjectionExample, LocalFaissVectorStore
from backend.detector.base import BaseDetector
from backend.models.contracts import DetectorResult, NormalizedContent


class EmbeddingDetectorPlaceholder(BaseDetector):
    """Backward-compatible placeholder retained for the existing orchestrator.

    The new semantic detector is intentionally not wired into ApplicationService.
    """

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

__all__ = [
    "EmbeddingConfig", "EmbeddingDetector", "EmbeddingDetectorPlaceholder",
    "InjectionExample", "LocalFaissVectorStore",
]
