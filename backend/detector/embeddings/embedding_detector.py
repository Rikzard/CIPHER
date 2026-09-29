"""Independent semantic prompt-injection detector using a local model and FAISS."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from backend.detector.base import BaseDetector
from backend.models.contracts import DetectorResult, NormalizedContent
from backend.normalization.normalizer import normalize_input
from backend.detector.embeddings.vector_store import (
    InjectionExample,
    LocalFaissVectorStore,
    VectorStoreError,
)


class TextEncoder(Protocol):
    """Minimal sentence-transformers-compatible encoder interface."""

    def encode(self, sentences: str | list[str], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class EmbeddingConfig:
    """Local model/index paths and initial, uncalibrated similarity threshold."""

    model_path: Path
    index_path: Path
    dataset_path: Path
    similarity_threshold: float = 0.65
    top_k: int = 3

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        root = Path(__file__).resolve().parents[3]
        return cls(
            model_path=Path(os.getenv("CIPHER_EMBEDDING_MODEL_PATH", root / "models" / "all-MiniLM-L6-v2")),
            index_path=Path(os.getenv("CIPHER_EMBEDDING_INDEX_PATH", root / "data" / "embeddings" / "injection_examples.faiss")),
            dataset_path=Path(os.getenv("CIPHER_EMBEDDING_DATASET_PATH", root / "data" / "embedding_examples.jsonl")),
            similarity_threshold=float(os.getenv("CIPHER_EMBEDDING_THRESHOLD", "0.65")),
            top_k=int(os.getenv("CIPHER_EMBEDDING_TOP_K", "3")),
        )

    def __post_init__(self) -> None:
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")


class EmbeddingDetector(BaseDetector):
    """Compare normalized input with known attacks via local cosine similarity.

    Model and FAISS index loading happen once during construction. Inference
    only encodes the provided text and queries the already loaded index. The
    score is a similarity signal, not a calibrated probability or final policy
    action. An unavailable detector returns a standard unavailable result.
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        encoder: TextEncoder | None = None,
        vector_store: LocalFaissVectorStore | None = None,
    ) -> None:
        self.config = config or EmbeddingConfig.from_env()
        self._encoder = encoder
        self._vector_store = vector_store
        self._load_error: str | None = None
        if self._encoder is None:
            try:
                self._encoder = self._load_encoder(self.config.model_path)
            except Exception as exc:
                self._load_error = f"model_unavailable: {type(exc).__name__}"
        if self._vector_store is None:
            try:
                self._vector_store = LocalFaissVectorStore.load(self.config.index_path)
            except Exception as exc:
                self._load_error = self._load_error or f"index_unavailable: {type(exc).__name__}"
        if self._load_error is None and self._encoder is not None and self._vector_store is not None:
            try:
                probe = self._encode(["dimension check"])
                if probe.shape[1] != self._vector_store.dimension:
                    self._load_error = "model_index_dimension_mismatch"
            except Exception as exc:
                self._load_error = f"model_validation_failed: {type(exc).__name__}"

    @property
    def name(self) -> str:
        return "embedding_detector"

    @property
    def version(self) -> str:
        return "1.0.0"

    @staticmethod
    def _load_encoder(model_path: Path) -> TextEncoder:
        if not model_path.is_dir():
            raise FileNotFoundError(f"Local Sentence Transformer model directory not found: {model_path}")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for embedding detection") from exc
        # Passing a local directory with local_files_only prevents implicit downloads.
        return SentenceTransformer(str(model_path), local_files_only=True)

    def _encode(self, texts: list[str]) -> Any:
        assert self._encoder is not None
        return _encode_texts(self._encoder, texts)

    def analyze(self, content: NormalizedContent) -> DetectorResult:
        if self._load_error or self._encoder is None or self._vector_store is None:
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                available=False,
                score=None,
                findings=[],
                metadata={"status": "unavailable", "reason": self._load_error or "components_not_loaded"},
            )
        if not isinstance(content, NormalizedContent):
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                available=False,
                score=None,
                findings=[],
                metadata={"status": "invalid_input", "reason": "expected NormalizedContent"},
            )
        text = content.canonical_text
        if not text.strip():
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                available=True,
                score=0.0,
                findings=[],
                metadata={"status": "evaluated", "nearest_examples": [], "reason": "empty_canonical_text"},
            )
        try:
            embedding = self._encode([text])[0]
            matches = self._vector_store.search(embedding, top_k=self.config.top_k)
            if not matches:
                raise VectorStoreError("FAISS index returned no examples")
        except Exception as exc:
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                available=False,
                score=None,
                findings=[],
                metadata={"status": "inference_error", "reason": type(exc).__name__},
            )

        nearest = matches[0]
        similarity = nearest["cosine_similarity"]
        score = round(max(0.0, min(1.0, similarity)) * 100.0, 2)
        findings: list[str] = []
        if similarity >= self.config.similarity_threshold:
            findings.append(
                f"[SEM-001] Similarity to {nearest['category']} example {nearest['example_id']} "
                f"(cosine similarity {similarity:.3f})"
            )
        return DetectorResult(
            detector_name=self.name,
            detector_version=self.version,
            available=True,
            score=score,
            findings=findings,
            metadata={
                "status": "evaluated",
                "model": "all-MiniLM-L6-v2",
                "similarity_metric": "cosine",
                "similarity_threshold": self.config.similarity_threshold,
                "threshold_calibrated": False,
                "nearest_examples": matches,
            },
        )


def build_local_index(
    config: EmbeddingConfig | None = None,
    *,
    encoder: TextEncoder | None = None,
) -> Path:
    """Build and save the local FAISS index from the versioned JSONL dataset."""
    selected = config or EmbeddingConfig.from_env()
    model = encoder or EmbeddingDetector._load_encoder(selected.model_path)
    examples = LocalFaissVectorStore.load_dataset(selected.dataset_path)
    # Query-time content is already canonicalized; apply the same transform to
    # reference examples, including obfuscated examples with invisible chars.
    canonical_examples = [normalize_input(example.text).canonical_text for example in examples]
    vectors = _encode_texts(model, canonical_examples)
    store = LocalFaissVectorStore.from_embeddings(examples, vectors)
    store.save(selected.index_path)
    return selected.index_path


def _encode_texts(encoder: TextEncoder, texts: list[str]) -> Any:
    import numpy as np

    vectors = encoder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] != len(texts) or not np.isfinite(matrix).all():
        raise ValueError("Encoder returned an invalid embedding matrix")
    return matrix
