"""Local semantic similarity detector based on Sentence Transformers and FAISS."""

from backend.detector.embeddings.embedding_detector import EmbeddingConfig, EmbeddingDetector
from backend.detector.embeddings.vector_store import InjectionExample, LocalFaissVectorStore

__all__ = [
    "EmbeddingConfig",
    "EmbeddingDetector",
    "InjectionExample",
    "LocalFaissVectorStore",
]
