"""Local FAISS index and example metadata for semantic detection."""

from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class VectorStoreError(ValueError):
    """Raised when a local vector index or its metadata is invalid."""


@dataclass(frozen=True)
class InjectionExample:
    """A labeled, version-controlled prompt-injection reference example."""

    example_id: str
    category: str
    text: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "InjectionExample":
        try:
            example = cls(
                example_id=str(value["id"]),
                category=str(value["category"]),
                text=str(value["text"]),
            )
        except (KeyError, TypeError) as exc:
            raise VectorStoreError("Dataset records require id, category, and text fields") from exc
        if not all((example.example_id.strip(), example.category.strip(), example.text.strip())):
            raise VectorStoreError("Dataset id, category, and text fields must not be empty")
        return example

    def as_mapping(self) -> dict[str, str]:
        return {"id": self.example_id, "category": self.category, "text": self.text}


class LocalFaissVectorStore:
    """Cosine-similarity FAISS index with a colocated JSON metadata file.

    Vectors passed to :meth:`from_embeddings` are normalized before being
    indexed, so inner product search is equivalent to cosine similarity.
    """

    METADATA_VERSION = 2

    def __init__(self, index: Any, examples: Sequence[InjectionExample], metadata: Mapping[str, Any]) -> None:
        self._faiss = _load_faiss()
        if index.ntotal != len(examples):
            raise VectorStoreError("FAISS index and example metadata have different record counts")
        dimension = metadata.get("embedding_dimension")
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise VectorStoreError("Metadata embedding_dimension must be a positive integer")
        if index.d != dimension:
            raise VectorStoreError("FAISS index dimension does not match its metadata")
        if not examples:
            raise VectorStoreError("Vector store must contain at least one example")
        self._validate_metadata(metadata, examples)
        self._index = index
        self._examples = tuple(examples)
        self.dimension = dimension
        self.metadata = dict(metadata)

    @classmethod
    def from_embeddings(
        cls,
        examples: Sequence[InjectionExample],
        embeddings: Any,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> "LocalFaissVectorStore":
        np = _load_numpy()
        faiss = _load_faiss()
        matrix = np.asarray(embeddings, dtype="float32")
        if matrix.ndim != 2 or matrix.shape[0] != len(examples) or matrix.shape[1] <= 0:
            raise VectorStoreError("Embedding matrix shape must match the non-empty example list")
        if not examples:
            raise VectorStoreError("Vector store must contain at least one example")
        if not np.isfinite(matrix).all():
            raise VectorStoreError("Embedding matrix contains non-finite values")
        matrix = matrix.copy()
        faiss.normalize_L2(matrix)
        if (np.linalg.norm(matrix, axis=1) == 0).any():
            raise VectorStoreError("Embedding vectors must have non-zero length")
        index = faiss.IndexFlatIP(int(matrix.shape[1]))
        index.add(matrix)
        example_mappings = [example.as_mapping() for example in examples]
        serialized_examples = json.dumps(
            example_mappings, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        supplied = dict(provenance or {})
        metadata: dict[str, Any] = {
            "metadata_version": cls.METADATA_VERSION,
            "dataset_version": supplied.get("dataset_version", "unspecified"),
            "dataset_sha256": supplied.get("dataset_sha256", hashlib.sha256(serialized_examples).hexdigest()),
            "embedding_model_name": supplied.get("embedding_model_name", "injected-test-encoder"),
            "embedding_model_path": supplied.get("embedding_model_path", "<injected>"),
            "embedding_dimension": int(matrix.shape[1]),
            "similarity_metric": "cosine",
            "indexed_example_count": len(examples),
            "examples": example_mappings,
        }
        return cls(index, examples, metadata)

    @classmethod
    def load(cls, index_path: str | Path) -> "LocalFaissVectorStore":
        path = Path(index_path)
        metadata_path = cls.metadata_path(path)
        if not path.is_file():
            raise VectorStoreError(f"FAISS index file not found: {path}")
        if not metadata_path.is_file():
            raise VectorStoreError(f"FAISS metadata file not found: {metadata_path}")
        faiss = _load_faiss()
        try:
            index = faiss.read_index(str(path))
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            actual_index_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception as exc:
            raise VectorStoreError("Could not read FAISS index or metadata") from exc
        if not isinstance(payload, dict) or payload.get("metadata_version") != cls.METADATA_VERSION:
            raise VectorStoreError("Unsupported FAISS metadata format")
        expected_index_hash = payload.get("faiss_index_sha256")
        if not isinstance(expected_index_hash, str) or expected_index_hash != actual_index_hash:
            raise VectorStoreError("FAISS index checksum does not match its metadata")
        try:
            examples = [InjectionExample.from_mapping(item) for item in payload["examples"]]
            return cls(index, examples, payload)
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, VectorStoreError):
                raise
            raise VectorStoreError("Invalid FAISS metadata contents") from exc

    @staticmethod
    def metadata_path(index_path: str | Path) -> Path:
        path = Path(index_path)
        return path.with_suffix(path.suffix + ".json")

    def save(self, index_path: str | Path) -> None:
        path = Path(index_path)
        metadata_path = self.metadata_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        index_temp = path.with_name(f".{path.stem}.tmp{path.suffix}")
        metadata_temp = metadata_path.with_name(f".{path.stem}.tmp{path.suffix}.json")
        try:
            self._faiss.write_index(self._index, str(index_temp))
            metadata_payload = dict(self.metadata)
            metadata_payload["faiss_index_sha256"] = hashlib.sha256(index_temp.read_bytes()).hexdigest()
            self._validate_metadata(metadata_payload, self._examples)
            metadata_temp.write_text(
                json.dumps(metadata_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(index_temp, path)
            os.replace(metadata_temp, metadata_path)
            self.metadata = metadata_payload
        except Exception as exc:
            raise VectorStoreError("Could not save FAISS index and metadata") from exc
        finally:
            index_temp.unlink(missing_ok=True)
            metadata_temp.unlink(missing_ok=True)

    @staticmethod
    def _validate_metadata(metadata: Mapping[str, Any], examples: Sequence[InjectionExample]) -> None:
        required_strings = (
            "dataset_version", "dataset_sha256", "embedding_model_name",
            "embedding_model_path", "similarity_metric",
        )
        if metadata.get("metadata_version") != LocalFaissVectorStore.METADATA_VERSION:
            raise VectorStoreError("Unsupported FAISS metadata format")
        for field in required_strings:
            value = metadata.get(field)
            if not isinstance(value, str) or not value.strip():
                raise VectorStoreError(f"Metadata {field} must be a non-empty string")
        checksum = metadata["dataset_sha256"]
        if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
            raise VectorStoreError("Metadata dataset_sha256 must be a lowercase SHA-256 digest")
        index_checksum = metadata.get("faiss_index_sha256")
        if index_checksum is not None and (
            not isinstance(index_checksum, str)
            or len(index_checksum) != 64
            or any(character not in "0123456789abcdef" for character in index_checksum)
        ):
            raise VectorStoreError("Metadata faiss_index_sha256 must be a lowercase SHA-256 digest")
        if metadata["similarity_metric"] != "cosine":
            raise VectorStoreError("Unsupported embedding similarity metric")
        count = metadata.get("indexed_example_count")
        if not isinstance(count, int) or isinstance(count, bool) or count != len(examples):
            raise VectorStoreError("Metadata indexed_example_count does not match its examples")
        metadata_examples = metadata.get("examples")
        if not isinstance(metadata_examples, list) or len(metadata_examples) != len(examples):
            raise VectorStoreError("Metadata example records do not match the indexed example count")
        if len({example.example_id for example in examples}) != len(examples):
            raise VectorStoreError("FAISS metadata contains duplicate example IDs")

    @property
    def examples(self) -> tuple[InjectionExample, ...]:
        """Indexed example metadata, in FAISS row order."""
        return self._examples

    def search(self, embedding: Any, top_k: int = 3) -> list[dict[str, Any]]:
        np = _load_numpy()
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vector = np.asarray(embedding, dtype="float32")
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.shape != (1, self.dimension) or not np.isfinite(vector).all():
            raise VectorStoreError("Query embedding has an invalid shape or non-finite values")
        vector = vector.copy()
        faiss = self._faiss
        faiss.normalize_L2(vector)
        if float(np.linalg.norm(vector)) == 0.0:
            raise VectorStoreError("Query embedding must have non-zero length")
        scores, indices = self._index.search(vector, min(top_k, len(self._examples)))
        matches: list[dict[str, Any]] = []
        for score, index_id in zip(scores[0].tolist(), indices[0].tolist()):
            if index_id < 0:
                continue
            example = self._examples[index_id]
            cosine = max(-1.0, min(1.0, float(score)))
            matches.append({
                "example_id": example.example_id,
                "category": example.category,
                "cosine_similarity": cosine,
                "cosine_distance": 1.0 - cosine,
            })
        return matches

    @classmethod
    def load_dataset(cls, dataset_path: str | Path) -> list[InjectionExample]:
        path = Path(dataset_path)
        if not path.is_file():
            raise VectorStoreError(f"Prompt-injection dataset not found: {path}")
        examples: list[InjectionExample] = []
        try:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise VectorStoreError(f"Invalid JSON on dataset line {line_number}") from exc
                if not isinstance(value, dict):
                    raise VectorStoreError(f"Dataset line {line_number} must contain a JSON object")
                examples.append(InjectionExample.from_mapping(value))
        except OSError as exc:
            raise VectorStoreError("Could not read prompt-injection dataset") from exc
        if not examples:
            raise VectorStoreError("Prompt-injection dataset contains no examples")
        if len({example.example_id for example in examples}) != len(examples):
            raise VectorStoreError("Prompt-injection dataset contains duplicate IDs")
        return examples


def _load_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise VectorStoreError("numpy is required for the local FAISS vector store") from exc
    return np


def _load_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise VectorStoreError("faiss-cpu is required for the local vector store") from exc
    return faiss
