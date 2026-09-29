"""Offline-capable provisioning and validation for local embedding resources.

Provision the model separately, then run this module's ``build`` command.
Both build and validation load models only from the configured local directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from backend.detector.embeddings.embedding_detector import (
    EmbeddingConfig,
    EmbeddingDetector,
    TextEncoder,
    _encode_texts,
)
from backend.detector.embeddings.vector_store import (
    InjectionExample,
    LocalFaissVectorStore,
    VectorStoreError,
)
from backend.normalization.normalizer import normalize_input


class ProvisioningError(RuntimeError):
    """A clear, user-facing error in local model/index provisioning."""


def _dataset_manifest(config: EmbeddingConfig) -> tuple[dict[str, str], str]:
    manifest_path = config.manifest_path
    if not manifest_path.is_file():
        raise ProvisioningError(f"Dataset manifest not found: {manifest_path}")
    if not config.dataset_path.is_file():
        raise ProvisioningError(f"Prompt-injection dataset not found: {config.dataset_path}")
    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Git checkout line-ending conversion must not alter the dataset ID.
        dataset_bytes = config.dataset_path.read_bytes().replace(b"\r\n", b"\n")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProvisioningError("Could not read the dataset or its manifest") from exc
    if not isinstance(manifest_value, dict):
        raise ProvisioningError("Dataset manifest must contain a JSON object")
    version = manifest_value.get("dataset_version")
    dataset_file = manifest_value.get("dataset_file")
    if not isinstance(version, str) or not version.strip():
        raise ProvisioningError("Dataset manifest requires a non-empty dataset_version")
    if not isinstance(dataset_file, str) or dataset_file != config.dataset_path.name:
        raise ProvisioningError("Dataset manifest dataset_file does not match the configured dataset path")
    return {"dataset_version": version, "dataset_file": dataset_file}, hashlib.sha256(dataset_bytes).hexdigest()


def _load_model(config: EmbeddingConfig, encoder: TextEncoder | None) -> TextEncoder:
    if encoder is not None:
        return encoder
    try:
        return EmbeddingDetector._load_encoder(config.model_path)
    except Exception as exc:
        raise ProvisioningError(
            f"Could not load local embedding model at {config.model_path}: {type(exc).__name__}"
        ) from exc


def _load_examples(config: EmbeddingConfig) -> list[InjectionExample]:
    try:
        return LocalFaissVectorStore.load_dataset(config.dataset_path)
    except VectorStoreError as exc:
        raise ProvisioningError(str(exc)) from exc
    except UnicodeError as exc:
        raise ProvisioningError("Prompt-injection dataset must be UTF-8 encoded") from exc


def build_index(
    config: EmbeddingConfig | None = None,
    *,
    encoder: TextEncoder | None = None,
) -> Path:
    """Build or safely replace the local index, then validate its contents."""
    selected = config or EmbeddingConfig.from_env()
    manifest, dataset_hash = _dataset_manifest(selected)
    examples = _load_examples(selected)
    model = _load_model(selected, encoder)
    canonical_texts = [normalize_input(example.text).canonical_text for example in examples]
    try:
        vectors = _encode_texts(model, canonical_texts)
        provenance: Mapping[str, Any] = {
            "dataset_version": manifest["dataset_version"],
            "dataset_sha256": dataset_hash,
            "embedding_model_name": selected.model_name,
            "embedding_model_path": selected.model_path_descriptor,
        }
        store = LocalFaissVectorStore.from_embeddings(examples, vectors, provenance=provenance)
        store.save(selected.index_path)
    except (ValueError, VectorStoreError, OSError) as exc:
        raise ProvisioningError(f"Could not build local FAISS index: {exc}") from exc

    # Reload from disk and query a real indexed example so a partial/corrupt
    # write is detected before build exits successfully.
    validate_resources(selected, encoder=model)
    return selected.index_path


def validate_resources(
    config: EmbeddingConfig | None = None,
    *,
    encoder: TextEncoder | None = None,
) -> dict[str, Any]:
    """Validate local model, dataset, FAISS index, metadata, and a sample query."""
    selected = config or EmbeddingConfig.from_env()
    manifest, dataset_hash = _dataset_manifest(selected)
    examples = _load_examples(selected)
    model = _load_model(selected, encoder)
    try:
        store = LocalFaissVectorStore.load(selected.index_path)
    except VectorStoreError as exc:
        raise ProvisioningError(str(exc)) from exc

    metadata = store.metadata
    expected_values = {
        "dataset_version": manifest["dataset_version"],
        "dataset_sha256": dataset_hash,
        "embedding_model_name": selected.model_name,
        "embedding_model_path": selected.model_path_descriptor,
        "indexed_example_count": len(examples),
    }
    for field, expected in expected_values.items():
        if metadata.get(field) != expected:
            raise ProvisioningError(f"Index metadata {field} does not match configured local resources")
    if metadata.get("similarity_metric") != "cosine":
        raise ProvisioningError("Index metadata uses an unsupported similarity metric")
    if store.examples != tuple(examples):
        raise ProvisioningError("FAISS metadata examples do not match the configured dataset")

    sample = examples[0]
    query = normalize_input(sample.text).canonical_text
    try:
        vector = _encode_texts(model, [query])
    except Exception as exc:
        raise ProvisioningError(f"Could not encode validation example: {type(exc).__name__}") from exc
    if vector.shape[1] != store.dimension:
        raise ProvisioningError(
            f"Model embedding dimension {vector.shape[1]} does not match FAISS index dimension {store.dimension}"
        )
    try:
        matches = store.search(vector[0], top_k=1)
    except VectorStoreError as exc:
        raise ProvisioningError(f"Could not query validation example against FAISS index: {exc}") from exc
    if not matches:
        raise ProvisioningError("FAISS index did not return a validation match")

    return {
        "status": "valid",
        "dataset_version": manifest["dataset_version"],
        "dataset_sha256": dataset_hash,
        "embedding_model_name": selected.model_name,
        "embedding_model_path": selected.model_path_descriptor,
        "embedding_dimension": store.dimension,
        "similarity_metric": metadata["similarity_metric"],
        "indexed_example_count": len(store.examples),
        "validation_example_id": sample.example_id,
        "nearest_example_id": matches[0]["example_id"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"), help="build the local index or validate resources")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            path = build_index()
            result = {"status": "built_and_validated", "index_path": str(path)}
        else:
            result = validate_resources()
    except ProvisioningError as exc:
        print(f"Embedding resource {args.command} failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
