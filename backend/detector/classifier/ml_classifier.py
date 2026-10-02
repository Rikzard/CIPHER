"""Local-only DistilBERT binary prompt-injection detector."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from backend.detector.base import BaseDetector
from backend.detector.classifier.chunking import FastTokenizer, TokenChunk, tokenize_overlapping_chunks
from backend.models.contracts import DetectorResult, NormalizedContent

CHECKPOINT_NAME = "distilbert/distilbert-base-uncased"
ARTIFACT_MANIFEST = "cipher_classifier_manifest.json"
ARTIFACT_VERSION_UNAVAILABLE = "distilbert-base-uncased:unavailable"


class InferenceRuntime(Protocol):
    tokenizer: FastTokenizer

    def predict_attack_probabilities(self, chunks: Sequence[TokenChunk]) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class ClassifierConfig:
    """Local checkpoint directory and binary finding threshold."""

    model_path: Path
    finding_threshold: float = 0.5
    max_length: int = 512
    stride: int = 256

    @classmethod
    def from_default(cls) -> "ClassifierConfig":
        repository_root = Path(__file__).resolve().parents[3]
        return cls(model_path=repository_root / "models" / "classifier" / "active")

    def __post_init__(self) -> None:
        if not 0.0 <= self.finding_threshold <= 1.0:
            raise ValueError("finding_threshold must be between 0 and 1")
        if self.max_length < 3 or self.stride < 0 or self.stride >= self.max_length - 2:
            raise ValueError("invalid classifier chunk length or stride")


class MLClassifierDetector(BaseDetector):
    """Classify normalized text as benign or prompt injection.

    Runtime resources are loaded from a local directory only. This detector
    returns evidence and a probability-derived score; it does not make policy
    decisions. Probabilities from a newly trained artifact are uncalibrated
    until a separate calibration artifact is produced.
    """

    def __init__(
        self,
        config: ClassifierConfig | None = None,
        *,
        runtime: InferenceRuntime | None = None,
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config or ClassifierConfig.from_default()
        self._runtime = runtime
        self._manifest: dict[str, Any] | None = dict(manifest) if manifest is not None else None
        self._load_error: str | None = None
        if self._manifest is None:
            try:
                self._manifest = self._load_manifest(self.config.model_path)
            except Exception as exc:
                self._load_error = self._safe_load_reason(exc)
        if self._load_error is None and self._manifest is not None:
            try:
                self._validate_manifest(self._manifest)
                if self._runtime is None:
                    self._verify_artifacts(self.config.model_path, self._manifest)
                    self._runtime = self._load_local_runtime(self.config.model_path)
            except Exception as exc:
                self._load_error = self._safe_load_reason(exc)

    @property
    def name(self) -> str:
        return "ml_classifier"

    @property
    def version(self) -> str:
        if self._manifest:
            return str(self._manifest.get("artifact_version", ARTIFACT_VERSION_UNAVAILABLE))
        return ARTIFACT_VERSION_UNAVAILABLE

    @property
    def available(self) -> bool:
        """Whether a compatible local model and inference runtime loaded."""
        return self._load_error is None and self._runtime is not None and self._manifest is not None

    @staticmethod
    def _safe_load_reason(exc: Exception) -> str:
        # Stable categories only. Never surface paths, exception text, or input.
        if isinstance(exc, FileNotFoundError):
            return "model_unavailable"
        if isinstance(exc, (json.JSONDecodeError, ValueError, TypeError)):
            return "model_manifest_invalid"
        return "model_load_failed"

    @staticmethod
    def _load_manifest(model_path: Path) -> dict[str, Any]:
        manifest_path = model_path / ARTIFACT_MANIFEST
        if not model_path.is_dir() or not manifest_path.is_file():
            raise FileNotFoundError("local classifier artifact is missing")
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("classifier artifact manifest must be a JSON object")
        return value

    @staticmethod
    def _validate_manifest(manifest: Mapping[str, Any]) -> None:
        required = (
            "artifact_version",
            "model_name",
            "model_checkpoint",
            "label_mapping",
            "max_sequence_length",
            "chunk_stride",
            "training_dataset_version",
            "training_timestamp_utc",
            "training_data_sha256",
            "tokenizer_version",
        )
        if any(not isinstance(manifest.get(key), str) or not str(manifest[key]).strip() for key in required if key not in {"max_sequence_length", "chunk_stride", "label_mapping"}):
            raise ValueError("classifier artifact manifest is incomplete")
        if manifest.get("model_checkpoint") != CHECKPOINT_NAME:
            raise ValueError("classifier checkpoint is incompatible")
        if manifest.get("label_mapping") != {"benign": 0, "prompt_injection": 1}:
            raise ValueError("classifier labels are incompatible")
        if manifest.get("max_sequence_length") != 512 or manifest.get("chunk_stride") != 256:
            raise ValueError("classifier chunk configuration is incompatible")
        digest = manifest.get("training_data_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("classifier training checksum is invalid")
        checksums = manifest.get("artifact_files_sha256")
        if not isinstance(checksums, dict) or not checksums:
            raise ValueError("classifier artifact checksums are missing")
        if any(not isinstance(name, str) or not name or not isinstance(value, str) or len(value) != 64 for name, value in checksums.items()):
            raise ValueError("classifier artifact checksums are malformed")

    @staticmethod
    def _verify_artifacts(model_path: Path, manifest: Mapping[str, Any]) -> None:
        for relative_name, expected in manifest["artifact_files_sha256"].items():
            path = (model_path / relative_name).resolve()
            if model_path.resolve() not in path.parents or not path.is_file():
                raise FileNotFoundError("classifier artifact file is missing")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError("classifier artifact checksum mismatch")

    @staticmethod
    def _load_local_runtime(model_path: Path) -> InferenceRuntime:
        if not model_path.is_dir():
            raise FileNotFoundError("local classifier checkpoint directory is missing")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("classifier runtime dependencies are not installed") from exc

        # These arguments prohibit Hub lookup/downloads. Runtime code never
        # resolves model names remotely or invokes a training path.
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False, use_fast=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        )
        if int(model.config.num_labels) != 2:
            raise ValueError("classifier checkpoint must have exactly two labels")
        model.to("cpu")
        model.eval()
        return _TorchRuntime(tokenizer, model, torch)

    def analyze(self, content: NormalizedContent) -> DetectorResult:
        if self._load_error or self._runtime is None or self._manifest is None:
            return self._unavailable(self._load_error or "model_unavailable")
        if not isinstance(content, NormalizedContent):
            return self._unavailable("invalid_input")
        if not content.canonical_text.strip():
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                available=True,
                score=0.0,
                findings=[],
                metadata=self._base_metadata() | {"status": "evaluated", "chunk_count": 0, "token_count": 0},
            )
        try:
            chunks, total_tokens = tokenize_overlapping_chunks(
                content.canonical_text,
                self._runtime.tokenizer,
                max_length=self.config.max_length,
                stride=self.config.stride,
            )
            probabilities = self._runtime.predict_attack_probabilities(chunks)
            if len(chunks) != len(probabilities) or not probabilities:
                raise ValueError("runtime returned an invalid chunk probability count")
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
                raise ValueError("runtime returned probabilities outside [0, 1]")
        except Exception:
            return self._unavailable("inference_failed")

        best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        probability = probabilities[best_index]
        highest = chunks[best_index]
        canonical_span = {"start": highest.canonical_start, "end": highest.canonical_end}
        source_span = _source_span(content, highest.canonical_start, highest.canonical_end)
        is_injection = probability >= self.config.finding_threshold
        findings = [
            "[ML-001] Prompt-injection behavior detected by the binary classifier"
        ] if is_injection else []
        metadata = self._base_metadata() | {
            "status": "evaluated",
            "prompt_injection_probability": round(probability, 8),
            "probability_calibrated": bool(self._manifest.get("probability_calibrated", False)),
            "predicted_label": "prompt_injection" if is_injection else "benign",
            "finding_threshold": self.config.finding_threshold,
            "aggregation": "max_chunk_probability",
            "chunk_count": len(chunks),
            "token_count": total_tokens,
            "highest_scoring_chunk_index": highest.chunk_index,
            "highest_scoring_chunk_canonical_span": canonical_span,
            "highest_scoring_chunk_source_span": source_span,
        }
        return DetectorResult(
            detector_name=self.name,
            detector_version=self.version,
            available=True,
            score=round(probability * 100.0, 2),
            findings=findings,
            metadata=metadata,
        )

    def _base_metadata(self) -> dict[str, Any]:
        assert self._manifest is not None
        return {
            "model_name": self._manifest["model_name"],
            "model_checkpoint": self._manifest["model_checkpoint"],
            "artifact_version": self._manifest["artifact_version"],
            "label_mapping": self._manifest["label_mapping"],
            "max_sequence_length": self.config.max_length,
            "chunk_stride": self.config.stride,
        }

    def _unavailable(self, reason: str) -> DetectorResult:
        return DetectorResult(
            detector_name=self.name,
            detector_version=self.version,
            available=False,
            score=None,
            findings=[],
            metadata={"status": "unavailable", "reason": reason},
        )


class _TorchRuntime:
    def __init__(self, tokenizer: FastTokenizer, model: Any, torch: Any) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.torch = torch

    def predict_attack_probabilities(self, chunks: Sequence[TokenChunk]) -> list[float]:
        probabilities: list[float] = []
        with self.torch.inference_mode():
            for chunk in chunks:
                inputs = {
                    "input_ids": self.torch.tensor([chunk.input_ids], dtype=self.torch.long),
                    "attention_mask": self.torch.tensor([chunk.attention_mask], dtype=self.torch.long),
                }
                logits = self.model(**inputs).logits
                probabilities.append(float(self.torch.softmax(logits, dim=-1)[0, 1].item()))
        return probabilities


def _source_span(content: NormalizedContent, start: int, end: int) -> dict[str, int] | None:
    spans = content.canonical_source_spans[start:end]
    if not spans:
        return None
    return {"start": min(span[0] for span in spans), "end": max(span[1] for span in spans)}
