"""Schema and leakage validation for calibration and held-out datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from backend.detector.embeddings.vector_store import LocalFaissVectorStore
from backend.normalization.normalizer import normalize_input

ATTACK_CATEGORIES = frozenset(
    {
        "instruction_override",
        "system_prompt_extraction",
        "role_manipulation",
        "task_redirection",
        "context_manipulation",
        "delimiter_manipulation",
        "obfuscation",
    }
)
LABELS = frozenset({"benign", "malicious"})


class DatasetValidationError(ValueError):
    """Raised when evaluation data is malformed or leaks across partitions."""


def _read_records(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise DatasetValidationError(f"Dataset file not found: {path}")
    records: list[dict[str, str]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                raw: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(f"Invalid JSON in {path} line {line_number}") from exc
            if not isinstance(raw, dict):
                raise DatasetValidationError(f"Record in {path} line {line_number} must be an object")
            if any(not isinstance(raw.get(field), str) for field in ("id", "text", "label", "category")):
                raise DatasetValidationError(
                    f"Record in {path} line {line_number} requires string id, text, label, category"
                )
            record = {field: raw[field] for field in ("id", "text", "label", "category")}
            if not record["id"].strip() or not record["text"].strip():
                raise DatasetValidationError(f"Record in {path} line {line_number} has empty id or text")
            if record["label"] not in LABELS:
                raise DatasetValidationError(f"Record {record['id']} has invalid label {record['label']!r}")
            if record["label"] == "benign" and record["category"] != "benign":
                raise DatasetValidationError(f"Benign record {record['id']} must use category 'benign'")
            if record["label"] == "malicious" and record["category"] not in ATTACK_CATEGORIES:
                raise DatasetValidationError(f"Malicious record {record['id']} has invalid attack category")
            records.append(record)
    except UnicodeError as exc:
        raise DatasetValidationError(f"Dataset must be UTF-8: {path}") from exc
    if not records:
        raise DatasetValidationError(f"Dataset is empty: {path}")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise DatasetValidationError(f"Duplicate IDs in dataset: {path}")
    return records


def _normalized(text: str) -> str:
    return normalize_input(text).canonical_text


def validate_datasets(
    calibration_path: str | Path,
    test_path: str | Path,
    reference_path: str | Path,
) -> dict[str, Any]:
    """Validate both datasets, their separation, and reference-corpus leakage."""
    calibration = _read_records(Path(calibration_path))
    test = _read_records(Path(test_path))
    reference = LocalFaissVectorStore.load_dataset(reference_path)

    calibration_ids = {record["id"] for record in calibration}
    test_ids = {record["id"] for record in test}
    if calibration_ids & test_ids:
        raise DatasetValidationError("Calibration and test IDs overlap")

    calibration_text = {record["text"] for record in calibration}
    test_text = {record["text"] for record in test}
    if calibration_text & test_text:
        raise DatasetValidationError("Calibration and test contain exact duplicate text")

    calibration_normalized = {_normalized(record["text"]) for record in calibration}
    test_normalized = {_normalized(record["text"]) for record in test}
    if calibration_normalized & test_normalized:
        raise DatasetValidationError("Calibration and test contain normalized duplicate text")
    if len(calibration_normalized) != len(calibration) or len(test_normalized) != len(test):
        raise DatasetValidationError("A dataset contains normalized duplicate text")

    reference_text = {example.text for example in reference}
    if calibration_text & reference_text or test_text & reference_text:
        raise DatasetValidationError("Evaluation data exactly duplicates the FAISS reference corpus")
    reference_normalized = {_normalized(example.text) for example in reference}
    if calibration_normalized & reference_normalized or test_normalized & reference_normalized:
        raise DatasetValidationError("Evaluation data contains normalized FAISS reference text")

    return {
        "status": "valid",
        "calibration_count": len(calibration),
        "test_count": len(test),
        "calibration_labels": dict(sorted(Counter(r["label"] for r in calibration).items())),
        "test_labels": dict(sorted(Counter(r["label"] for r in test).items())),
        "malicious_category_counts": dict(
            sorted(Counter(r["category"] for r in calibration + test if r["label"] == "malicious").items())
        ),
        "reference_count": len(reference),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    try:
        report = validate_datasets(
            root / "data/evaluation/calibration.jsonl",
            root / "data/evaluation/test.jsonl",
            root / "data/embedding_examples.jsonl",
        )
    except DatasetValidationError as exc:
        raise SystemExit(f"Evaluation dataset validation failed: {exc}") from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
