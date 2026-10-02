"""Validate classifier split schema and template/source-family separation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from backend.evaluation.datasets import ATTACK_CATEGORIES
from backend.normalization.normalizer import normalize_input


class ClassifierDatasetError(ValueError):
    """Raised when classifier data has invalid labels or split leakage."""


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ClassifierDatasetError(f"Dataset split not found: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ClassifierDatasetError(f"Invalid JSON in {path} line {line_number}") from exc
        if not isinstance(item, dict):
            raise ClassifierDatasetError(f"Record in {path} line {line_number} must be an object")
        required_strings = ("id", "group_id", "text", "source_kind")
        if any(not isinstance(item.get(key), str) or not item[key].strip() for key in required_strings):
            raise ClassifierDatasetError(f"Record in {path} line {line_number} has missing/empty text fields")
        label = item.get("label")
        if not isinstance(label, int) or isinstance(label, bool) or label not in (0, 1):
            raise ClassifierDatasetError(f"Record {item['id']} must have label 0 or 1")
        category = item.get("attack_category")
        if label == 0 and category is not None:
            raise ClassifierDatasetError(f"Benign record {item['id']} must have null attack_category")
        if label == 1 and category not in ATTACK_CATEGORIES:
            raise ClassifierDatasetError(f"Malicious record {item['id']} has an invalid attack_category")
        records.append(item)
    if not records:
        raise ClassifierDatasetError(f"Dataset split is empty: {path}")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ClassifierDatasetError(f"Duplicate IDs within {path.name}")
    return records


def validate_classifier_data(data_dir: str | Path, frozen_evaluation_dir: str | Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    frozen_dir = Path(frozen_evaluation_dir)
    split_names = ("train.jsonl", "validation.jsonl", "test.jsonl")
    splits = {name: _read(data_dir / name) for name in split_names}
    split_ids: dict[str, set[str]] = {}
    split_groups: dict[str, set[str]] = {}
    split_texts: dict[str, set[str]] = {}
    split_canonical: dict[str, set[str]] = {}
    for name, records in splits.items():
        split_ids[name] = {record["id"] for record in records}
        split_groups[name] = {record["group_id"] for record in records}
        split_texts[name] = {record["text"] for record in records}
        split_canonical[name] = {normalize_input(record["text"]).canonical_text for record in records}
        if len(split_canonical[name]) != len(records):
            raise ClassifierDatasetError(f"Normalized duplicate texts within {name}")

    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            if split_ids[left] & split_ids[right]:
                raise ClassifierDatasetError(f"IDs overlap between {left} and {right}")
            if split_groups[left] & split_groups[right]:
                raise ClassifierDatasetError(f"Source/template group_ids overlap between {left} and {right}")
            if split_texts[left] & split_texts[right]:
                raise ClassifierDatasetError(f"Exact text duplicates between {left} and {right}")
            if split_canonical[left] & split_canonical[right]:
                raise ClassifierDatasetError(f"Normalized text duplicates between {left} and {right}")

    frozen_records = []
    for name in ("calibration.jsonl", "test.jsonl"):
        frozen_records.extend(_read_frozen(frozen_dir / name))
    frozen_raw = {record["text"] for record in frozen_records}
    frozen_canonical = {normalize_input(record["text"]).canonical_text for record in frozen_records}
    for name in split_names:
        if split_texts[name] & frozen_raw:
            raise ClassifierDatasetError(f"{name} exactly overlaps frozen evaluation text")
        if split_canonical[name] & frozen_canonical:
            raise ClassifierDatasetError(f"{name} overlaps normalized frozen evaluation text")

    manifest_path = data_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClassifierDatasetError("Classifier dataset manifest is missing or malformed") from exc
    if not isinstance(manifest, dict) or manifest.get("label_mapping") != {"benign": 0, "prompt_injection": 1}:
        raise ClassifierDatasetError("Classifier dataset manifest has an invalid label mapping")
    count_summary = {}
    for name, records in splits.items():
        counts = Counter("benign" if record["label"] == 0 else "malicious" for record in records)
        count_summary[name] = {"count": len(records), **dict(sorted(counts.items()))}
        declared = manifest.get("splits", {}).get(name)
        if not isinstance(declared, dict) or declared.get("count") != len(records):
            raise ClassifierDatasetError(f"Manifest count does not match {name}")
    return {
        "status": "valid",
        "split_counts": count_summary,
        "frozen_evaluation_examples_checked": len(frozen_records),
        "template_groups_disjoint": True,
        "normalized_text_separation": True,
    }


def _read_frozen(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ClassifierDatasetError(f"Frozen evaluation data not found: {path}")
    items: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ClassifierDatasetError(f"Malformed frozen evaluation data {path.name}:{line_number}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("text"), str):
            raise ClassifierDatasetError(f"Malformed frozen evaluation record in {path.name}:{line_number}")
        items.append(value)
    return items


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    result = validate_classifier_data(root / "data/classifier", root / "data/evaluation")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
