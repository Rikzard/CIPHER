"""Validate classifier corpus schema, split isolation, balance, and quality signals."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from backend.evaluation.datasets import ATTACK_CATEGORIES
from backend.normalization.normalizer import normalize_input

HARD_NEGATIVE_TERMS = ("prompt", "system", "instruction", "ignore", "override", "administrator", "security")
SPLITS = (
    "train.jsonl",
    "validation.jsonl",
    "calibration.jsonl",
    "test.jsonl",
    "hr_document_final_test.jsonl",
)
SOURCE_TYPES = {"synthetic", "resume", "cover_letter", "candidate_document", "security_question"}
STRUCTURE_TYPES = {
    "paragraph", "heading", "table_header", "table_cell", "bullet", "short_fragment", "long_prose"
}
LENGTH_BUCKETS = {"1-8", "9-16", "17-32", "33-64", "65+"}
LABEL_LEAKAGE = re.compile(r"\b(?:label|class)\s*[:=]\s*(?:0|1|benign|malicious|prompt[_ -]?injection)\b", re.IGNORECASE)
IGNORE_PREVIOUS = re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
TOKENIZE = re.compile(r"[\w']+", re.UNICODE)


class ClassifierDatasetError(ValueError):
    """Raised when classifier data has invalid labels, metadata, or leakage."""


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
        required_strings = (
            "id", "group_id", "source_family", "template_family", "text", "source_kind",
            "source_type", "document_type", "structure_type", "length_bucket",
        )
        if any(not isinstance(item.get(key), str) or not item[key].strip() for key in required_strings):
            raise ClassifierDatasetError(f"Record in {path} line {line_number} has missing/empty required string fields")
        if item["group_id"] != item["source_family"] or item["group_id"] != item["template_family"]:
            raise ClassifierDatasetError(f"Record {item['id']} has inconsistent source/template family metadata")
        label = item.get("label")
        if not isinstance(label, int) or isinstance(label, bool) or label not in (0, 1):
            raise ClassifierDatasetError(f"Record {item['id']} must have label 0 or 1")
        category = item.get("attack_category")
        if label == 0 and category is not None:
            raise ClassifierDatasetError(f"Benign record {item['id']} must have null attack_category")
        if label == 1 and category not in ATTACK_CATEGORIES:
            raise ClassifierDatasetError(f"Malicious record {item['id']} has an invalid attack_category")
        if LABEL_LEAKAGE.search(item["text"]):
            raise ClassifierDatasetError(f"Record {item['id']} appears to state its label in text")
        if item["source_type"] not in SOURCE_TYPES:
            raise ClassifierDatasetError(f"Record {item['id']} has an invalid source_type")
        if item["structure_type"] not in STRUCTURE_TYPES:
            raise ClassifierDatasetError(f"Record {item['id']} has an invalid structure_type")
        if item["length_bucket"] not in LENGTH_BUCKETS:
            raise ClassifierDatasetError(f"Record {item['id']} has an invalid length_bucket")
        records.append(item)
    if not records:
        raise ClassifierDatasetError(f"Dataset split is empty: {path}")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ClassifierDatasetError(f"Duplicate IDs within {path.name}")
    return records


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
        # Only text is returned/read by callers; labels and scores are deliberately ignored.
        items.append({"text": value["text"]})
    return items


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(text: str) -> str:
    return normalize_input(text).canonical_text


def _split_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter("benign" if row["label"] == 0 else "malicious" for row in records)
    categories = Counter(row["attack_category"] for row in records if row["attack_category"] is not None)
    source_kinds = Counter(row["source_kind"] for row in records)
    source_types = Counter(row["source_type"] for row in records)
    structure_types = Counter(row["structure_type"] for row in records)
    length_buckets = Counter(row["length_bucket"] for row in records)
    length_buckets_by_label = {
        label_name: dict(sorted(Counter(row["length_bucket"] for row in records if row["label"] == label).items()))
        for label, label_name in ((0, "benign"), (1, "malicious"))
    }
    source_types_by_label = {
        label_name: dict(sorted(Counter(row["source_type"] for row in records if row["label"] == label).items()))
        for label, label_name in ((0, "benign"), (1, "malicious"))
    }
    structures_by_label = {
        label_name: dict(sorted(Counter(row["structure_type"] for row in records if row["label"] == label).items()))
        for label, label_name in ((0, "benign"), (1, "malicious"))
    }
    hard_negative_count = sum(
        row["label"] == 0 and any(term in row["text"].casefold() for term in HARD_NEGATIVE_TERMS)
        for row in records
    )
    hr_document_count = sum(row["source_kind"] in {"candidate_document", "hr_document"} for row in records)
    return {
        "count": len(records),
        "labels": dict(sorted(labels.items())),
        "attack_categories": dict(sorted(categories.items())),
        "source_kinds": dict(sorted(source_kinds.items())),
        "source_types": dict(sorted(source_types.items())),
        "source_types_by_label": source_types_by_label,
        "structure_types": dict(sorted(structure_types.items())),
        "structure_types_by_label": structures_by_label,
        "length_buckets_whitespace_estimate": dict(sorted(length_buckets.items())),
        "length_buckets_whitespace_estimate_by_label": length_buckets_by_label,
        "group_count": len({row["group_id"] for row in records}),
        "hard_negative_count": hard_negative_count,
        "hr_document_count": hr_document_count,
    }


def _quality_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    folded_texts = Counter(row["text"].casefold() for row in records)
    sentence_counts: Counter[str] = Counter()
    for row in records:
        for sentence in SENTENCE_SPLIT.split(row["text"]):
            normalized_sentence = _canonical(sentence).casefold().strip()
            if len(normalized_sentence) >= 60:
                sentence_counts[normalized_sentence] += 1
    repeated_sentences = [
        {"count": count, "sample": sentence[:180]}
        for sentence, count in sentence_counts.most_common()
        if count > 1
    ]
    malicious = [row for row in records if row["label"] == 1]
    malicious_casefold = Counter(row["text"].casefold() for row in malicious)
    near_duplicate_examples: list[dict[str, Any]] = []
    token_sets = [set(TOKENIZE.findall(_canonical(row["text"]).casefold())) for row in records]
    for left_index, left_tokens in enumerate(token_sets):
        for right_index in range(left_index + 1, len(token_sets)):
            # Similar resume headings and boilerplate can recur within a
            # training split; leakage is the concern when related text crosses
            # a data boundary.
            if records[left_index].get("_split") == records[right_index].get("_split"):
                continue
            right_tokens = token_sets[right_index]
            # Jaccard is unstable for very short, common document fields (for
            # example, "Degree" or "Year"). Exact/canonical duplicates are
            # still rejected for all lengths above.
            if min(len(left_tokens), len(right_tokens)) < 12:
                continue
            intersection = len(left_tokens & right_tokens)
            union = len(left_tokens | right_tokens)
            similarity = intersection / union
            if similarity >= 0.85:
                near_duplicate_examples.append({
                    "left_id": records[left_index]["id"],
                    "right_id": records[right_index]["id"],
                    "token_jaccard": round(similarity, 4),
                })
    return {
        "case_only_duplicate_text_groups": sum(count > 1 for count in folded_texts.values()),
        "repeated_long_sentence_groups": len(repeated_sentences),
        "repeated_long_sentence_examples": repeated_sentences[:5],
        "ignore_previous_instructions_count": sum(bool(IGNORE_PREVIOUS.search(row["text"])) for row in malicious),
        "malicious_casefold_text_groups": sum(count > 1 for count in malicious_casefold.values()),
        "near_duplicate_pair_count_at_token_jaccard_0_85": len(near_duplicate_examples),
        "near_duplicate_examples": near_duplicate_examples[:10],
    }


def validate_classifier_data(data_dir: str | Path, frozen_evaluation_dir: str | Path) -> dict[str, Any]:
    data_dir, frozen_dir = Path(data_dir), Path(frozen_evaluation_dir)
    splits = {name: _read(data_dir / name) for name in SPLITS}
    ids: dict[str, set[str]] = {}
    groups: dict[str, set[str]] = {}
    families: dict[str, set[str]] = {}
    texts: dict[str, set[str]] = {}
    canonical: dict[str, set[str]] = {}
    for name, records in splits.items():
        ids[name] = {row["id"] for row in records}
        groups[name] = {row["group_id"] for row in records}
        families[name] = {row["template_family"] for row in records}
        texts[name] = {row["text"] for row in records}
        canonical[name] = {_canonical(row["text"]) for row in records}
        if len(texts[name]) != len(records):
            raise ClassifierDatasetError(f"Exact text duplicates within {name}")
        if len(canonical[name]) != len(records):
            raise ClassifierDatasetError(f"Normalized duplicate texts within {name}")

    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            if ids[left] & ids[right]:
                raise ClassifierDatasetError(f"IDs overlap between {left} and {right}")
            if groups[left] & groups[right] or families[left] & families[right]:
                raise ClassifierDatasetError(f"Source/template groups overlap between {left} and {right}")
            if texts[left] & texts[right]:
                raise ClassifierDatasetError(f"Exact text duplicates between {left} and {right}")
            if canonical[left] & canonical[right]:
                raise ClassifierDatasetError(f"Normalized text duplicates between {left} and {right}")

    # Frozen sets contribute text only. Their labels are never opened or inspected.
    frozen_records = [
        record
        for file_name in (
            "calibration.jsonl",
            "test.jsonl",
            "classifier_short_probes.jsonl",
            "classifier_short_probes_postfit.jsonl",
        )
        for record in _read_frozen(frozen_dir / file_name)
    ]
    frozen_raw = {record["text"] for record in frozen_records}
    frozen_canonical = {_canonical(record["text"]) for record in frozen_records}
    overlap_raw = sum(len(text_set & frozen_raw) for text_set in texts.values())
    overlap_canonical = sum(len(text_set & frozen_canonical) for text_set in canonical.values())
    if overlap_raw or overlap_canonical:
        raise ClassifierDatasetError("Classifier texts overlap frozen evaluation text")

    # The short probe files are classifier-specific final checks. Reject
    # longer lexical near duplicates against training; short common questions
    # are excluded because token Jaccard is unstable at small token counts.
    probe_records = [
        record
        for file_name in ("classifier_short_probes.jsonl", "classifier_short_probes_postfit.jsonl")
        for record in _read_frozen(frozen_dir / file_name)
    ]
    train_records = splits["train.jsonl"]
    train_token_sets = [set(TOKENIZE.findall(_canonical(row["text"]).casefold())) for row in train_records]
    probe_token_sets = [set(TOKENIZE.findall(_canonical(row["text"]).casefold())) for row in probe_records]
    for train_row, train_tokens in zip(train_records, train_token_sets):
        if len(train_tokens) < 12:
            continue
        for probe_row, probe_tokens in zip(probe_records, probe_token_sets):
            if len(probe_tokens) < 12:
                continue
            similarity = len(train_tokens & probe_tokens) / len(train_tokens | probe_tokens)
            if similarity >= 0.85:
                raise ClassifierDatasetError(
                    f"Training record {train_row['id']} is near-duplicate to independent probe {probe_row['text'][:80]!r}"
                )

    manifest_path = data_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClassifierDatasetError("Classifier dataset manifest is missing or malformed") from exc
    if not isinstance(manifest, dict) or manifest.get("label_mapping") != {"benign": 0, "prompt_injection": 1}:
        raise ClassifierDatasetError("Classifier dataset manifest has an invalid label mapping")
    if not manifest.get("dataset_version") or not isinstance(manifest.get("source_metadata"), dict):
        raise ClassifierDatasetError("Classifier dataset manifest lacks version/source metadata")

    split_stats: dict[str, Any] = {}
    all_records = [dict(row, _split=name) for name, rows in splits.items() for row in rows]
    split_hashes = {name: _sha256((data_dir / name).read_bytes()) for name in SPLITS}
    for name, records in splits.items():
        stats = _split_stats(records)
        split_stats[name] = stats
        minimum_records = {
            "train.jsonl": 1000,
            "validation.jsonl": 200,
            "test.jsonl": 200,
            "calibration.jsonl": 100,
            "hr_document_final_test.jsonl": 100,
        }[name]
        if stats["count"] < minimum_records:
            raise ClassifierDatasetError(f"Dataset split is below its minimum size: {name}")
        if stats["labels"].get("benign", 0) != stats["labels"].get("malicious", 0):
            raise ClassifierDatasetError(f"Label distribution is not balanced in {name}")
        if set(stats["attack_categories"]) != set(ATTACK_CATEGORIES):
            raise ClassifierDatasetError(f"Attack category coverage is incomplete in {name}")
        category_minimum = 20 if name == "train.jsonl" else 5
        if any(count < category_minimum for count in stats["attack_categories"].values()):
            raise ClassifierDatasetError(f"An attack category is underrepresented in {name}")
        if stats["hard_negative_count"] < stats["labels"]["benign"] * 0.2:
            raise ClassifierDatasetError(f"Hard-negative coverage is too low in {name}")
        if stats["hr_document_count"] < stats["count"] * 0.2:
            raise ClassifierDatasetError(f"Candidate/HR document examples are too few in {name}")
        declared = manifest.get("splits", {}).get(name)
        if declared != stats:
            raise ClassifierDatasetError(f"Manifest statistics do not match {name}")
        if manifest.get("split_sha256", {}).get(name) != split_hashes[name]:
            raise ClassifierDatasetError(f"Manifest SHA-256 does not match {name}")

    total_sha = _sha256("".join(split_hashes[name] for name in sorted(split_hashes)).encode("ascii"))
    if manifest.get("dataset_sha256") != total_sha:
        raise ClassifierDatasetError("Manifest dataset SHA-256 does not match split files")
    if manifest.get("record_count") != len(all_records):
        raise ClassifierDatasetError("Manifest total record count does not match split files")
    group_count = len({row["group_id"] for row in all_records})
    if manifest.get("group_count") != group_count:
        raise ClassifierDatasetError("Manifest group count does not match split files")
    if manifest.get("totals") != _split_stats(all_records):
        raise ClassifierDatasetError("Manifest aggregate statistics do not match split files")

    quality = _quality_counts(all_records)
    if quality["ignore_previous_instructions_count"] > max(1, len([row for row in all_records if row["label"] == 1]) * 0.02):
        raise ClassifierDatasetError("Over-reliance on the 'ignore previous instructions' phrase")
    if quality["near_duplicate_pair_count_at_token_jaccard_0_85"]:
        raise ClassifierDatasetError("Near-duplicate text pairs exceed the 0.85 token Jaccard threshold")
    # Reused descriptive sentences are expected in synthetic document
    # templates; record the count for review, while rejecting full-record
    # exact/canonical and cross-split near duplicates above.
    return {
        "status": "valid",
        "dataset_version": manifest["dataset_version"],
        "split_counts": split_stats,
        "total_records": len(all_records),
        "total_groups": group_count,
        "hard_negative_count": sum(stats["hard_negative_count"] for stats in split_stats.values()),
        "hr_document_count": sum(stats["hr_document_count"] for stats in split_stats.values()),
        "frozen_evaluation_examples_checked": len(frozen_records),
        "frozen_evaluation_overlap": False,
        "template_groups_disjoint": True,
        "normalized_text_separation": True,
        "quality_checks": quality,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    result = validate_classifier_data(root / "data/classifier", root / "data/evaluation")
    report = {
        "status": result["status"],
        "dataset_version": result["dataset_version"],
        "total_records": result["total_records"],
        "total_groups": result["total_groups"],
        "splits": {
            name: {
                "count": stats["count"],
                "labels": stats["labels"],
                "category_counts": stats["attack_categories"],
                "source_types": stats["source_types"],
                "source_types_by_label": stats["source_types_by_label"],
                "structure_types": stats["structure_types"],
                "structure_types_by_label": stats["structure_types_by_label"],
                "length_buckets_whitespace_estimate": stats["length_buckets_whitespace_estimate"],
                "length_buckets_whitespace_estimate_by_label": stats["length_buckets_whitespace_estimate_by_label"],
                "groups": stats["group_count"],
            }
            for name, stats in result["split_counts"].items()
        },
        "hard_negative_count": result["hard_negative_count"],
        "hr_document_count": result["hr_document_count"],
        "frozen_evaluation_overlap": result["frozen_evaluation_overlap"],
        "quality_checks": result["quality_checks"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
