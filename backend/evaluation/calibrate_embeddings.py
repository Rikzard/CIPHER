"""Calibration-only analysis for CIPHER's real local embedding detector.

This module reads only ``calibration.jsonl``. It never opens the held-out test
dataset. Pass ``--staging`` to write review files with a ``.staging`` suffix;
the held-out evaluation is a separate, later action after the candidate is
reviewed and frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from backend.detector.embeddings.embedding_detector import EmbeddingConfig, EmbeddingDetector
from backend.detector.embeddings.vector_store import LocalFaissVectorStore
from backend.evaluation.datasets import ATTACK_CATEGORIES
from backend.models.contracts import NormalizedContent
from backend.normalization.normalizer import normalize_input


def confusion_metrics(labels: Iterable[str], similarities: Iterable[float], threshold: float) -> dict[str, float | int]:
    """Calculate binary metrics using ``similarity >= threshold`` as positive."""
    truth = list(labels)
    scores = list(similarities)
    if not truth or len(truth) != len(scores):
        raise ValueError("labels and similarities must have equal non-zero lengths")
    if any(label not in {"benign", "malicious"} for label in truth):
        raise ValueError("labels must be benign or malicious")
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in scores):
        raise ValueError("similarities must be finite values between 0 and 1")

    tp = sum(label == "malicious" and score >= threshold for label, score in zip(truth, scores))
    tn = sum(label == "benign" and score < threshold for label, score in zip(truth, scores))
    fp = sum(label == "benign" and score >= threshold for label, score in zip(truth, scores))
    fn = sum(label == "malicious" and score < threshold for label, score in zip(truth, scores))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    benign_count = sum(label == "benign" for label in truth)
    malicious_count = sum(label == "malicious" for label in truth)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / benign_count if benign_count else 0.0,
        "false_negative_rate": fn / malicious_count if malicious_count else 0.0,
        "accuracy": (tp + tn) / len(truth),
        "benign_count": benign_count,
        "malicious_count": malicious_count,
        "benign_classified_benign": tn,
        "benign_classified_malicious": fp,
        "malicious_classified_malicious": tp,
        "malicious_classified_benign": fn,
    }


def threshold_candidates(similarities: Iterable[float], current_threshold: float) -> list[float]:
    """Generate decision boundaries from observed values plus score intervals."""
    scores = sorted(set(float(value) for value in similarities))
    if not scores:
        raise ValueError("at least one similarity is required")
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in scores):
        raise ValueError("similarities must be finite values between 0 and 1")
    if not 0.0 <= current_threshold <= 1.0:
        raise ValueError("current threshold must be between 0 and 1")
    candidates = {0.0, 1.0, float(current_threshold), *scores}
    candidates.update((left + right) / 2.0 for left, right in zip(scores, scores[1:]))
    return sorted(candidates)


def summarize_thresholds(
    examples: list[Mapping[str, Any]], current_threshold: float = 0.65
) -> dict[str, Any]:
    """Compute calibration metrics and select/report threshold alternatives."""
    if not examples:
        raise ValueError("calibration examples must not be empty")
    labels = [str(example["label"]) for example in examples]
    similarities = [float(example["similarity"]) for example in examples]
    if set(labels) != {"benign", "malicious"}:
        raise ValueError("calibration data must contain both labels")
    thresholds = threshold_candidates(similarities, current_threshold)
    rows = [
        {"threshold": threshold, **confusion_metrics(labels, similarities, threshold)}
        for threshold in thresholds
    ]
    max_f1 = max(float(row["f1"]) for row in rows)
    f1_winners = [row for row in rows if math.isclose(float(row["f1"]), max_f1, abs_tol=1e-12)]
    # When metrics tie, choose the most conservative boundary and report all
    # tied/near-tied values explicitly rather than implying precision.
    primary = max(f1_winners, key=lambda row: float(row["threshold"]))
    fpr_lower = [row for row in rows if float(row["false_positive_rate"]) < float(primary["false_positive_rate"])]
    if fpr_lower:
        minimum_fpr = min(float(row["false_positive_rate"]) for row in fpr_lower)
        lowest_fpr_rows = [row for row in fpr_lower if float(row["false_positive_rate"]) == minimum_fpr]
        lower_fpr = max(lowest_fpr_rows, key=lambda row: (float(row["recall"]), float(row["f1"]), float(row["threshold"])))
    else:
        lower_fpr = max(rows, key=lambda row: (-float(row["false_positive_rate"]), float(row["f1"]), float(row["threshold"])))
    recall_higher = [row for row in rows if float(row["recall"]) > float(primary["recall"])]
    higher_recall = (
        max(recall_higher, key=lambda row: (float(row["recall"]), float(row["f1"]), -float(row["false_positive_rate"])))
        if recall_higher
        else max(rows, key=lambda row: (float(row["recall"]), float(row["f1"]), -float(row["false_positive_rate"])))
    )
    near_tie_tolerance = 0.02
    near_ties = [row for row in rows if max_f1 - float(row["f1"]) <= near_tie_tolerance]
    return {
        "current_threshold": current_threshold,
        "observed_similarity_min": min(similarities),
        "observed_similarity_max": max(similarities),
        "positive_rule": "cosine_similarity >= threshold",
        "primary_selection_criterion": "maximum calibration F1; tied maximum uses the highest threshold and lists all ties",
        "near_tie_f1_tolerance": near_tie_tolerance,
        "rows": rows,
        "primary_f1_candidate": primary,
        "f1_tied_thresholds": [float(row["threshold"]) for row in f1_winners],
        "near_f1_thresholds": [float(row["threshold"]) for row in near_ties],
        "lower_fpr_alternative": lower_fpr,
        "higher_recall_alternative": higher_recall,
    }


def _read_calibration(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Calibration dataset not found: {path}")
    records: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or any(not isinstance(item.get(k), str) for k in ("id", "text", "label", "category")):
            raise ValueError(f"Malformed calibration record on line {line_number}")
        if item["label"] not in {"benign", "malicious"}:
            raise ValueError(f"Invalid calibration label on line {line_number}")
        if item["label"] == "benign" and item["category"] != "benign":
            raise ValueError(f"Invalid benign category on line {line_number}")
        if item["label"] == "malicious" and item["category"] not in ATTACK_CATEGORIES:
            raise ValueError(f"Invalid malicious category on line {line_number}")
        records.append({k: item[k] for k in ("id", "text", "label", "category")})
    if not records:
        raise ValueError("Calibration dataset contains no examples")
    return records


def collect_real_scores(
    records: list[Mapping[str, str]], detector: EmbeddingDetector
) -> list[dict[str, Any]]:
    """Run each input through the provisioned real detector and retain cosine evidence."""
    scored: list[dict[str, Any]] = []
    for record in records:
        result = detector.analyze(normalize_input(record["text"]))
        if not result.available:
            raise RuntimeError(f"Embedding detector unavailable for calibration record {record['id']}")
        matches = result.metadata.get("nearest_examples")
        if not isinstance(matches, list) or not matches:
            raise RuntimeError(f"No nearest-example evidence for calibration record {record['id']}")
        nearest = matches[0]
        cosine = float(nearest["cosine_similarity"])
        scored.append(
            {
                **record,
                "similarity": cosine,
                "nearest_reference_example_id": str(nearest["example_id"]),
                "nearest_reference_category": str(nearest["category"]),
                "reported_detector_score": result.score,
            }
        )
    return scored


def _rounded_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: round(value, 8) if isinstance(value, float) else value for key, value in row.items()}


def write_calibration_outputs(
    scored: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
    *,
    staging: bool,
    config: EmbeddingConfig,
) -> tuple[Path, Path, Path]:
    """Write calibration metrics and a reviewable candidate report only."""
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".staging" if staging else ""
    csv_path = output_dir / f"embedding_thresholds{suffix}.csv"
    json_path = output_dir / f"embedding_thresholds{suffix}.json"
    report_path = output_dir / f"THRESHOLD_CALIBRATION{suffix}.md"
    fields = ["threshold", "tp", "tn", "fp", "fn", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate", "accuracy", "benign_count", "malicious_count", "benign_classified_benign", "benign_classified_malicious", "malicious_classified_malicious", "malicious_classified_benign"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_rounded_row(row) for row in summary["rows"])

    payload = {
        "status": "calibration_candidate_staged" if staging else "calibration_complete",
        "dataset": "data/evaluation/calibration.jsonl",
        "dataset_count": len(scored),
        "dataset_label_counts": dict(sorted(Counter(row["label"] for row in scored).items())),
        "model": config.model_name,
        "model_path": config.model_path_descriptor,
        "reference_index": config.index_path.as_posix(),
        "reference_count": len(LocalFaissVectorStore.load_dataset(config.dataset_path)),
        "similarity_metric": "cosine similarity from FAISS IndexFlatIP over L2-normalized vectors",
        "detector_reported_score_mapping": "reported score = round(clamp(cosine_similarity, 0, 1) * 100, 2); threshold compares unrounded cosine similarity",
        "summary": {
            key: _rounded_row(value) if isinstance(value, dict) else value
            for key, value in summary.items()
            if key != "rows"
        },
        "threshold_rows": [_rounded_row(row) for row in summary["rows"]],
        "examples": [
            {
                **example,
                "predicted_labels_by_threshold": {
                    f"{float(row['threshold']):.8f}": (
                        "malicious" if float(example["similarity"]) >= float(row["threshold"]) else "benign"
                    )
                    for row in summary["rows"]
                },
            }
            for example in scored
        ],
        "heldout_evaluated": False,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_render_report(payload), encoding="utf-8")
    return csv_path, json_path, report_path


def _render_report(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    assert isinstance(summary, Mapping)
    rows = payload["threshold_rows"]
    table = "\n".join(
        f"| {row['threshold']:.6f} | {row['tp']} | {row['tn']} | {row['fp']} | {row['fn']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['false_positive_rate']:.3f} | {row['false_negative_rate']:.3f} | {row['accuracy']:.3f} |"
        for row in rows
    )
    primary = summary["primary_f1_candidate"]
    low = summary["lower_fpr_alternative"]
    high = summary["higher_recall_alternative"]
    return f"""# Threshold Calibration — Review Draft

**Status:** Calibration-only candidate; held-out evaluation has not been run.

## Dataset and model

- Dataset: `{payload['dataset']}` ({payload['dataset_count']} examples; labels {payload['dataset_label_counts']}).
- Model: `{payload['model']}` at `{payload['model_path']}`.
- Reference corpus: the unchanged, 16-example local FAISS index at `{payload['reference_index']}`.
- Metric: {payload['similarity_metric']}.
- Current threshold: **{summary['current_threshold']:.2f}**.
- Observed calibration range: {summary['observed_similarity_min']:.6f}–{summary['observed_similarity_max']:.6f}.
- Internal comparison: cosine similarity `>= threshold`; public detector score is `round(clamp(similarity, 0, 1) * 100, 2)`. Calibration uses the unrounded cosine similarity, not the reported 0–100 score or fused risk score.

## Calibration procedure

The real locally provisioned Sentence Transformer and FAISS index were used. Each calibration text was normalized with CIPHER's normalizer, embedded, and queried against the fixed reference index. Candidate points span the observed score range, include observed values and intervening decision boundaries, and include the existing 0.65 threshold. The held-out dataset was not read or evaluated.

## Threshold performance table

| Threshold | TP | TN | FP | FN | Precision | Recall | F1 | FPR | FNR | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

## Candidate selection

- **Primary maximum-F1 candidate:** `{primary['threshold']:.8f}` (F1 {primary['f1']:.3f}; TP/TN/FP/FN {primary['tp']}/{primary['tn']}/{primary['fp']}/{primary['fn']}). Exact F1 ties: `{summary['f1_tied_thresholds']}`. Thresholds within 0.02 F1: `{summary['near_f1_thresholds']}`.
- **Lower-FPR alternative:** `{low['threshold']:.8f}` (FPR {low['false_positive_rate']:.3f}, recall {low['recall']:.3f}); {('strictly lower FPR than primary' if low['false_positive_rate'] < primary['false_positive_rate'] else 'no candidate improved FPR over primary; this is the minimum-FPR point')}.
- **Higher-recall alternative:** `{high['threshold']:.8f}` (recall {high['recall']:.3f}, FPR {high['false_positive_rate']:.3f}); {('strictly higher recall than primary' if high['recall'] > primary['recall'] else 'no candidate improved recall over primary; this is the maximum-recall point')}.

Primary selection criterion: maximum F1 on calibration data. If exact ties occur, the highest tied threshold is selected as the displayed candidate; tied and near-tied alternatives are shown because differences are not meaningful at this sample size. This is a candidate only. **The production threshold remains 0.65.**

## Limitations

The calibration set contains only 28 hand-authored project examples and is not representative of real-world traffic or all attack variants. Metrics are descriptive and uncertain; semantic similarity is not a calibrated probability. Category coverage is uneven. This result must not be treated as universal or statistically representative.

## Held-Out Evaluation

Not run. The candidate must be reviewed and frozen before the separate held-out dataset is evaluated exactly once. Held-out results must not be used to retune the candidate.
"""


def run_calibration(output_dir: Path, *, staging: bool = True) -> tuple[Path, Path, Path]:
    """Run real-model calibration from calibration.jsonl only."""
    root = Path(__file__).resolve().parents[2]
    dataset_path = root / "data/evaluation/calibration.jsonl"
    config = EmbeddingConfig.from_env()
    detector = EmbeddingDetector(config)
    records = _read_calibration(dataset_path)
    scored = collect_real_scores(records, detector)
    summary = summarize_thresholds(scored, current_threshold=config.similarity_threshold)
    return write_calibration_outputs(scored, summary, output_dir, staging=staging, config=config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/evaluation/reports"))
    parser.add_argument("--staging", action="store_true", help="write review-draft names; recommended before candidate approval")
    args = parser.parse_args(argv)
    paths = run_calibration(args.output_dir, staging=args.staging)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
