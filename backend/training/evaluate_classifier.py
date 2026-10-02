"""Evaluate a local classifier artifact on its classifier-specific test split."""

from __future__ import annotations

import json
from pathlib import Path

from backend.detector.classifier.ml_classifier import ClassifierConfig, MLClassifierDetector
from backend.evaluation.calibrate_embeddings import confusion_metrics
from backend.normalization.normalizer import normalize_input
from backend.training.train_classifier import _read


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    artifact = root / "models/classifier/active"
    manifest = json.loads((artifact / "cipher_classifier_manifest.json").read_text(encoding="utf-8"))
    detector = MLClassifierDetector(ClassifierConfig(model_path=artifact))
    if not detector.available:
        raise SystemExit("Local classifier artifact is unavailable")
    records = _read(root / "data/classifier/test.jsonl")
    scored = []
    for record in records:
        result = detector.analyze(normalize_input(record["text"]))
        if not result.available:
            raise SystemExit("Classifier inference failed on its classifier-specific test split")
        scored.append((record["label"], float(result.metadata["prompt_injection_probability"])))
    threshold = float(manifest.get("finding_threshold", 0.5))
    metrics = confusion_metrics(
        ("malicious" if label == 1 else "benign" for label, _ in scored),
        (probability for _, probability in scored),
        threshold,
    )
    print(json.dumps({"dataset": "data/classifier/test.jsonl", "threshold": threshold, "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
