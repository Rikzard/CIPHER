# Classifier dataset

This versioned corpus contains 1,400 training records, 280 validation records,
and 280 classifier-test records. Each split is balanced between benign text
and generic prompt-injection behavior. Malicious records use the seven CIPHER
attack-category annotations; the model label remains binary (`0` benign,
`1` prompt injection).

The records are deterministic synthetic examples from fixed, hand-authored
phrase banks. No external dataset, citation, or license is claimed. Examples
cover candidate resumes, cover letters, project and technical documents, HR
review material, and benign technical/security questions. Benign hard
negatives use security vocabulary in ordinary technical and document contexts.
The text concerns evaluator manipulation behavior, not candidate quality or
hiring suitability.

Groups/template families are allocated explicitly to a single split. The
generator checks in
[`generate_classifier_data.py`](../../backend/training/generate_classifier_data.py)
and can reproduce all JSONL files and SHA-256 metadata with:

```powershell
uv run --offline python -m backend.training.generate_classifier_data
```

Validate schema, balance, category coverage, hard negatives, group and text
leakage, frozen-evaluation separation, hashes, and duplicate/quality signals:

```powershell
uv run --offline python -m backend.training.validate_classifier_data
```

The generator does not read `data/evaluation/`. Validation reads only its text
fields to check exact and normalized overlap. The frozen embedding evaluation
corpus is not used for classifier training. These synthetic examples are
workflow and research scaffolding, not a representative benchmark or evidence
of model performance. Do not train until the corpus receives independent
review and more diverse source material.
