# CIPHER ML classifier

## Status and boundary

CIPHER has a standalone binary text classifier implementation under
`backend/detector/classifier/`. It is not selected by `ApplicationService` or
`POST /analyze`. No checkpoint is committed or trained by default, and the
versioned synthetic dataset is only a workflow scaffold. It provides no
quality or security guarantee.

The detector implements the existing `BaseDetector` contract. It accepts
`NormalizedContent`, emits a `DetectorResult`, and reports the probability of
the `prompt_injection` class as a 0–100 detector score. The default finding
threshold is 0.5 probability. The probability is marked uncalibrated unless a
separate calibration process later establishes otherwise. It makes no
ALLOW/REVIEW/BLOCK decision and contains no HR or business policy.

## Model and labels

The initial architecture fine-tunes `distilbert/distilbert-base-uncased` as a
two-class sequence classifier:

| Label | Meaning |
|---|---|
| `0` | benign text |
| `1` | prompt-injection behavior |

An optional `attack_category` annotation retains one of the dataset taxonomy
labels: `instruction_override`, `system_prompt_extraction`,
`role_manipulation`, `task_redirection`, `context_manipulation`,
`delimiter_manipulation`, and `obfuscation`. Categories are not model output
in this binary implementation. HR documents are examples of untrusted text;
there is no candidate scoring, hiring recommendation, or HR-specific logic.

Long input is encoded with a fast tokenizer in windows up to 512 tokens, with
256-token overlap. The highest per-window injection probability is returned,
with chunk count, total token count, highest-scoring window index, and canonical
and source character spans recorded as metadata. This max aggregation is
deterministic but can be sensitive to a single problematic chunk.

## Data and separation

`data/classifier/` contains JSONL train, validation, and held-out classifier
test splits plus a manifest. Each record has `id`, `group_id`, `text`, binary
`label`, optional `attack_category`, and `source_kind`. The validation command
checks schema, label/category consistency, duplicate and normalized-text
leakage, group/template split separation, manifest counts, and exact/normalized
overlap with the frozen evaluation text.

Only `train.jsonl` contributes gradients. `validation.jsonl` selects the best
epoch by validation loss. `data/classifier/test.jsonl` is for the separate
classifier evaluation command. The frozen embedding calibration and held-out
data in `data/evaluation/` are only read for text-leakage checks; their labels
are not read by classifier training or evaluation. Keep future classifier
calibration data separately designated and do not tune on a held-out test.

The current hand-authored splits are small, synthetic scaffolding. They need
broader source diversity and independent review before meaningful performance
claims.

## Local provisioning and commands

Classifier dependencies are opt-in; the regular API/test environment does not
install PyTorch or Transformers. `classifier` is for local inference and
`classifier-train` is for training. The optional model checkpoint is not
downloaded during API startup or inference.

One-time base model provisioning (requires network access):

```powershell
uv run --group classifier-train hf download distilbert/distilbert-base-uncased --local-dir models/classifier/base/distilbert-base-uncased
```

Validate the versioned data without heavyweight model dependencies:

```powershell
uv run python -m backend.training.validate_classifier_data
```

Train locally, using only train/validation and writing the ignored artifact to
`models/classifier/active/`:

```powershell
uv run --group classifier-train python -m backend.training.train_classifier
```

Evaluate once on the classifier's own `data/classifier/test.jsonl`:

```powershell
uv run --extra classifier python -m backend.training.evaluate_classifier
```

The runtime default is `models/classifier/active/`; callers can inject a
`ClassifierConfig(model_path=...)` for another local location. Model/tokenizer
loading uses `local_files_only=True`, disables remote code, and validates a
manifest and SHA-256 checksums for all files listed in the artifact. A missing,
malformed, incompatible, or checksum-invalid artifact returns
`available: false`, `score: null`, and a stable reason code. There is no Hub
fallback.

## Training and inference workflow

Training loads the base model/tokenizer from a local path, uses deterministic
seeds and CPU training, chooses the best epoch using validation loss, and
writes model/tokenizer assets plus `cipher_classifier_manifest.json`. The
manifest records the base checkpoint, tokenizer and library versions, label
mapping, chunking and aggregation settings, dataset version, timestamp,
training-data checksum, and model/tokenizer checksums. The manifest also
marks probabilities uncalibrated.

Inference loads and verifies resources once when the detector is constructed.
Each call tokenizes the supplied canonical text into overlapping windows,
runs local CPU inference, and returns bounded structured evidence. It does not
train, download, persist input, or make a policy decision. The caller remains
responsible for deciding how an unavailable detector is handled.

## Limitations

- No trained classifier artifact is included; a fresh checkout reports the
  detector unavailable until a developer provisions and trains one locally.
- The data is small and synthetic; metrics from it would not establish
  deployment suitability.
- Scores are not calibrated and are not integrated into risk fusion.
- Binary output does not predict attack categories.
- Window-max aggregation and fixed chunking need evaluation on longer and
  multilingual inputs.
- DOCX parsing/extraction, document provenance, and indirect-injection
  scanning are not implemented. A future extractor can pass extracted text
  through the same generic detector while retaining document/page/paragraph
  provenance outside the classifier.
