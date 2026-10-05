# CIPHER ML classifier

## Status and boundary

CIPHER's binary text classifier is implemented under
`backend/detector/classifier/` and is part of the application detector
pipeline. The locally trained checkpoint is an ignored development artifact;
fresh checkouts need a locally provisioned and trained checkpoint to make the
detector available. The synthetic dataset is a workflow and research scaffold
and provides no quality or security guarantee.

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

Dataset version 0.4.0 contains 2,560 training, 730 validation, 730 synthetic
test, 220 calibration-reserve, and 420 held-out HR-document records. Each split
is class-balanced. A separate 33-row independent short-probe file at
`data/evaluation/classifier_short_probes.jsonl` contains the required
regression questions and independently authored examples; none are training
data. The corpus includes short technical/security questions, short benign
statements, resume headings and table values, and malicious short/document
fragments across the seven categories. Records are deterministic synthetic
compositions; no external dataset or source license is claimed. The generator
is `backend.training.generate_classifier_data`.

Compared with dataset version 0.3.0, the corpus grew from 3,500 to 4,660
records: +600 train, +240 validation, +240 synthetic test, and +80 calibration
reserve rows. The HR split remains 420 records and stays separate from train.
The 40-row post-fit probe set was authored after the current model candidate
was frozen.

Dataset metadata records approximate whitespace length buckets by label,
source type, and structure type. Evaluation uses the saved fast tokenizer to
report exact WordPiece buckets and source/structure distributions by label.

The validator checks schema, label/category consistency, balanced classes,
coverage of all seven categories, hard-negative and candidate-document
coverage, group/template separation, exact/canonical text overlap across
splits and with frozen evaluation text, manifest distributions and SHA-256
values, label-in-text leakage, repeated long sentences, case-only duplicates,
and near duplicates at token-set Jaccard similarity 0.85.

Only `train.jsonl` contributes gradients. `validation.jsonl` selects the
checkpoint with the lowest cross-entropy and includes short questions and
document structures. The synthetic test, calibration reserve, HR-document
test, and independent short probes do not select the checkpoint. The frozen
`data/evaluation/` text is read only for leakage checks; labels are not
training input. Calibration data remains reserved, and independent probes are
scored only during evaluation. A second 40-example post-fit probe set was
authored after this candidate was frozen and evaluated once without further
checkpoint changes; it is separate from the required regression probes.

The expanded hand-authored splits remain synthetic scaffolding. They need
independent review and broader real-world source diversity before meaningful
performance claims. The current local artifact has been trained on the 0.4.0
training split; its scores do not establish deployment performance.

## Step 20C local evaluation

The current ignored local artifact was trained from the local DistilBERT base
checkpoint for one CPU epoch (`batch_size=16`, learning rate `2e-5`, seed 17).
The saved checkpoint is epoch 1 with validation cross-entropy 0.09727.
Checkpoint selection within the run uses lowest validation cross-entropy; no
test or probe split selects the epoch.

| Dataset | Accuracy | Precision | Recall | F1 | FPR | FNR | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation (730) | 97.53% | 95.30% | 100% | 97.59% | 4.93% | 0% | 347 / 18 / 0 / 365 |
| Synthetic test (730) | 99.86% | 99.73% | 100% | 99.86% | 0.27% | 0% | 364 / 1 / 0 / 365 |
| HR-document test (420) | 99.05% | 98.13% | 100% | 99.06% | 1.90% | 0% | 206 / 4 / 0 / 210 |
| Required short probes (33) | 100% | 100% | 100% | 100% | 0% | 0% | 22 / 0 / 0 / 11 |
| Post-fit probes (40) | 100% | 100% | 100% | 100% | 0% | 0% | 20 / 0 / 0 / 20 |

The required 33-row probe set revealed two false positives in the first
candidate. That result informed the addition of broader, varied explanation
question forms to the training data before the final candidate was trained.
Therefore its final 33/33 score is a regression result, not an untouched
independent estimate. The separate 40-row post-fit probe set was authored only
after the candidate was frozen and was not used for training or selection.
The validation split still has 18 false positives, including 15 short benign
questions; the HR-document test has four benign false positives, including two
table cells. No missed malicious examples occurred in these current splits.

Probability calibration is **not ready**. Scores remain explicitly
uncalibrated, and residual validation/document false positives plus the small,
synthetic-only evaluation sets do not support calibration or production claims.
Detailed baseline/candidate metrics, subgroup counts, misclassified examples,
and exact WordPiece/source/structure distributions are saved under
`data/evaluation/reports/classifier_step20c/`.

The previous Step 20A model manifest recorded the HR-document test hash as
`39a56c7325791d5ee2b3244be730b43de1de2127231f79fa7f8379c7aca68b0c`; the
current regenerated v0.4.0 HR split hash is
`7c8bb93295a6f06b7a7bce5504d1f3b2af83689c1367d5bb44858709f897203d`. It
remains held out and no HR row was moved into training, but the regenerated
test file is not byte-identical to the version used by Step 20A, so the
historical HR metrics are not a paired comparison. The Step 20A baseline in
the comparison report was rescored on the current v0.4.0 evaluation files.

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

Recreate the versioned JSONL files and manifest deterministically:

```powershell
uv run --offline python -m backend.training.generate_classifier_data
```

Dataset version 0.4.0 adds naturally phrased short technical/security
questions, varied benign short statements, and attacks in short and
document-framed forms. `train.jsonl` is the only gradient-training input;
`validation.jsonl` selects the checkpoint by lowest cross-entropy. The
calibration reserve, synthetic test, HR document test, and independent probes
are reserved from training. The HR test includes the previously observed clean
resume fragments. All examples are synthetic; results do not establish
production performance.

Train locally, using only train/validation and writing the ignored artifact to
`models/classifier/active/`:

```powershell
uv run --group classifier-train python -m backend.training.train_classifier
```

Evaluate on validation, the synthetic test, the reserved HR-document test, and
the independent short-probe set. Reports include short benign questions and
statements, table cells, headings, benign security questions, and malicious
short/document content, plus exact WordPiece length buckets and
source/structure distributions by label:

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
seeds, chooses the checkpoint with the lowest validation cross-entropy, and
writes model/tokenizer assets plus `cipher_classifier_manifest.json`. The
manifest records the base checkpoint, tokenizer and library versions, label
mapping, chunking and aggregation settings, dataset version, timestamp,
training-data checksum, and model/tokenizer checksums. The manifest also
marks probabilities uncalibrated.

Both training and inference call the existing
`normalize_input(...).canonical_text` before applying the same overlapping
tokenizer chunking. Training does not introduce a second normalizer.

Inference loads and verifies resources once when the detector is constructed.
Each call tokenizes the supplied canonical text into overlapping windows,
runs local CPU inference, and returns bounded structured evidence. It does not
train, download, persist input, or make a policy decision. The caller remains
responsible for deciding how an unavailable detector is handled.

## Limitations

- No trained classifier artifact is included; a fresh checkout reports the
  detector unavailable until a developer provisions and trains one locally.
- The expanded data is synthetic; even the HR-document final-test metrics do
  not establish deployment suitability.
- Scores remain uncalibrated. The classifier is integrated in the runtime
  pipeline, but calibration has not been performed.
- Binary output does not predict attack categories.
- Window-max aggregation and fixed chunking need evaluation on longer and
  multilingual inputs.
- DOCX parsing/extraction, document provenance, and indirect-injection
  scanning are not implemented. A future extractor can pass extracted text
  through the same generic detector while retaining document/page/paragraph
  provenance outside the classifier.
