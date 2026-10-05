# Classifier dataset

Dataset version 0.4.0 contains 2,560 training records, 730 validation
records, 730 synthetic-test records, 220 calibration-reserve records, and a
separate 420-record `hr_document_final_test.jsonl`. Each split is
label-balanced. The HR document test includes resume and cover-letter
headings, education and certification fragments, table-like cells, bullets,
paragraphs, longer prose, and benign AI/security discussion. It retains the
clean-resume examples used in the previous evaluation and is reserved from
training.

The data is deterministic synthetic text from hand-authored phrase banks; no
external dataset, source license, or production representativeness is claimed.
Version 0.4.0 expands the previous 3,500-row version to 4,660 rows, adding
short questions/statements and short attack/document fragments to train,
validation, synthetic test, and calibration reserve. The HR-document split
remains separate at 420 records and is never used as training input.
It covers technical/security questions, HR document fragments, and malicious
prompt-injection behavior across seven attack categories. Benign content uses
security vocabulary in educational and ordinary technical contexts. The text
concerns untrusted document behavior, not candidate quality or hiring
suitability.

The 33-record required regression probe set is maintained separately in
`data/evaluation/classifier_short_probes.jsonl`. It contains the required
regression probes plus additional independently authored examples and is not
used for training or checkpoint selection.

A second 40-record post-fit probe set was authored after the current candidate
was frozen and evaluated once without further model changes:
`data/evaluation/classifier_short_probes_postfit.jsonl`.

Groups/template families are allocated to one split. Metadata records the
binary label, source/document/structure types, approximate whitespace length
bucket by label, and malicious attack category. WordPiece length buckets are
measured using the saved tokenizer during evaluation. Short examples include
benign questions and statements, familiar resume headings and table fields,
and malicious phrases in comparable document structures.

Generate the JSONL files and SHA-256 manifest deterministically with:

```powershell
uv run --offline python -m backend.training.generate_classifier_data
```

Validate schema, class balance, category coverage, group separation,
exact/canonical overlap, frozen-set separation, hashes, and duplicate signals,
including near-duplicate checks against short-probe files:

```powershell
uv run --offline python -m backend.training.validate_classifier_data
```

Training uses only `train.jsonl`; `validation.jsonl` selects the checkpoint by
lowest validation cross-entropy. `calibration.jsonl`, `test.jsonl`,
`hr_document_final_test.jsonl`, and the independent probes remain held out from
gradient training. Repeated sentences are reported because some synthetic
templates deliberately share neutral context; the corpus is a workflow and
research scaffold, not a representative benchmark or evidence of production
performance.
