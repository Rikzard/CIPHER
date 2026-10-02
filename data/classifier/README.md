# Classifier dataset

This directory contains a small, hand-authored development scaffold for the
generic binary prompt-injection classifier. Labels are `0` for benign and `1`
for prompt injection. Malicious records carry an optional `attack_category`;
benign records use `null`. `source_kind` identifies the source style, and
`group_id` keeps related templates within one split.

`train.jsonl` and `validation.jsonl` are the only files used by the training
script. `test.jsonl` is reserved for classifier evaluation. The frozen
`data/evaluation/` corpus is checked only for exact and normalized text
leakage and is never passed to model training or classifier evaluation.

These small synthetic examples are suitable for validating the workflow, not
for claiming model quality or production readiness. Expand and review the
dataset, preserve source-family separation, and calibrate probabilities on a
separate calibration set before using scores in a production fusion policy.
