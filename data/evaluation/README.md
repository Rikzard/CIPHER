# CIPHER evaluation datasets

`calibration.jsonl` and `test.jsonl` are project evaluation data for the
semantic embedding detector. Each file contains benign and malicious records
with an `id`, `text`, `label`, and `category`. Malicious records use the
detector's seven-category taxonomy. Benign records include security education,
quoted examples, ordinary technical and development requests, document tasks,
and authorized testing questions.

The two datasets each contain a balanced set of 14 benign and 14 malicious
calibration records, and 10 benign and 10 malicious held-out records. These are
small, hand-authored project fixtures; they are not representative of all
real-world prompt injections or benign inputs.

Run the schema and leakage checks from the repository root:

```powershell
uv run --offline --with-requirements backend/detector/embeddings/requirements.txt -- python -m backend.evaluation.datasets
```

The validator uses CIPHER's `normalize_input` for normalized-text comparisons.
It checks schema and category validity, ID uniqueness and disjointness, raw and
normalized duplicates within and across evaluation partitions, and raw or
normalized overlap with `data/embedding_examples.jsonl`. The manifest records
the counts, purpose, and separation rules.

Use calibration data for threshold analysis only. Keep `test.jsonl` untouched
while selecting a threshold, then use it once as a held-out assessment. Do not
move examples between these files based on observed scores without documenting
the resulting version change. The existing FAISS reference corpus remains
separate and is not modified by this evaluation data.
