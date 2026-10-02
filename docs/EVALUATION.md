# CIPHER evaluation data and threshold workflow

## Purpose and current status

`data/evaluation/calibration.jsonl` and `data/evaluation/test.jsonl` are
separate, labeled project evaluation datasets for studying the semantic
embedding score. They do not change the FAISS reference corpus, detector,
threshold, risk fusion, or API. Their size and hand-authored composition are
limited; they are not representative of all real-world prompt injections or
benign traffic, and they do not establish a security guarantee.

Both datasets use the schema `{ "id", "text", "label", "category" }`.
Labels are `benign` or `malicious`; benign records use category `benign`, and
malicious records use CIPHER's seven existing attack categories. Calibration
has 28 records (14 benign, 14 malicious). The held-out test set has 20 records
(10 benign, 10 malicious). Counts and the separation policy are recorded in
`data/evaluation/manifest.json`.

## Why two datasets

Calibration examples may be used to inspect score distributions and choose a
candidate threshold. The test examples have unseen wording and must remain
unused during that selection. After a candidate is fixed, Step 13C will run the
held-out set once to report detection outcomes, false-positive and
false-negative counts/rates, and results by attack category, alongside the
calibration analysis. Any threshold change must be based on the calibration
set and documented before the held-out results are examined. The current
threshold remains unchanged until that work is requested and the evaluation
results are reviewed.

Benign hard negatives matter because malicious-only reference examples can be
semantically close to legitimate questions about prompt injection, system
prompts, security testing, and hostile examples quoted for analysis. Ordinary
technical questions, development requests, document tasks, and benign
instructions are also included to sample non-security language. These examples
help expose false-positive behavior; they do not guarantee coverage.

## Leakage and validation

Dataset leakage occurs when calibration or held-out examples overlap with each
other or with the examples used to build the FAISS reference index. Exact
duplicates can inflate apparent performance; normalized duplicates can do the
same even when superficial whitespace or removable invisible characters
differ. The validator uses CIPHER's actual `normalize_input` function and
checks record schema, labels/categories, unique and disjoint IDs, exact and
normalized duplicates within/across the splits, and exact/normalized
duplicates against `data/embedding_examples.jsonl`.

Run validation from the repository root:

```powershell
uv run --offline --with-requirements backend/detector/embeddings/requirements.txt -- python -m backend.evaluation.datasets
```

The reference corpus is also analyzed with the same normalizer. It currently
has 16 raw unique examples and 16 unique canonical examples; the zero-width
character in `obfuscation-001` is removed during canonicalization, but the
resulting string does not equal `override-001` because the remainder of the
sentence differs. No normalized duplicate was found in the reference corpus.
This analysis does not modify the corpus or index.

## Step 13C and limitations

Step 13C is intended to analyze calibration scores first, select and document
a candidate threshold against explicit false-positive/false-negative goals,
then evaluate that fixed candidate on the held-out set. It must preserve the
separation and report per-category coverage. The held-out set must not be used
iteratively to tune a threshold.

The fixtures are small and manually authored, categories are unevenly sampled,
wording and annotator judgment introduce bias, and no independent adjudication
or broad adversarial generation has been applied. Their metrics will be
descriptive only, with substantial uncertainty. Detector scores are similarity
signals rather than calibrated probabilities. Indirect injection and
production traffic are outside the evaluation scope.
