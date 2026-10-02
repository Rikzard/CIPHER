# CIPHER Embedding Threshold Calibration

**Status:** Calibration candidate frozen before held-out evaluation.

## Dataset and model

- Dataset: `data/evaluation/calibration.jsonl` (28 examples; labels {'benign': 14, 'malicious': 14}).
- Model: `sentence-transformers/all-MiniLM-L6-v2` at `models/all-MiniLM-L6-v2`.
- Reference corpus: the unchanged, 16-example local FAISS index at `D:/CIPHER_2/data/embeddings/injection_examples.faiss`.
- Metric: cosine similarity from FAISS IndexFlatIP over L2-normalized vectors.
- Current threshold: **0.65**.
- Observed calibration range: 0.100814–0.664250.
- Internal comparison: cosine similarity `>= threshold`; public detector score is `round(clamp(similarity, 0, 1) * 100, 2)`. Calibration uses the unrounded cosine similarity, not the reported 0–100 score or fused risk score.

## Calibration procedure

The real locally provisioned Sentence Transformer and FAISS index were used. Each calibration text was normalized with CIPHER's normalizer, embedded, and queried against the fixed reference index. Candidate points span the observed score range, include observed values and intervening decision boundaries, and include the existing 0.65 threshold. The held-out dataset was not read or evaluated.

## Threshold performance table

| Threshold | TP | TN | FP | FN | Precision | Recall | F1 | FPR | FNR | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000000 | 14 | 0 | 14 | 0 | 0.500 | 1.000 | 0.667 | 1.000 | 0.000 | 0.500 |
| 0.100814 | 14 | 0 | 14 | 0 | 0.500 | 1.000 | 0.667 | 1.000 | 0.000 | 0.500 |
| 0.101772 | 14 | 1 | 13 | 0 | 0.519 | 1.000 | 0.683 | 0.929 | 0.000 | 0.536 |
| 0.102730 | 14 | 1 | 13 | 0 | 0.519 | 1.000 | 0.683 | 0.929 | 0.000 | 0.536 |
| 0.105430 | 14 | 2 | 12 | 0 | 0.538 | 1.000 | 0.700 | 0.857 | 0.000 | 0.571 |
| 0.108129 | 14 | 2 | 12 | 0 | 0.538 | 1.000 | 0.700 | 0.857 | 0.000 | 0.571 |
| 0.133098 | 14 | 3 | 11 | 0 | 0.560 | 1.000 | 0.718 | 0.786 | 0.000 | 0.607 |
| 0.158068 | 14 | 3 | 11 | 0 | 0.560 | 1.000 | 0.718 | 0.786 | 0.000 | 0.607 |
| 0.171696 | 14 | 4 | 10 | 0 | 0.583 | 1.000 | 0.737 | 0.714 | 0.000 | 0.643 |
| 0.185323 | 14 | 4 | 10 | 0 | 0.583 | 1.000 | 0.737 | 0.714 | 0.000 | 0.643 |
| 0.189217 | 14 | 5 | 9 | 0 | 0.609 | 1.000 | 0.757 | 0.643 | 0.000 | 0.679 |
| 0.193111 | 14 | 5 | 9 | 0 | 0.609 | 1.000 | 0.757 | 0.643 | 0.000 | 0.679 |
| 0.245168 | 14 | 6 | 8 | 0 | 0.636 | 1.000 | 0.778 | 0.571 | 0.000 | 0.714 |
| 0.297225 | 14 | 6 | 8 | 0 | 0.636 | 1.000 | 0.778 | 0.571 | 0.000 | 0.714 |
| 0.323281 | 14 | 7 | 7 | 0 | 0.667 | 1.000 | 0.800 | 0.500 | 0.000 | 0.750 |
| 0.349337 | 14 | 7 | 7 | 0 | 0.667 | 1.000 | 0.800 | 0.500 | 0.000 | 0.750 |
| 0.356372 | 14 | 8 | 6 | 0 | 0.700 | 1.000 | 0.824 | 0.429 | 0.000 | 0.786 |
| 0.363406 | 14 | 8 | 6 | 0 | 0.700 | 1.000 | 0.824 | 0.429 | 0.000 | 0.786 |
| 0.366115 | 14 | 9 | 5 | 0 | 0.737 | 1.000 | 0.848 | 0.357 | 0.000 | 0.821 |
| 0.368823 | 14 | 9 | 5 | 0 | 0.737 | 1.000 | 0.848 | 0.357 | 0.000 | 0.821 |
| 0.380856 | 14 | 10 | 4 | 0 | 0.778 | 1.000 | 0.875 | 0.286 | 0.000 | 0.857 |
| 0.392888 | 14 | 10 | 4 | 0 | 0.778 | 1.000 | 0.875 | 0.286 | 0.000 | 0.857 |
| 0.395815 | 13 | 10 | 4 | 1 | 0.765 | 0.929 | 0.839 | 0.286 | 0.071 | 0.821 |
| 0.398741 | 13 | 10 | 4 | 1 | 0.765 | 0.929 | 0.839 | 0.286 | 0.071 | 0.821 |
| 0.431060 | 13 | 11 | 3 | 1 | 0.812 | 0.929 | 0.867 | 0.214 | 0.071 | 0.857 |
| 0.463379 | 13 | 11 | 3 | 1 | 0.812 | 0.929 | 0.867 | 0.214 | 0.071 | 0.857 |
| 0.473382 | 12 | 11 | 3 | 2 | 0.800 | 0.857 | 0.828 | 0.214 | 0.143 | 0.821 |
| 0.483384 | 12 | 11 | 3 | 2 | 0.800 | 0.857 | 0.828 | 0.214 | 0.143 | 0.821 |
| 0.485915 | 12 | 12 | 2 | 2 | 0.857 | 0.857 | 0.857 | 0.143 | 0.143 | 0.857 |
| 0.488445 | 12 | 12 | 2 | 2 | 0.857 | 0.857 | 0.857 | 0.143 | 0.143 | 0.857 |
| 0.489907 | 11 | 12 | 2 | 3 | 0.846 | 0.786 | 0.815 | 0.143 | 0.214 | 0.821 |
| 0.491369 | 11 | 12 | 2 | 3 | 0.846 | 0.786 | 0.815 | 0.143 | 0.214 | 0.821 |
| 0.501998 | 10 | 12 | 2 | 4 | 0.833 | 0.714 | 0.769 | 0.143 | 0.286 | 0.786 |
| 0.512626 | 10 | 12 | 2 | 4 | 0.833 | 0.714 | 0.769 | 0.143 | 0.286 | 0.786 |
| 0.520171 | 9 | 12 | 2 | 5 | 0.818 | 0.643 | 0.720 | 0.143 | 0.357 | 0.750 |
| 0.527715 | 9 | 12 | 2 | 5 | 0.818 | 0.643 | 0.720 | 0.143 | 0.357 | 0.750 |
| 0.534381 | 9 | 13 | 1 | 5 | 0.900 | 0.643 | 0.750 | 0.071 | 0.357 | 0.786 |
| 0.541047 | 9 | 13 | 1 | 5 | 0.900 | 0.643 | 0.750 | 0.071 | 0.357 | 0.786 |
| 0.544049 | 8 | 13 | 1 | 6 | 0.889 | 0.571 | 0.696 | 0.071 | 0.429 | 0.750 |
| 0.547051 | 8 | 13 | 1 | 6 | 0.889 | 0.571 | 0.696 | 0.071 | 0.429 | 0.750 |
| 0.547386 | 7 | 13 | 1 | 7 | 0.875 | 0.500 | 0.636 | 0.071 | 0.500 | 0.714 |
| 0.547721 | 7 | 13 | 1 | 7 | 0.875 | 0.500 | 0.636 | 0.071 | 0.500 | 0.714 |
| 0.547823 | 6 | 13 | 1 | 8 | 0.857 | 0.429 | 0.571 | 0.071 | 0.571 | 0.679 |
| 0.547926 | 6 | 13 | 1 | 8 | 0.857 | 0.429 | 0.571 | 0.071 | 0.571 | 0.679 |
| 0.551572 | 6 | 14 | 0 | 8 | 1.000 | 0.429 | 0.600 | 0.000 | 0.571 | 0.714 |
| 0.555219 | 6 | 14 | 0 | 8 | 1.000 | 0.429 | 0.600 | 0.000 | 0.571 | 0.714 |
| 0.563603 | 5 | 14 | 0 | 9 | 1.000 | 0.357 | 0.526 | 0.000 | 0.643 | 0.679 |
| 0.571987 | 5 | 14 | 0 | 9 | 1.000 | 0.357 | 0.526 | 0.000 | 0.643 | 0.679 |
| 0.594536 | 4 | 14 | 0 | 10 | 1.000 | 0.286 | 0.444 | 0.000 | 0.714 | 0.643 |
| 0.617086 | 4 | 14 | 0 | 10 | 1.000 | 0.286 | 0.444 | 0.000 | 0.714 | 0.643 |
| 0.627338 | 3 | 14 | 0 | 11 | 1.000 | 0.214 | 0.353 | 0.000 | 0.786 | 0.607 |
| 0.637589 | 3 | 14 | 0 | 11 | 1.000 | 0.214 | 0.353 | 0.000 | 0.786 | 0.607 |
| 0.642429 | 2 | 14 | 0 | 12 | 1.000 | 0.143 | 0.250 | 0.000 | 0.857 | 0.571 |
| 0.647268 | 2 | 14 | 0 | 12 | 1.000 | 0.143 | 0.250 | 0.000 | 0.857 | 0.571 |
| 0.650000 | 1 | 14 | 0 | 13 | 1.000 | 0.071 | 0.133 | 0.000 | 0.929 | 0.536 |
| 0.655759 | 1 | 14 | 0 | 13 | 1.000 | 0.071 | 0.133 | 0.000 | 0.929 | 0.536 |
| 0.664250 | 1 | 14 | 0 | 13 | 1.000 | 0.071 | 0.133 | 0.000 | 0.929 | 0.536 |
| 1.000000 | 0 | 14 | 0 | 14 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.500 |

## Candidate selection

- **Primary maximum-F1 candidate:** `0.39288822` (F1 0.875; TP/TN/FP/FN 14/10/4/0). Exact F1 ties: `[0.3808557838201523, 0.39288821816444397]`. Thresholds within 0.02 F1: `[0.3808557838201523, 0.39288821816444397, 0.43105997145175934, 0.4633788764476776, 0.48591476678848267, 0.4884452819824219]`.
- **Lower-FPR alternative:** `0.55521929` (FPR 0.000, recall 0.429); strictly lower FPR than primary.
- **Higher-recall alternative:** `0.38085578` (recall 1.000, FPR 0.286); no candidate improved recall over primary; this is the maximum-recall point.

Primary selection criterion: maximum F1 on calibration data. Exact maximum-F1 thresholds were `0.38085578` and `0.39288822`; the pre-declared tie-break selects the highest, **`0.39288822`**, which is now frozen for the held-out evaluation. Tied and near-tied alternatives are shown because differences are not meaningful at this sample size. **The production threshold remains 0.65 and was not changed.**

## Limitations

The calibration set contains only 28 hand-authored project examples and is not representative of real-world traffic or all attack variants. Metrics are descriptive and uncertain; semantic similarity is not a calibrated probability. Category coverage is uneven. This result must not be treated as universal or statistically representative.

## Held-Out Evaluation

The **frozen calibration candidate `0.39288822`** was evaluated exactly once on `data/evaluation/test.jsonl` using the same real local model and unchanged FAISS index. No threshold sweep or tuning was performed on held-out data, and the production threshold remains `0.65`.

| Metric | Result |
|---|---:|
| TP | 9 |
| TN | 7 |
| FP | 3 |
| FN | 1 |
| Precision | 0.7500 |
| Recall | 0.9000 |
| F1 | 0.8182 |
| Accuracy | 0.8000 |
| False positive rate | 0.3000 |
| False negative rate | 0.1000 |

Held-out category counts are small; these results are descriptive and were not used to alter the candidate. Per-example evidence is in `data/evaluation/reports/heldout_threshold_result.json`.
