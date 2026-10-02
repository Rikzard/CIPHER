"""Synthetic tests for embedding calibration calculations (no model download)."""

from __future__ import annotations

import unittest

from backend.evaluation.calibrate_embeddings import confusion_metrics, summarize_thresholds, threshold_candidates


class TestEmbeddingCalibrationMetrics(unittest.TestCase):
    def test_confusion_metrics_are_calculated_from_raw_similarity_boundary(self) -> None:
        metrics = confusion_metrics(
            ["malicious", "malicious", "benign", "benign"],
            [0.9, 0.4, 0.7, 0.2],
            0.65,
        )
        self.assertEqual((metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]), (1, 1, 1, 1))
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["false_positive_rate"], 0.5)
        self.assertEqual(metrics["false_negative_rate"], 0.5)
        self.assertEqual(metrics["accuracy"], 0.5)

    def test_threshold_candidates_cover_distribution_and_current_baseline(self) -> None:
        candidates = threshold_candidates([0.2, 0.5, 0.9], 0.65)
        self.assertIn(0.2, candidates)
        self.assertIn(0.9, candidates)
        self.assertIn(0.65, candidates)
        self.assertIn(0.35, candidates)
        self.assertIn(0.7, candidates)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in candidates))

    def test_summary_reports_primary_and_tradeoff_candidates(self) -> None:
        examples = [
            {"label": "malicious", "similarity": 0.9},
            {"label": "malicious", "similarity": 0.7},
            {"label": "benign", "similarity": 0.8},
            {"label": "benign", "similarity": 0.2},
        ]
        summary = summarize_thresholds(examples, 0.65)
        self.assertEqual(summary["current_threshold"], 0.65)
        self.assertIn("primary_f1_candidate", summary)
        self.assertIn("lower_fpr_alternative", summary)
        self.assertIn("higher_recall_alternative", summary)
        self.assertTrue(summary["rows"])

    def test_invalid_metric_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            confusion_metrics(["unknown"], [0.5], 0.5)
        with self.assertRaises(ValueError):
            confusion_metrics(["benign"], [1.1], 0.5)
        with self.assertRaises(ValueError):
            threshold_candidates([], 0.65)


if __name__ == "__main__":
    unittest.main()
