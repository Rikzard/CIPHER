"""Risk score fusion engine."""

from __future__ import annotations

from typing import List
from backend.models.contracts import DetectorResult, RiskAssessment


def fuse_risk(detector_results: List[DetectorResult]) -> RiskAssessment:
    """Combine detector results into a fused RiskAssessment.

    Distinguishes available vs. unavailable detectors. Avoids double counting.
    """
    contributing_versions = {}
    valid_scores = []

    for result in detector_results:
        contributing_versions[result.detector_name] = result.detector_version
        if result.available and result.score is not None:
            valid_scores.append(result.score)

    if valid_scores:
        fused_score = max(valid_scores)  # Simple max-score fusion baseline
    else:
        fused_score = 0.0

    fused_score = round(min(100.0, max(0.0, fused_score)), 1)

    if fused_score >= 80.0:
        band = "CRITICAL"
    elif fused_score >= 60.0:
        band = "HIGH"
    elif fused_score >= 30.0:
        band = "MEDIUM"
    else:
        band = "LOW"

    return RiskAssessment(
        fused_score=fused_score,
        risk_band=band,
        contributing_detector_versions=contributing_versions,
        calibration_version="1.0.0",
    )
