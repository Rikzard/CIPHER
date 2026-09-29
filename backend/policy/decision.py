"""Decision engine for mapping risk assessments to policy actions."""

from __future__ import annotations

from typing import List
from backend.models.contracts import Decision, DetectorResult, RiskAssessment


def evaluate_decision(risk: RiskAssessment, detector_results: List[DetectorResult]) -> Decision:
    """Apply versioned policy rules to risk score and detector findings."""
    reason_codes: List[str] = []

    for det in detector_results:
        if det.available and det.findings:
            reason_codes.extend(det.findings)

    if risk.fused_score >= 70.0:
        action = "block"
        if not reason_codes:
            reason_codes.append("HIGH_RISK_SCORE_THRESHOLD_EXCEEDED")
    elif risk.fused_score >= 30.0:
        action = "flag"
        if not reason_codes:
            reason_codes.append("MODERATE_RISK_SUSPICION")
    else:
        action = "allow"
        if not reason_codes:
            reason_codes.append("CLEAN_EVALUATION")

    return Decision(
        action=action,
        policy_version="1.0.0",
        reason_codes=reason_codes,
    )
