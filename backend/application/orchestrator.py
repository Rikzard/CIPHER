"""Application service layer orchestrating the prompt injection evaluation workflow."""

from __future__ import annotations

from typing import List, Sequence
from pydantic import BaseModel, ConfigDict

from backend.detector.base import BaseDetector
from backend.detector.embeddings import EmbeddingDetector
from backend.detector.rules import RuleDetector
from backend.models.contracts import (
    AnalysisInput,
    Decision,
    DetectorResult,
    NormalizedContent,
    RiskAssessment,
)
from backend.normalization.normalizer import normalize_input
from backend.policy.decision import evaluate_decision
from backend.risk.fusion import fuse_risk


class AnalysisWorkflowResult(BaseModel):
    """Complete container for orchestration workflow output."""

    model_config = ConfigDict(frozen=True)

    input_data: AnalysisInput
    normalized: NormalizedContent
    detector_results: List[DetectorResult]
    risk_assessment: RiskAssessment
    decision: Decision


class ApplicationService:
    """Orchestrates normalization, detector execution, risk fusion, and policy decisions."""

    def __init__(self, detectors: Sequence[BaseDetector] | None = None) -> None:
        if detectors is not None:
            self._detectors = list(detectors)
        else:
            self._detectors = [
                RuleDetector(),
                EmbeddingDetector(),
            ]

    def analyze_prompt(self, input_data: AnalysisInput) -> AnalysisWorkflowResult:
        """Run complete workflow: request -> normalization -> detector execution -> risk assessment -> decision -> response."""
        # 1. Normalization
        normalized = normalize_input(input_data)

        # 2. Detector execution
        detector_results: List[DetectorResult] = []
        for detector in self._detectors:
            try:
                res = detector.analyze(normalized)
            except Exception as exc:  # Explicit error isolation per detector
                res = DetectorResult(
                    detector_name=detector.name,
                    detector_version=detector.version,
                    available=False,
                    score=None,
                    findings=[f"Detector error: {str(exc)}"],
                    metadata={"error": str(exc)},
                )
            detector_results.append(res)

        # 3. Risk Assessment Fusion
        risk = fuse_risk(detector_results)

        # 4. Decision Policy Evaluation
        decision = evaluate_decision(risk, detector_results)

        # 5. Workflow Output Response
        return AnalysisWorkflowResult(
            input_data=input_data,
            normalized=normalized,
            detector_results=detector_results,
            risk_assessment=risk,
            decision=decision,
        )
