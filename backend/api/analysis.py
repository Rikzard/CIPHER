"""Prompt analysis HTTP adapter."""

from __future__ import annotations

import re
import time
from typing import Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from backend.application.orchestrator import ApplicationService
from backend.models.contracts import AnalysisInput, DetectorResult
from backend.normalization.normalizer import NormalizationInputTooLarge

router = APIRouter()
_RULE_CATEGORY = re.compile(r"^\[RULE-\d+\]\s+(.+?)\s+matched$")


class AnalyzeRequest(BaseModel):
    """Request body for direct user text analysis."""

    model_config = ConfigDict(extra="forbid")

    text: str


class AnalyzeResponse(BaseModel):
    """Structured result from the current rule-based MVP pipeline."""

    analysis_id: UUID
    verdict: Literal["allow", "flag", "block"]
    risk_score: float = Field(ge=0.0, le=100.0)
    risk_band: str
    detector_results: list[DetectorResult]
    findings: list[str]
    attack_categories: list[str]
    processing_latency_ms: float = Field(ge=0.0)


def get_application_service(request: Request) -> ApplicationService:
    """Return the app-scoped analysis orchestrator."""
    service = getattr(request.app.state, "analysis_service", None)
    if service is None:
        raise HTTPException(status_code=500, detail="Analysis service not initialized")
    return service


@router.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
def analyze(
    body: AnalyzeRequest,
    service: ApplicationService = Depends(get_application_service),
) -> AnalyzeResponse:
    """Run the existing normalization, detection, fusion, and policy pipeline."""
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    started_at = time.perf_counter()
    try:
        workflow = service.analyze_prompt(AnalysisInput(content=body.text, source_kind="user"))
    except NormalizationInputTooLarge:
        raise HTTPException(status_code=413, detail="text exceeds the analysis size limit") from None
    latency_ms = (time.perf_counter() - started_at) * 1000.0

    # Do not expose exception strings captured by the orchestrator if a
    # detector fails. Current successful rule results retain their metadata.
    safe_detector_results = [
        result
        if result.available
        else DetectorResult(
            detector_name=result.detector_name,
            detector_version=result.detector_version,
            available=False,
            score=None,
            findings=[],
            metadata={"status": "unavailable"},
        )
        for result in workflow.detector_results
    ]
    findings = [finding for result in safe_detector_results for finding in result.findings]
    categories = sorted(
        {
            match.group(1)
            for finding in findings
            if (match := _RULE_CATEGORY.match(finding)) is not None
        }
    )

    return AnalyzeResponse(
        analysis_id=uuid4(),
        verdict=workflow.decision.action,  # type: ignore[arg-type]
        risk_score=workflow.risk_assessment.fused_score,
        risk_band=workflow.risk_assessment.risk_band,
        detector_results=safe_detector_results,
        findings=findings,
        attack_categories=categories,
        processing_latency_ms=latency_ms,
    )
