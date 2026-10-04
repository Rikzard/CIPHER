"""Run extracted DOCX text through CIPHER and retain finding provenance."""

from __future__ import annotations

import re
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from backend.application.orchestrator import ApplicationService
from backend.document.docx_parser import (
    DocumentContentTooLarge,
    DocumentContentUnit,
    ExtractedDocument,
    extract_docx,
)
from backend.models.contracts import DetectorResult, RiskAssessment, TrustClassification
from backend.normalization.normalizer import NormalizationInputTooLarge
from backend.policy.decision import evaluate_decision
from backend.risk.fusion import fuse_risk

_RULE_CATEGORY = re.compile(r"^\[RULE-\d+\]\s+(.+?)\s+matched$")
_SEMANTIC_CATEGORY = re.compile(r"^\[SEM-\d+\]\s+Similarity to ([a-z0-9_]+) example\b")


class DocumentSourceLocation(BaseModel):
    """Zero-based location of extracted text in the source DOCX."""

    model_config = ConfigDict(frozen=True)

    trust_classification: TrustClassification
    document_id: UUID
    filename: str
    content_type: str
    paragraph_index: int | None = None
    table_index: int | None = None
    row_index: int | None = None
    column_index: int | None = None


class LocatedDetectorResult(BaseModel):
    """Detector result tied to the paragraph or cell it analyzed."""

    model_config = ConfigDict(frozen=True)

    source_location: DocumentSourceLocation
    result: DetectorResult


class DocumentFinding(BaseModel):
    """Finding text plus its exact extracted paragraph or table-cell location."""

    model_config = ConfigDict(frozen=True)

    detector_name: str
    finding: str
    source_location: DocumentSourceLocation


class DocumentAnalysisResponse(BaseModel):
    """Document-level outcome with each segment's detector evidence and provenance."""

    model_config = ConfigDict(frozen=True)

    document_id: UUID
    trust_classification: TrustClassification
    filename: str
    content_type: str
    verdict: Literal["allow", "flag", "block"]
    risk_score: float = Field(ge=0.0, le=100.0)
    risk_band: str
    detector_results: list[LocatedDetectorResult]
    findings: list[DocumentFinding]
    attack_categories: list[str]
    processing_latency_ms: float = Field(ge=0.0)


def _source_location(segment: DocumentContentUnit) -> DocumentSourceLocation:
    return DocumentSourceLocation(
        trust_classification=segment.trust_classification,
        document_id=segment.document_id,
        filename=segment.filename,
        content_type=segment.content_type,
        paragraph_index=segment.paragraph_index,
        table_index=segment.table_index,
        row_index=segment.row_index,
        column_index=segment.column_index,
    )


def _safe_result(result: DetectorResult) -> DetectorResult:
    """Strip implementation exception data from unavailable detector results."""
    if result.available:
        return result
    return DetectorResult(
        detector_name=result.detector_name,
        detector_version=result.detector_version,
        available=False,
        score=None,
        findings=[],
        metadata={"status": "unavailable"},
    )


def _category_for_finding(finding: str) -> str | None:
    rule_match = _RULE_CATEGORY.match(finding)
    if rule_match:
        return rule_match.group(1)
    semantic_match = _SEMANTIC_CATEGORY.match(finding)
    return semantic_match.group(1) if semantic_match else None


class DocumentAnalysisService:
    """DOCX-specific composition over the established CIPHER analysis service."""

    def __init__(self, application_service: ApplicationService) -> None:
        self._application_service = application_service

    def analyze_docx(
        self,
        data: bytes,
        *,
        filename: str,
        max_size_bytes: int,
        processing_latency_ms: float,
    ) -> DocumentAnalysisResponse:
        """Analyze each non-empty text segment, then reuse existing fusion and policy."""
        document: ExtractedDocument = extract_docx(
            data,
            filename=filename,
            max_size_bytes=max_size_bytes,
        )
        located_results: list[LocatedDetectorResult] = []
        raw_results: list[DetectorResult] = []
        unit_risks: list[RiskAssessment] = []
        findings: list[DocumentFinding] = []
        categories: set[str] = set()

        for segment in document.segments:
            if not segment.extracted_text.strip():
                continue
            try:
                workflow = self._application_service.analyze_applicant_document_content(
                    segment.extracted_text,
                    metadata={
                        "document_id": str(document.document_id),
                        "filename": document.filename,
                        "content_type": document.content_type,
                    },
                )
            except NormalizationInputTooLarge as exc:
                raise DocumentContentTooLarge("An extracted text segment exceeds the analysis limit") from exc

            location = _source_location(segment)
            unit_results: list[DetectorResult] = []
            for result in workflow.detector_results:
                safe = _safe_result(result)
                unit_results.append(safe)
                raw_results.append(safe)
                located_results.append(LocatedDetectorResult(source_location=location, result=safe))
                if safe.available:
                    for finding in safe.findings:
                        findings.append(
                            DocumentFinding(
                                detector_name=safe.detector_name,
                                finding=finding,
                                source_location=location,
                            )
                        )
                        category = _category_for_finding(finding)
                        if category:
                            categories.add(category)
            # Each paragraph/cell is the source-mapped analysis unit. Existing
            # fusion ignores unavailable scores and applies the current fusion
            # policy to the detector results for this unit.
            unit_risks.append(fuse_risk(unit_results))

        # Document aggregation is the maximum available unit risk, using the
        # existing fusion result rather than introducing a new scoring formula.
        risk = max(unit_risks, key=lambda assessment: assessment.fused_score)
        decision = evaluate_decision(risk, raw_results)
        return DocumentAnalysisResponse(
            document_id=document.document_id,
            trust_classification=document.trust_classification,
            filename=document.filename,
            content_type=document.content_type,
            verdict=decision.action,  # type: ignore[arg-type]
            risk_score=risk.fused_score,
            risk_band=risk.risk_band,
            detector_results=located_results,
            findings=findings,
            attack_categories=sorted(categories),
            processing_latency_ms=processing_latency_ms,
        )
