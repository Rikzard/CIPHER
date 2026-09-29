"""Core domain contracts for CIPHER."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnalysisInput(BaseModel):
    """Raw input submitted for prompt injection analysis."""

    model_config = ConfigDict(frozen=True)

    content: str = Field(..., description="Original untrusted text content to analyze")
    source_kind: str = Field(default="user", description="Source type: user, retrieval, tool, etc.")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Optional request/source metadata")

    @field_validator("content", "source_kind")
    @classmethod
    def validate_non_empty_str(cls, value: str, info: Any) -> str:
        if not value or not value.strip():
            raise ValueError(f"{info.field_name} must not be empty or whitespace-only")
        return value.strip()


class NormalizedContent(BaseModel):
    """Canonical form of input text for detector evaluation."""

    model_config = ConfigDict(frozen=True)

    original_text: str = Field(..., description="Preserved raw input text")
    canonical_text: str = Field(..., description="Normalized text representation")
    normalization_version: str = Field(default="1.0", description="Normalization algorithm version")
    char_count: int = Field(..., ge=0, description="Character count of canonical text")
    word_count: int = Field(..., ge=0, description="Word count of canonical text")
    normalization_signals: Dict[str, int] = Field(
        default_factory=dict,
        description="Counts of Unicode or control-character normalization signals",
    )
    source_offsets: Optional[Dict[str, int]] = Field(default=None, description="Optional offset mapping back to source")

    @field_validator("normalization_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("normalization_version must not be empty")
        return value.strip()


class DetectorResult(BaseModel):
    """Standardized output emitted by a detector implementation."""

    model_config = ConfigDict(frozen=True)

    detector_name: str = Field(..., description="Unique detector name")
    detector_version: str = Field(..., description="Version of detector implementation/rules")
    available: bool = Field(..., description="True if detector executed successfully; False if unavailable/failed")
    score: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Calibrated score 0-100 if evaluated")
    findings: List[str] = Field(default_factory=list, description="Bounded evidence or matched rule rationale")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional non-sensitive detector metadata")

    @field_validator("detector_name", "detector_version")
    @classmethod
    def validate_non_empty(cls, value: str, info: Any) -> str:
        if not value or not value.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        return value.strip()


class RiskAssessment(BaseModel):
    """Fused risk score and risk band generated from detector evidence."""

    model_config = ConfigDict(frozen=True)

    fused_score: float = Field(..., ge=0.0, le=100.0, description="Combined calibrated risk score")
    risk_band: str = Field(..., description="Risk category: LOW, MEDIUM, HIGH, CRITICAL")
    contributing_detector_versions: Dict[str, str] = Field(..., description="Map of detector names to versions used")
    calibration_version: str = Field(default="1.0", description="Version of score fusion calibration")

    @field_validator("risk_band")
    @classmethod
    def validate_risk_band(cls, value: str) -> str:
        allowed = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        upper_val = value.upper()
        if upper_val not in allowed:
            raise ValueError(f"risk_band must be one of {allowed}, got '{value}'")
        return upper_val


class Decision(BaseModel):
    """Policy decision outcome."""

    model_config = ConfigDict(frozen=True)

    action: str = Field(..., description="Policy action: allow, flag, or block")
    policy_version: str = Field(..., description="Version of decision policy rules applied")
    reason_codes: List[str] = Field(default_factory=list, description="Stable reason identifiers explaining the action")

    @field_validator("action")
    @classmethod
    def validate_action(cls, value: str) -> str:
        allowed = {"allow", "flag", "block"}
        lower_val = value.lower()
        if lower_val not in allowed:
            raise ValueError(f"action must be one of {allowed}, got '{value}'")
        return lower_val


class PromptEnvelope(BaseModel):
    """Structured representation preserving separation between trusted instructions and untrusted data."""

    model_config = ConfigDict(frozen=True)

    trusted_instructions: str = Field(..., description="System / prompt instructions from trusted caller")
    untrusted_data: str = Field(..., description="Untrusted payload analyzed by CIPHER")
    envelope_version: str = Field(default="1.0", description="Envelope format specification version")

    @field_validator("envelope_version")
    @classmethod
    def validate_envelope_version(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("envelope_version must not be empty")
        return value.strip()
