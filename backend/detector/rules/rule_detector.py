"""Rule-based prompt injection detector implementation."""

from __future__ import annotations

import re
from typing import List, Tuple
from backend.detector.base import BaseDetector
from backend.models.contracts import DetectorResult, NormalizedContent

DEFAULT_RULES: List[Tuple[str, str, float, str]] = [
    (
        "RULE-001",
        "Direct Override",
        90.0,
        r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules|context)",
    ),
    (
        "RULE-002",
        "Instruction Suppression",
        85.0,
        r"(?i)\bdisregard\s+(all\s+)?(previous|prior|system)\s+(directives|instructions|rules)",
    ),
    (
        "RULE-003",
        "Persona / Jailbreak Adoption",
        95.0,
        r"(?i)\byou\s+are\s+now\s+(in\s+)?(DAN|developer\ mode|unrestricted\ mode)\b",
    ),
    (
        "RULE-004",
        "System Prompt Exfiltration",
        80.0,
        r"(?i)\b(print|repeat|output|reveal)\s+(your\s+)?(system\s+prompt|initial\s+instructions)",
    ),
    (
        "RULE-005",
        "Fake System Tag",
        85.0,
        r"(?i)(</?system>|<\|im_start\|>system|\[SYSTEM_INSTRUCTION\])",
    ),
]


class RuleDetector(BaseDetector):
    """Deterministic rule-based detector for configured terms and patterns."""

    def __init__(self, rules: List[Tuple[str, str, float, str]] | None = None) -> None:
        self._rules = rules if rules is not None else DEFAULT_RULES

    @property
    def name(self) -> str:
        return "rule_detector"

    @property
    def version(self) -> str:
        return "1.0.0"

    def analyze(self, content: NormalizedContent) -> DetectorResult:
        text = content.canonical_text
        findings: List[str] = []
        highest_severity = 0.0
        match_count = 0

        for rule_id, category, severity, pattern in self._rules:
            if re.search(pattern, text):
                match_count += 1
                highest_severity = max(highest_severity, severity)
                findings.append(f"[{rule_id}] {category} matched")

        if match_count == 0:
            score = 0.0
        elif match_count == 1:
            score = highest_severity * 0.85
        else:
            score = min(100.0, highest_severity + (match_count - 1) * 10.0)

        return DetectorResult(
            detector_name=self.name,
            detector_version=self.version,
            available=True,
            score=round(score, 1),
            findings=findings,
            metadata={
                "rules_checked": len(self._rules),
                "matches_found": match_count,
            },
        )
