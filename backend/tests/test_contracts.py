import unittest
from pydantic import ValidationError

from backend.models.contracts import (
    AnalysisInput,
    Decision,
    DetectorResult,
    NormalizedContent,
    PromptEnvelope,
    RiskAssessment,
)


class TestContracts(unittest.TestCase):
    def test_analysis_input_valid(self) -> None:
        inp = AnalysisInput(content="  Hello world  ", source_kind="user")
        self.assertEqual(inp.content, "Hello world")
        self.assertEqual(inp.source_kind, "user")

    def test_analysis_input_invalid_empty_content(self) -> None:
        with self.assertRaises(ValidationError):
            AnalysisInput(content="   ", source_kind="user")

    def test_normalized_content_valid(self) -> None:
        norm = NormalizedContent(
            original_text="Raw",
            canonical_text="raw",
            normalization_version="1.0",
            char_count=3,
            word_count=1,
        )
        self.assertEqual(norm.canonical_text, "raw")
        self.assertEqual(norm.char_count, 3)

    def test_detector_result_valid_and_invalid(self) -> None:
        res = DetectorResult(
            detector_name="rules",
            detector_version="1.0",
            available=True,
            score=45.5,
            findings=["Rule 1 matched"],
        )
        self.assertEqual(res.score, 45.5)

        with self.assertRaises(ValidationError):
            DetectorResult(
                detector_name="rules",
                detector_version="1.0",
                available=True,
                score=150.0,  # Out of range 0-100
            )

    def test_risk_assessment_valid_and_invalid(self) -> None:
        risk = RiskAssessment(
            fused_score=75.0,
            risk_band="HIGH",
            contributing_detector_versions={"rules": "1.0"},
        )
        self.assertEqual(risk.risk_band, "HIGH")

        with self.assertRaises(ValidationError):
            RiskAssessment(
                fused_score=75.0,
                risk_band="EXTREME",
                contributing_detector_versions={"rules": "1.0"},
            )

    def test_decision_valid_and_invalid(self) -> None:
        dec = Decision(action="ALLOW", policy_version="1.0", reason_codes=["CLEAN"])
        self.assertEqual(dec.action, "allow")

        with self.assertRaises(ValidationError):
            Decision(action="TERMINATE", policy_version="1.0", reason_codes=[])

    def test_prompt_envelope_valid(self) -> None:
        env = PromptEnvelope(trusted_instructions="You are helpful.", untrusted_data="User text")
        self.assertEqual(env.trusted_instructions, "You are helpful.")
        self.assertEqual(env.untrusted_data, "User text")


if __name__ == "__main__":
    unittest.main()
