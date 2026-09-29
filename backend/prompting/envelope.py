"""Structured instruction and data envelope builder."""

from __future__ import annotations

from backend.models.contracts import PromptEnvelope


def build_prompt_envelope(trusted_instructions: str, untrusted_data: str) -> PromptEnvelope:
    """Represent trusted instructions and untrusted data in typed distinct fields."""
    return PromptEnvelope(
        trusted_instructions=trusted_instructions,
        untrusted_data=untrusted_data,
        envelope_version="1.0.0",
    )
