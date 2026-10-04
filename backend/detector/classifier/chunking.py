"""Tokenizer-driven overlapping windows for transformer inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from backend.models.contracts import TrustClassification


class FastTokenizer(Protocol):
    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class TokenChunk:
    """One model input window, canonical span, and inherited source trust."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    token_count: int
    chunk_index: int
    canonical_start: int
    canonical_end: int
    trust_classification: TrustClassification = field(default_factory=TrustClassification.user_data)


def tokenize_overlapping_chunks(
    text: str,
    tokenizer: FastTokenizer,
    *,
    max_length: int = 512,
    stride: int = 256,
    trust_classification: TrustClassification | None = None,
) -> tuple[list[TokenChunk], int]:
    """Tokenize full text into overlapping model windows without truncation loss.

    The tokenizer must be a fast tokenizer returning ``offset_mapping``. Its
    overflow windows follow Hugging Face semantics: ``stride`` is the number
    of tokens carried over from the preceding window.
    """
    if max_length < 3:
        raise ValueError("max_length must allow special tokens and content")
    if stride < 0 or stride >= max_length - 2:
        raise ValueError("stride must be non-negative and smaller than content window")
    if not text:
        return [], 0

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )
    raw_ids = encoded.get("input_ids")
    raw_masks = encoded.get("attention_mask")
    raw_offsets = encoded.get("offset_mapping")
    if raw_ids is None or raw_masks is None or raw_offsets is None:
        raise ValueError("fast tokenizer must return input_ids, attention_mask, and offset_mapping")
    ids = _as_windows(raw_ids)
    masks = _as_windows(raw_masks)
    offsets = _as_windows(raw_offsets)
    if not (len(ids) == len(masks) == len(offsets)):
        raise ValueError("tokenizer returned mismatched overflow window counts")

    chunks: list[TokenChunk] = []
    for index, (input_ids, attention_mask, char_offsets) in enumerate(zip(ids, masks, offsets)):
        if not (len(input_ids) == len(attention_mask) == len(char_offsets)):
            raise ValueError("tokenizer returned mismatched window fields")
        non_special = [(int(start), int(end)) for start, end in char_offsets if int(end) > int(start)]
        if not non_special:
            continue
        chunks.append(
            TokenChunk(
                input_ids=tuple(int(value) for value in input_ids),
                attention_mask=tuple(int(value) for value in attention_mask),
                token_count=sum(int(value) for value in attention_mask),
                chunk_index=index,
                canonical_start=min(start for start, _ in non_special),
                canonical_end=max(end for _, end in non_special),
                trust_classification=trust_classification or TrustClassification.user_data(),
            )
        )

    original_tokens = tokenizer(text, add_special_tokens=False).get("input_ids")
    if original_tokens is None:
        token_count = max((chunk.token_count for chunk in chunks), default=0)
    else:
        token_count = len(original_tokens[0]) if original_tokens and isinstance(original_tokens[0], list) else len(original_tokens)
    return chunks, token_count


def _as_windows(values: Any) -> list[list[Any]]:
    """Normalize single-window and overflow tokenizer output to nested lists."""
    values = values.tolist() if hasattr(values, "tolist") else values
    if not isinstance(values, (list, tuple)):
        raise ValueError("tokenizer output must be a list")
    if not values:
        return []
    first = values[0]
    if isinstance(first, (list, tuple)):
        return [list(window) for window in values]
    return [list(values)]
