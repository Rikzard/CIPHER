"""Bounded, deterministic text normalization for security analysis only."""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Iterator

from backend.models.contracts import AnalysisInput, NormalizedContent, TrustClassification

NORMALIZATION_VERSION = f"1.2+UCD-{unicodedata.unidata_version}"
MAX_INPUT_CHARACTERS = 250_000
MAX_CANONICAL_CHARACTERS = 1_000_000
MAX_NORMALIZATION_EVENTS = 512
MAX_EVENT_CODEPOINTS = 32

_ZERO_WIDTH_CHARACTERS = frozenset(
    {
        "\u034f",  # Combining grapheme joiner
        "\u180e",  # Mongolian vowel separator (deprecated format character)
        "\u200b",  # Zero-width space
        "\u200c",  # Zero-width non-joiner
        "\u200d",  # Zero-width joiner
        "\u2060",  # Word joiner
        "\ufeff",  # Zero-width no-break space / BOM
    }
)
_REMOVED_ZERO_WIDTH_CHARACTERS = frozenset({"\u180e", "\u200b", "\u2060", "\ufeff"})
_PRESERVED_JOIN_CONTROLS = frozenset({"\u200c", "\u200d"})
_LINE_BREAKS = frozenset({"\n", "\x85", "\u2028", "\u2029"})
class NormalizationInputTooLarge(ValueError):
    """Raised when input exceeds the configured normalization size bound."""


def _hangul_jamo_type(character: str) -> str | None:
    codepoint = ord(character)
    if 0x1100 <= codepoint <= 0x115F or 0xA960 <= codepoint <= 0xA97C:
        return "L"
    if 0x1160 <= codepoint <= 0x11A7 or 0xD7B0 <= codepoint <= 0xD7C6:
        return "V"
    if 0x11A8 <= codepoint <= 0x11FF or 0xD7CB <= codepoint <= 0xD7FB:
        return "T"
    return None


def _hangul_continues(text: str, index: int, character: str) -> bool:
    current_type = _hangul_jamo_type(character)
    previous_type = _hangul_jamo_type(text[index - 1])
    if current_type == "V" and previous_type == "L":
        return True
    if current_type == "T":
        if previous_type == "V" and index > 1 and _hangul_jamo_type(text[index - 2]) == "L":
            return True
        previous_codepoint = ord(text[index - 1])
        if 0xAC00 <= previous_codepoint <= 0xD7A3 and (previous_codepoint - 0xAC00) % 28 == 0:
            return True
    return False


def _normalization_segments(text: str) -> Iterator[tuple[int, int]]:
    """Yield practical canonical-composition segments with source boundaries."""
    if not text:
        return

    segment_start = 0
    for index in range(1, len(text)):
        character = text[index]
        decomposition = unicodedata.normalize("NFKD", character)
        begins_with_mark = bool(decomposition) and unicodedata.combining(decomposition[0]) != 0
        if begins_with_mark or _hangul_continues(text, index, character):
            continue
        yield segment_start, index
        segment_start = index
    yield segment_start, len(text)


def _codepoints(text: str) -> str:
    visible = text[:MAX_EVENT_CODEPOINTS]
    result = " ".join(f"U+{ord(character):04X}" for character in visible)
    if len(text) > MAX_EVENT_CODEPOINTS:
        result += f" … (+{len(text) - MAX_EVENT_CODEPOINTS} codepoints)"
    return result


def _is_variation_selector(character: str) -> bool:
    return unicodedata.name(character, "").startswith("VARIATION SELECTOR")


def _normalize_with_source_spans(
    text: str,
) -> tuple[str, list[tuple[int, int]], list[tuple[int, int, str, str]]]:
    """NFKC-normalize text and map each output character to a source span.

    Each segment's output maps to the full source segment; this handles
    composition and compatibility expansions without guessing per-codepoint
    offsets. A conservative whole-input mapping is used if segment boundaries
    do not reproduce Python's full-string NFKC result.
    """
    normalized_text = unicodedata.normalize("NFKC", text)
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    changed_segments: list[tuple[int, int, str, str]] = []

    for start, end in _normalization_segments(text):
        source_segment = text[start:end]
        normalized_segment = unicodedata.normalize("NFKC", source_segment)
        parts.append(normalized_segment)
        spans.extend([(start, end)] * len(normalized_segment))
        if normalized_segment != source_segment:
            changed_segments.append((start, end, source_segment, normalized_segment))

    if "".join(parts) == normalized_text:
        return normalized_text, spans, changed_segments

    # Keep the authoritative full-string NFKC output. If an unusual Unicode
    # composition crosses our practical segment boundary, retain a broad but
    # correct source span for every output codepoint instead of false offsets.
    return normalized_text, [(0, len(text))] * len(normalized_text), [(0, len(text), text, normalized_text)]


def normalize_input(input_data: AnalysisInput | str) -> NormalizedContent:
    """Produce a canonical analysis form while retaining the exact source.

    ``AnalysisInput.content`` keeps its legacy trimmed value for existing
    callers; ``AnalysisInput.original_content`` carries the original supplied
    string for forensic preservation.
    """
    if isinstance(input_data, str):
        original_text = input_data
    else:
        original_text = input_data.original_content or input_data.content
        # model_copy(update=...) intentionally skips validation. Avoid pairing
        # a changed legacy content value with stale captured source text.
        if original_text.strip() != input_data.content:
            original_text = input_data.content

    if len(original_text) > MAX_INPUT_CHARACTERS:
        raise NormalizationInputTooLarge(
            f"input exceeds the {MAX_INPUT_CHARACTERS}-character normalization limit"
        )

    signals: Counter[str] = Counter()
    events: list[dict[str, str | int]] = []
    event_count = 0

    def record_event(
        signal: str,
        start: int,
        end: int,
        code_points: str,
        action: str,
    ) -> None:
        nonlocal event_count
        event_count += 1
        if len(events) < MAX_NORMALIZATION_EVENTS:
            events.append(
                {
                    "signal": signal,
                    "source_start": start,
                    "source_end": end,
                    "code_points": code_points,
                    "action": action,
                }
            )

    # Record the source-level whitespace before NFKC can turn it into ASCII.
    for offset, character in enumerate(original_text):
        if character.isspace() and character not in " \t\r\n":
            signals["unusual_whitespace_detected"] += 1
            record_event(
                "unusual_whitespace_detected",
                offset,
                offset + 1,
                f"U+{ord(character):04X}",
                "mapped_to_line_break" if character in _LINE_BREAKS else "mapped_to_space",
            )

        category = unicodedata.category(character)
        if character in _ZERO_WIDTH_CHARACTERS:
            signals["zero_width_character_detected"] += 1
            if character in _PRESERVED_JOIN_CONTROLS:
                signals["join_control_detected"] += 1
                action = "preserved"
            else:
                action = "removed" if character in _REMOVED_ZERO_WIDTH_CHARACTERS else "preserved"
            record_event(
                "zero_width_character_detected",
                offset,
                offset + 1,
                f"U+{ord(character):04X}",
                action,
            )
            if character in _PRESERVED_JOIN_CONTROLS:
                record_event(
                    "join_control_detected",
                    offset,
                    offset + 1,
                    f"U+{ord(character):04X}",
                    "preserved",
                )
        elif _is_variation_selector(character):
            signals["variation_selector_detected"] += 1
            record_event(
                "variation_selector_detected",
                offset,
                offset + 1,
                f"U+{ord(character):04X}",
                "preserved",
            )
        elif category == "Cf":
            signals["invisible_format_character_detected"] += 1
            record_event(
                "invisible_format_character_detected",
                offset,
                offset + 1,
                f"U+{ord(character):04X}",
                "removed",
            )

        if category == "Cc" and character not in "\t\r\n":
            signals["control_character_detected"] += 1
            record_event(
                "control_character_detected",
                offset,
                offset + 1,
                f"U+{ord(character):04X}",
                "mapped_to_space" if character.isspace() else "removed",
            )

    compatibility_text, compatibility_spans, nfkc_segments = _normalize_with_source_spans(original_text)
    if compatibility_text != original_text:
        signals["unicode_nfkc_changed_text"] = 1
    for start, end, source_segment, normalized_segment in nfkc_segments:
        record_event(
            "unicode_nfkc_changed_text",
            start,
            end,
            _codepoints(source_segment),
            f"normalized_to_{_codepoints(normalized_segment)}",
        )

    if len(compatibility_text) > MAX_CANONICAL_CHARACTERS:
        raise NormalizationInputTooLarge(
            f"normalized input exceeds the {MAX_CANONICAL_CHARACTERS}-character limit"
        )

    canonical_characters: list[str] = []
    canonical_spans: list[tuple[int, int]] = []
    index = 0
    while index < len(compatibility_text):
        character = compatibility_text[index]
        source_start, source_end = compatibility_spans[index]

        if character == "\r":
            if index + 1 < len(compatibility_text) and compatibility_text[index + 1] == "\n":
                source_end = compatibility_spans[index + 1][1]
                index += 1
            output_character = "\n"
        elif character in _LINE_BREAKS:
            output_character = "\n"
        elif character in _REMOVED_ZERO_WIDTH_CHARACTERS:
            index += 1
            continue
        elif character in _PRESERVED_JOIN_CONTROLS:
            output_character = character
        elif character in _ZERO_WIDTH_CHARACTERS:
            # Combining grapheme joiners are preserved to avoid changing
            # legitimate grapheme behavior; their signal remains available.
            output_character = character
        elif unicodedata.category(character) == "Cf":
            index += 1
            continue
        elif character == "\t" or character.isspace():
            output_character = " "
        elif unicodedata.category(character) == "Cc":
            index += 1
            continue
        else:
            output_character = character

        if output_character == " " and canonical_characters and canonical_characters[-1] == " ":
            previous_start, _ = canonical_spans[-1]
            canonical_spans[-1] = (previous_start, source_end)
        else:
            canonical_characters.append(output_character)
            canonical_spans.append((source_start, source_end))
        index += 1

    left = 0
    right = len(canonical_characters)
    while left < right and canonical_characters[left] in " \n":
        left += 1
    while right > left and canonical_characters[right - 1] in " \n":
        right -= 1
    canonical_characters = canonical_characters[left:right]
    canonical_spans = canonical_spans[left:right]

    if event_count > MAX_NORMALIZATION_EVENTS:
        signals["normalization_events_truncated"] = event_count - MAX_NORMALIZATION_EVENTS

    canonical_text = "".join(canonical_characters)
    return NormalizedContent(
        original_text=original_text,
        canonical_text=canonical_text,
        normalization_version=NORMALIZATION_VERSION,
        trust_classification=(
            input_data.trust_classification if isinstance(input_data, AnalysisInput)
            else TrustClassification.user_data()
        ),
        char_count=len(canonical_text),
        word_count=len(canonical_text.split()),
        normalization_signals=dict(sorted(signals.items())),
        normalization_events=events,
        canonical_source_spans=canonical_spans,
        source_offsets={"start": 0, "end": len(original_text)},
    )
