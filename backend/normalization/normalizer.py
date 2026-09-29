"""Deterministic text normalization for security analysis only.

The caller's source text is retained separately and is never rewritten. The
canonical representation is intended for detectors; it is not a replacement
for the original text when displaying or forwarding user content.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from backend.models.contracts import AnalysisInput, NormalizedContent

NORMALIZATION_VERSION = "1.1"

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
_LINE_BREAKS = frozenset({"\n", "\x85", "\u2028", "\u2029"})
_HORIZONTAL_SPACE_RUN = re.compile(r" {2,}")


def normalize_input(input_data: AnalysisInput | str) -> NormalizedContent:
    """Return a canonical analysis form while preserving available source text.

    Accepting a raw string allows callers that have not yet constructed an
    ``AnalysisInput`` to preserve all leading and trailing source characters.
    The existing application service can continue to pass ``AnalysisInput``.
    """
    original_text = input_data if isinstance(input_data, str) else input_data.content
    signals: Counter[str] = Counter()

    # Capture source-level signals before compatibility normalization can
    # convert unusual spaces or other code points into ordinary characters.
    unusual_whitespace_count = sum(
        1
        for character in original_text
        if character.isspace() and character not in " \t\r\n"
    )
    if unusual_whitespace_count:
        signals["unusual_whitespace_detected"] = unusual_whitespace_count

    compatibility_text = unicodedata.normalize("NFKC", original_text)
    if compatibility_text != original_text:
        signals["unicode_nfkc_changed_text"] = 1

    canonical_characters: list[str] = []
    index = 0
    while index < len(compatibility_text):
        character = compatibility_text[index]

        # Normalize all common Unicode line separators to LF. CRLF is one
        # structural break, and repeated breaks are retained as paragraphs.
        if character == "\r":
            if index + 1 < len(compatibility_text) and compatibility_text[index + 1] == "\n":
                index += 1
            canonical_characters.append("\n")
        elif character in _LINE_BREAKS:
            canonical_characters.append("\n")
        elif character in _ZERO_WIDTH_CHARACTERS:
            signals["zero_width_character_detected"] += 1
        elif unicodedata.category(character) == "Cf":
            # Includes directional overrides and other invisible formatting
            # controls that can obscure the analyzed character sequence.
            signals["invisible_format_character_detected"] += 1
        elif character == "\t":
            canonical_characters.append(" ")
        elif character.isspace():
            canonical_characters.append(" ")
        elif unicodedata.category(character) == "Cc":
            signals["control_character_detected"] += 1
        else:
            canonical_characters.append(character)
        index += 1

    canonical_text = "".join(canonical_characters)
    canonical_text = _HORIZONTAL_SPACE_RUN.sub(" ", canonical_text)
    # Ignore boundary whitespace in the analysis form while preserving it in
    # original_text. Internal line breaks and blank lines remain meaningful.
    canonical_text = canonical_text.strip(" \n")

    return NormalizedContent(
        original_text=original_text,
        canonical_text=canonical_text,
        normalization_version=NORMALIZATION_VERSION,
        char_count=len(canonical_text),
        word_count=len(canonical_text.split()),
        normalization_signals=dict(sorted(signals.items())),
        source_offsets={"start": 0, "end": len(original_text)},
    )
