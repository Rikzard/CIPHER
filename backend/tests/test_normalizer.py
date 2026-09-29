import unittest
import unicodedata

from backend.models.contracts import AnalysisInput
from backend.normalization.normalizer import (
    MAX_CANONICAL_CHARACTERS,
    MAX_EVENT_CODEPOINTS,
    MAX_INPUT_CHARACTERS,
    MAX_NORMALIZATION_EVENTS,
    NORMALIZATION_VERSION,
    NormalizationInputTooLarge,
    normalize_input,
)


class TestInputNormalizer(unittest.TestCase):
    def test_ordinary_benign_text_is_unchanged_without_signals(self) -> None:
        text = "Explain quantum physics in plain English."

        result = normalize_input(text)

        self.assertEqual(result.original_text, text)
        self.assertEqual(result.canonical_text, text)
        self.assertEqual(result.normalization_signals, {})

    def test_empty_and_whitespace_only_text_normalizes_to_empty(self) -> None:
        for text in ("", " \t  \n\r ", "\u00a0\u2003"):
            with self.subTest(text=repr(text)):
                result = normalize_input(text)
                self.assertEqual(result.original_text, text)
                self.assertEqual(result.canonical_text, "")
                self.assertEqual(result.char_count, 0)
                self.assertEqual(result.word_count, 0)

    def test_repeated_horizontal_whitespace_is_collapsed(self) -> None:
        text = "  multiple   spaces\t\tbetween  words  "

        result = normalize_input(text)

        self.assertEqual(result.original_text, text)
        self.assertEqual(result.canonical_text, "multiple spaces between words")
        self.assertEqual(result.normalization_signals, {})

    def test_tabs_and_newlines_preserve_line_structure(self) -> None:
        text = "first\tsecond\r\nthird\rfourth\n\nfifth"

        result = normalize_input(text)

        self.assertEqual(result.original_text, text)
        self.assertEqual(result.canonical_text, "first second\nthird\nfourth\n\nfifth")
        self.assertEqual(result.normalization_signals, {})

    def test_unicode_nfkc_normalization_is_deterministic_and_signaled(self) -> None:
        text = "Cafe\u0301 ＡＢＣ"

        first = normalize_input(text)
        second = normalize_input(text)

        self.assertEqual(first.original_text, text)
        self.assertEqual(first.canonical_text, "Café ABC")
        self.assertEqual(first.model_dump(), second.model_dump())
        self.assertEqual(first.normalization_signals["unicode_nfkc_changed_text"], 1)

    def test_zero_width_characters_are_removed_from_analysis_and_signaled(self) -> None:
        text = "ig\u200bno\u200bre this"

        result = normalize_input(text)

        self.assertEqual(result.original_text, text)
        self.assertEqual(result.canonical_text, "ignore this")
        self.assertEqual(result.normalization_signals["zero_width_character_detected"], 2)

    def test_invisible_format_and_control_characters_are_signaled(self) -> None:
        text = "A\u202eB\u2066C\x00D"

        result = normalize_input(text)

        self.assertEqual(result.original_text, text)
        self.assertEqual(result.canonical_text, "ABCD")
        self.assertEqual(result.normalization_signals["invisible_format_character_detected"], 2)
        self.assertEqual(result.normalization_signals["control_character_detected"], 1)

    def test_unusual_unicode_whitespace_is_normalized_and_signaled(self) -> None:
        text = "alpha\u00a0beta\u2003gamma"

        result = normalize_input(text)

        self.assertEqual(result.original_text, text)
        self.assertEqual(result.canonical_text, "alpha beta gamma")
        self.assertEqual(result.normalization_signals["unusual_whitespace_detected"], 2)

    def test_mixed_unicode_and_ascii_text_is_preserved(self) -> None:
        text = "User: naïve 世界 — test"

        result = normalize_input(text)

        self.assertEqual(result.original_text, text)
        self.assertEqual(result.canonical_text, text)
        self.assertEqual(result.normalization_signals, {})

    def test_long_input_is_processed_without_truncation(self) -> None:
        text = ("word   " * 20_000).rstrip()

        result = normalize_input(text)

        self.assertEqual(result.original_text, text)
        self.assertEqual(result.canonical_text, " ".join(["word"] * 20_000))
        self.assertEqual(result.word_count, 20_000)

    def test_already_normalized_text_is_idempotent(self) -> None:
        text = "First line\n\nSecond line"

        first = normalize_input(text)
        second = normalize_input(first.canonical_text)

        self.assertEqual(first.canonical_text, second.canonical_text)
        self.assertEqual(first.normalization_signals, {})

    def test_original_text_and_source_boundary_are_preserved(self) -> None:
        text = "\u200b  leading\ttext  \n"

        result = normalize_input(text)

        self.assertEqual(result.original_text, text)
        self.assertEqual(result.source_offsets, {"start": 0, "end": len(text)})

    def test_analysis_input_remains_supported_for_application_compatibility(self) -> None:
        raw_text = "  Existing application input  "
        input_data = AnalysisInput.model_validate({"content": raw_text, "source_kind": "user"})

        result = normalize_input(input_data)

        self.assertEqual(input_data.content, "Existing application input")
        self.assertEqual(result.original_text, raw_text)
        self.assertEqual(result.canonical_text, "Existing application input")
        self.assertEqual(result.source_offsets, {"start": 0, "end": len(raw_text)})

        updated_input = input_data.model_copy(update={"content": "Updated content"})
        self.assertEqual(normalize_input(updated_input).original_text, "Updated content")

    def test_original_content_is_not_exposed_in_the_legacy_contract_schema(self) -> None:
        input_data = AnalysisInput(content="  exact source  ")

        self.assertEqual(input_data.content, "exact source")
        self.assertEqual(input_data.original_content, "  exact source  ")
        self.assertNotIn("original_content", input_data.model_dump())
        self.assertNotIn("original_content", AnalysisInput.model_json_schema()["properties"])

    def test_canonical_source_spans_cover_unicode_composition_and_removed_characters(self) -> None:
        result = normalize_input("e\u0301 A\u200bB")

        self.assertEqual(result.canonical_text, "é AB")
        self.assertEqual(
            result.canonical_source_spans,
            [(0, 2), (2, 3), (3, 4), (5, 6)],
        )

    def test_forensic_events_include_source_positions_and_codepoints(self) -> None:
        result = normalize_input("x\u200by")

        self.assertEqual(
            result.normalization_events,
            [
                {
                    "signal": "zero_width_character_detected",
                    "source_start": 1,
                    "source_end": 2,
                    "code_points": "U+200B",
                    "action": "removed",
                }
            ],
        )

    def test_join_controls_and_variation_selectors_are_preserved_and_signaled(self) -> None:
        text = "می\u200cروم 👩\u200d💻 ✈\ufe0f"

        result = normalize_input(text)

        self.assertEqual(result.canonical_text, text)
        self.assertEqual(result.normalization_signals["join_control_detected"], 2)
        self.assertEqual(result.normalization_signals["variation_selector_detected"], 1)
        self.assertEqual(
            sum(event["signal"] == "join_control_detected" for event in result.normalization_events),
            2,
        )

    def test_normalization_version_records_the_unicode_database(self) -> None:
        self.assertEqual(NORMALIZATION_VERSION, f"1.2+UCD-{unicodedata.unidata_version}")

    def test_oversized_source_input_is_rejected_before_normalization(self) -> None:
        with self.assertRaises(NormalizationInputTooLarge):
            normalize_input("a" * (MAX_INPUT_CHARACTERS + 1))

    def test_oversized_nfkc_expansion_is_rejected(self) -> None:
        expansion = unicodedata.normalize("NFKC", "\ufdfa")
        self.assertGreater(len(expansion), 1)
        repeat_count = MAX_CANONICAL_CHARACTERS // len(expansion) + 1

        with self.assertRaises(NormalizationInputTooLarge):
            normalize_input("\ufdfa" * repeat_count)

    def test_forensic_event_list_is_bounded_but_signal_count_is_complete(self) -> None:
        text = "\u200b" * (MAX_NORMALIZATION_EVENTS + 7)

        result = normalize_input(text)

        self.assertEqual(result.normalization_signals["zero_width_character_detected"], len(text))
        self.assertEqual(len(result.normalization_events), MAX_NORMALIZATION_EVENTS)
        self.assertEqual(result.normalization_signals["normalization_events_truncated"], 7)

    def test_forensic_event_codepoint_details_are_bounded(self) -> None:
        text = "e" + "\u0301" * (MAX_EVENT_CODEPOINTS + 10)

        result = normalize_input(text)

        self.assertEqual(result.original_text, text)
        self.assertLessEqual(
            max(len(str(value)) for event in result.normalization_events for value in event.values()),
            300,
        )

    def test_long_input_keeps_a_source_span_for_every_canonical_character(self) -> None:
        text = ("word   " * 20_000).rstrip()

        result = normalize_input(text)

        self.assertEqual(result.char_count, len(result.canonical_source_spans))
        self.assertEqual(result.canonical_source_spans[0], (0, 1))
        self.assertEqual(result.canonical_source_spans[-1], (len(text) - 1, len(text)))


if __name__ == "__main__":
    unittest.main()
