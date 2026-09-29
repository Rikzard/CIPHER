import unittest

from backend.models.contracts import AnalysisInput
from backend.normalization.normalizer import normalize_input


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
        input_data = AnalysisInput(content="Existing application input", source_kind="user")

        result = normalize_input(input_data)

        self.assertEqual(result.original_text, input_data.content)
        self.assertEqual(result.canonical_text, input_data.content)


if __name__ == "__main__":
    unittest.main()
