from __future__ import annotations

import unittest

from audio_restoration_colab.catalog import (
    MODEL_SPECS,
    default_browser_settings,
    get_model,
    normalize_settings,
)


class CatalogTests(unittest.TestCase):
    def test_first_version_contains_exactly_five_models(self) -> None:
        self.assertEqual(
            list(MODEL_SPECS),
            [
                "denoise_normal",
                "denoise_aggressive",
                "lavasr_small",
                "flashsr_medium",
                "audiosr_large",
            ],
        )

    def test_small_restoration_model_warns_that_it_is_for_speech(self) -> None:
        model = get_model("lavasr_small")

        self.assertIn("реч", model.warning.lower())
        self.assertIn("музык", model.warning.lower())

    def test_each_model_has_separate_default_settings(self) -> None:
        settings = default_browser_settings()

        self.assertEqual(set(settings), set(MODEL_SPECS))
        self.assertIsNot(
            settings["denoise_normal"],
            settings["denoise_aggressive"],
        )

    def test_unknown_or_out_of_range_settings_fall_back_safely(self) -> None:
        normalized = normalize_settings(
            "audiosr_large",
            {
                "mode": "unknown",
                "steps": 10_000,
                "guidance": -5,
                "seed": "not-a-number",
                "lowpass": "yes",
                "ignored": "value",
            },
        )

        self.assertEqual(normalized["mode"], "basic")
        self.assertEqual(normalized["steps"], 100)
        self.assertEqual(normalized["guidance"], 1.0)
        self.assertEqual(normalized["seed"], 42)
        self.assertIs(normalized["lowpass"], True)
        self.assertNotIn("ignored", normalized)

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Неизвестная модель"):
            get_model("made_up")


if __name__ == "__main__":
    unittest.main()
