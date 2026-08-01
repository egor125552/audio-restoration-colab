from __future__ import annotations

import unittest

from audio_restoration_colab.catalog import (
    MODEL_SPECS,
    default_browser_settings,
    get_model,
    normalize_settings,
)


class CatalogTests(unittest.TestCase):
    def test_existing_models_are_preserved(self) -> None:
        for model_id in (
            "denoise_normal",
            "denoise_aggressive",
            "lavasr_small",
            "flashsr_medium",
            "audiosr_large",
        ):
            self.assertIn(model_id, MODEL_SPECS)

    def test_stem_catalog_contains_specialized_top_tasks(self) -> None:
        required = {
            "stems_vocal_balanced",
            "stems_vocal_clean",
            "stems_instrumental_clean",
            "stems_six",
            "stems_four",
            "stems_guitar",
            "dereverb_big",
            "dereverb_super",
            "dereverb_echo",
            "stems_bleed_suppressor",
        }
        self.assertTrue(required.issubset(MODEL_SPECS))
        self.assertEqual(
            get_model("stems_six").output_roles,
            ("vocals", "drums", "bass", "guitar", "piano", "other"),
        )

    def test_universal_models_use_versioned_bs_registry(self) -> None:
        self.assertEqual(
            get_model("stems_six").model_filename,
            "bsinfer:roformer-model-bs-roformer-sw-by-jarredou",
        )
        self.assertEqual(
            get_model("stems_four").model_filename,
            "bsinfer:roformer-model-bs-roformer-musdb18hq-by-zfturbo",
        )
        self.assertEqual(
            get_model("stems_guitar").model_filename,
            "melband_roformer_guitar_becruily.ckpt",
        )
        self.assertNotIn("stems_cinematic", MODEL_SPECS)
        self.assertNotIn("stems_drums_detailed", MODEL_SPECS)

    def test_no_demucs_checkpoint_is_visible(self) -> None:
        for model in MODEL_SPECS.values():
            filename = (model.model_filename or "").lower()
            self.assertFalse(filename.startswith("demucs:"))
            self.assertNotIn("htdemucs", filename)
            self.assertFalse(filename.endswith(".th"))

    def test_stem_models_are_lazy_and_declarative(self) -> None:
        for model in MODEL_SPECS.values():
            if model.backend != "stems":
                continue
            self.assertTrue(model.model_filename or model.ensemble_preset)
            self.assertTrue(model.output_roles)
            self.assertTrue(model.source_text)

    def test_each_model_has_separate_default_settings(self) -> None:
        settings = default_browser_settings()
        self.assertEqual(set(settings), set(MODEL_SPECS))
        self.assertIsNot(
            settings["stems_six"],
            settings["stems_four"],
        )

    def test_stem_settings_are_clamped(self) -> None:
        normalized = normalize_settings(
            "stems_six",
            {
                "quality": "unknown",
                "segment": 9999,
                "overlap": -10,
                "chunk_minutes": 100,
                "keep_loaded": "yes",
            },
        )
        self.assertEqual(normalized["quality"], "balanced")
        self.assertEqual(normalized["segment"], 512)
        self.assertEqual(normalized["overlap"], 2)
        self.assertEqual(normalized["chunk_minutes"], 30)
        self.assertTrue(normalized["keep_loaded"])

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Неизвестная модель"):
            get_model("made_up")


if __name__ == "__main__":
    unittest.main()
