from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from audio_restoration_colab.catalog import default_browser_settings
from audio_restoration_colab.ui_state import (
    DEFAULT_MODEL_ID,
    build_model_information,
    merge_active_settings,
    selection_view,
)


class UiStateTests(unittest.TestCase):
    def test_default_model_is_general_audio_flashsr(self) -> None:
        self.assertEqual(DEFAULT_MODEL_ID, "flashsr_medium")

    def test_stem_selection_opens_only_stem_panel(self) -> None:
        view = selection_view("stems_six", default_browser_settings())
        self.assertEqual(
            view.visible_panels,
            {
                "denoise": False,
                "lavasr": False,
                "flashsr": False,
                "audiosr": False,
                "stems": True,
            },
        )
        self.assertTrue(view.values["stems_keep_loaded"])
        self.assertEqual(view.values["stems_segment"], 256)

    def test_settings_remain_separate_per_stem_task(self) -> None:
        state = default_browser_settings()
        state = merge_active_settings(
            state,
            "stems_six",
            {
                "quality": "maximum",
                "segment": 320,
                "overlap": 16,
                "chunk_minutes": 5,
                "keep_loaded": True,
            },
        )
        self.assertEqual(state["stems_six"]["segment"], 320)
        self.assertEqual(state["stems_four"]["segment"], 256)

    def test_information_mentions_cache_source_and_license(self) -> None:
        information = build_model_information("stems_six")
        self.assertIn("Кэш", information)
        self.assertIn("Источник", information)
        self.assertIn("Лицензия", information)

    def test_command_help_does_not_require_gradio_import(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        environment = {
            **os.environ,
            "PYTHONPATH": str(project_root / "src"),
        }
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "audio_restoration_colab.app",
                "--help",
            ],
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--share", completed.stdout)


if __name__ == "__main__":
    unittest.main()
