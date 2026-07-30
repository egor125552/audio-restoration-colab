from __future__ import annotations

import unittest
import os
import subprocess
import sys
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

    def test_selection_view_only_opens_relevant_panel(self) -> None:
        view = selection_view(
            "audiosr_large",
            default_browser_settings(),
        )

        self.assertEqual(
            view.visible_panels,
            {
                "denoise": False,
                "lavasr": False,
                "flashsr": False,
                "audiosr": True,
            },
        )
        self.assertEqual(view.values["audiosr_steps"], 50)

    def test_normal_and_aggressive_denoise_settings_stay_separate(self) -> None:
        state = default_browser_settings()
        state = merge_active_settings(
            state,
            "denoise_normal",
            {"quality": "maximum", "segment": 320},
        )
        state = merge_active_settings(
            state,
            "denoise_aggressive",
            {"quality": "fast", "segment": 128},
        )

        self.assertEqual(
            state["denoise_normal"],
            {"quality": "maximum", "segment": 320},
        )
        self.assertEqual(
            state["denoise_aggressive"],
            {"quality": "fast", "segment": 128},
        )

    def test_information_includes_size_purpose_and_warning(self) -> None:
        information = build_model_information("lavasr_small")

        self.assertIn("50 МБ", information)
        self.assertIn("Речь", information)
        self.assertIn("музык", information.lower())

    def test_corrupt_browser_state_falls_back_to_defaults(self) -> None:
        view = selection_view(
            "audiosr_large",
            {"audiosr_large": {"steps": "bad", "guidance": 999}},
        )

        self.assertEqual(view.values["audiosr_steps"], 50)
        self.assertEqual(view.values["audiosr_guidance"], 10.0)

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
