from __future__ import annotations

import unittest

from audio_restoration_colab.result_ui import build_result_layout


class ResultUiTests(unittest.TestCase):
    def test_two_stem_layout_only_shows_relevant_controls(self) -> None:
        layout = build_result_layout(["instrumental", "vocals"])

        self.assertEqual(
            layout.choices,
            (("минусовка", "instrumental"), ("вокал", "vocals")),
        )
        self.assertTrue(layout.editor_visible)
        self.assertTrue(layout.gain_visibility["vocals"])
        self.assertTrue(layout.gain_visibility["other"])
        self.assertFalse(layout.gain_visibility["drums"])
        self.assertFalse(layout.gain_visibility["bass"])
        self.assertEqual(layout.other_gain_label, "Минусовка, %")
        self.assertTrue(layout.show_no_vocals_preset)
        self.assertTrue(layout.show_only_vocals_preset)
        self.assertFalse(layout.show_no_drums_preset)

    def test_six_stem_layout_exposes_all_present_instruments(self) -> None:
        layout = build_result_layout(
            ["vocals", "drums", "bass", "guitar", "piano", "other"]
        )

        self.assertEqual(len(layout.choices), 6)
        self.assertTrue(all(layout.gain_visibility.values()))
        self.assertEqual(layout.other_gain_label, "Остальное, %")
        self.assertTrue(layout.show_no_drums_preset)

    def test_single_result_does_not_show_stem_editor(self) -> None:
        layout = build_result_layout(["restored"])

        self.assertFalse(layout.editor_visible)
        self.assertFalse(layout.show_all_preset)
        self.assertFalse(layout.show_no_vocals_preset)
        self.assertFalse(layout.show_no_drums_preset)


if __name__ == "__main__":
    unittest.main()
