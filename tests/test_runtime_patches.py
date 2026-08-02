from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from audio_restoration_colab.runtime_patches import (
    canonicalize_separator_outputs,
    output_role_from_name,
)


class RuntimePatchTests(unittest.TestCase):
    def test_actual_parenthesized_stem_wins_over_preset_name(self) -> None:
        self.assertEqual(
            output_role_from_name(
                Path("song_(Vocals)_preset_instrumental_full.wav")
            ),
            "vocals",
        )
        self.assertEqual(
            output_role_from_name(
                Path("song_(Instrumental)_preset_vocal_clean.wav")
            ),
            "instrumental",
        )

    def test_simple_role_suffix_is_recognized(self) -> None:
        self.assertEqual(output_role_from_name(Path("input_vocals.wav")), "vocals")
        self.assertEqual(output_role_from_name(Path("track_no_reverb.wav")), "dry")

    def test_outputs_are_copied_to_short_unambiguous_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocals = root / "song_(Vocals)_preset_instrumental_full.wav"
            instrumental = (
                root / "song_(Instrumental)_preset_instrumental_full.wav"
            )
            vocals.write_bytes(b"actual-vocals")
            instrumental.write_bytes(b"actual-instrumental")

            mapped = canonicalize_separator_outputs(
                [str(vocals), str(instrumental)],
                root,
            )

            mapped_paths = [Path(item) for item in mapped]
            self.assertEqual(mapped_paths[0].name, "mapped-00-vocals.wav")
            self.assertEqual(
                mapped_paths[1].name,
                "mapped-01-instrumental.wav",
            )
            self.assertEqual(mapped_paths[0].read_bytes(), b"actual-vocals")
            self.assertEqual(
                mapped_paths[1].read_bytes(),
                b"actual-instrumental",
            )

    def test_unknown_output_is_left_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unknown = root / "model-result.wav"
            unknown.write_bytes(b"audio")

            mapped = canonicalize_separator_outputs([str(unknown)], root)

            self.assertEqual(mapped, [str(unknown.resolve())])


if __name__ == "__main__":
    unittest.main()
