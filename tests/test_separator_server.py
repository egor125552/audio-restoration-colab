from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

SERVER_PATH = Path(__file__).resolve().parents[1] / "workers" / "separator_server.py"


def load_server():
    spec = importlib.util.spec_from_file_location("separator_server", SERVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Не удалось загрузить separator_server.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SeparatorServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = load_server()

    def test_role_mapping_uses_actual_stem_not_preset_name(self) -> None:
        vocal = Path("song_(Vocals)_preset_instrumental_full.wav")
        instrumental = Path("song_(Instrumental)_preset_vocal_clean.wav")

        self.assertTrue(
            self.server._matches_role(
                path=vocal,
                role="vocals",
                expected_roles=("instrumental", "vocals"),
            )
        )
        self.assertFalse(
            self.server._matches_role(
                path=vocal,
                role="instrumental",
                expected_roles=("instrumental", "vocals"),
            )
        )
        self.assertTrue(
            self.server._matches_role(
                path=instrumental,
                role="instrumental",
                expected_roles=("vocals", "instrumental"),
            )
        )

    def test_role_mapping_handles_dereverb(self) -> None:
        self.assertTrue(
            self.server._matches_role(
                path=Path("song_(No Reverb)_model.wav"),
                role="dry",
                expected_roles=("dry", "reverb"),
            )
        )
        self.assertFalse(
            self.server._matches_role(
                path=Path("song_(No Reverb)_model.wav"),
                role="reverb",
                expected_roles=("dry", "reverb"),
            )
        )

    def test_canonicalizer_does_not_swap_full_instrumental_preset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocal = root / "song_(Vocals)_preset_instrumental_full.wav"
            instrumental = (
                root / "song_(Instrumental)_preset_instrumental_full.wav"
            )
            vocal.write_bytes(b"actual-vocals")
            instrumental.write_bytes(b"actual-instrumental")

            mapped = self.server._canonicalize_outputs(
                paths=[vocal, instrumental],
                output_dir=root,
                expected_roles=("instrumental", "vocals"),
            )

            self.assertEqual(mapped["vocals"].read_bytes(), b"actual-vocals")
            self.assertEqual(
                mapped["instrumental"].read_bytes(),
                b"actual-instrumental",
            )

    def test_two_stem_other_maps_to_known_counterpart(self) -> None:
        cases = (
            (("dry", "reverb"), "dry", "reverb"),
            (("clean", "bleed"), "bleed", "clean"),
            (("clean", "breaths"), "aspiration", "clean"),
            (("vocals", "instrumental"), "vocals", "instrumental"),
        )
        for expected_roles, known_label, remaining_role in cases:
            with (
                self.subTest(expected_roles=expected_roles),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                known = root / f"song_({known_label})_model.wav"
                residual = root / "song_(other)_model.wav"
                known.write_bytes(b"known")
                residual.write_bytes(b"counterpart")

                mapped = self.server._canonicalize_outputs(
                    paths=[residual, known],
                    output_dir=root,
                    expected_roles=expected_roles,
                )

                self.assertEqual(
                    mapped[remaining_role].read_bytes(),
                    b"counterpart",
                )

    def test_multiple_unknown_outputs_fail_instead_of_guessing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "unknown-a.wav"
            second = root / "unknown-b.wav"
            first.write_bytes(b"a")
            second.write_bytes(b"b")

            with self.assertRaisesRegex(ValueError, "Не удалось определить"):
                self.server._canonicalize_outputs(
                    paths=[first, second],
                    output_dir=root,
                    expected_roles=("vocals", "instrumental"),
                )


if __name__ == "__main__":
    unittest.main()
