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

    def test_role_mapping_handles_demucs_and_dereverb(self) -> None:
        self.assertTrue(
            self.server._matches_role(
                path=Path("song_(Vocals)_model.wav"),
                role="vocals",
                expected_roles=("vocals", "instrumental"),
            )
        )
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
        self.assertTrue(
            self.server._matches_role(
                path=Path("input_vocals.wav"),
                role="vocals",
                expected_roles=("vocals", "drums", "bass", "other"),
            )
        )

    def test_model_or_preset_name_cannot_override_stem_label(self) -> None:
        vocal_path = Path(
            "padded-input_(Vocals)_preset_instrumental_full.wav"
        )
        instrumental_path = Path(
            "padded-input_(Instrumental)_preset_vocal_clean.wav"
        )

        self.assertTrue(
            self.server._matches_role(
                path=vocal_path,
                role="vocals",
                expected_roles=("instrumental", "vocals"),
            )
        )
        self.assertFalse(
            self.server._matches_role(
                path=vocal_path,
                role="instrumental",
                expected_roles=("instrumental", "vocals"),
            )
        )
        self.assertTrue(
            self.server._matches_role(
                path=instrumental_path,
                role="instrumental",
                expected_roles=("vocals", "instrumental"),
            )
        )
        self.assertFalse(
            self.server._matches_role(
                path=instrumental_path,
                role="vocals",
                expected_roles=("vocals", "instrumental"),
            )
        )

    def test_canonicalizer_renames_outputs_to_stable_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocal = root / "track_(Vocals).wav"
            inst = root / "track_(Instrumental).wav"
            vocal.write_bytes(b"v")
            inst.write_bytes(b"i")
            mapped = self.server._canonicalize_outputs(
                paths=[inst, vocal],
                output_dir=root,
                expected_roles=("vocals", "instrumental"),
            )
            self.assertEqual(set(mapped), {"vocals", "instrumental"})
            self.assertEqual((root / "vocals.wav").read_bytes(), b"v")
            self.assertEqual((root / "instrumental.wav").read_bytes(), b"i")

    def test_canonicalizer_does_not_swap_full_instrumental_preset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocal = root / "song_(Vocals)_preset_instrumental_full.wav"
            inst = root / "song_(Instrumental)_preset_instrumental_full.wav"
            vocal.write_bytes(b"actual-vocals")
            inst.write_bytes(b"actual-instrumental")

            mapped = self.server._canonicalize_outputs(
                paths=[vocal, inst],
                output_dir=root,
                expected_roles=("instrumental", "vocals"),
            )

            self.assertEqual(
                mapped["vocals"].read_bytes(),
                b"actual-vocals",
            )
            self.assertEqual(
                mapped["instrumental"].read_bytes(),
                b"actual-instrumental",
            )

    def test_multiple_unknown_outputs_fail_instead_of_guessing_order(self) -> None:
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
