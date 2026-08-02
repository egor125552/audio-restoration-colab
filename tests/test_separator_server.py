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
                path=Path("mapped-00-vocals.wav"),
                role="vocals",
                expected_roles=("vocals", "instrumental"),
            )
        )
        self.assertFalse(
            self.server._matches_role(
                path=Path("mapped-00-vocals.wav"),
                role="instrumental",
                expected_roles=("vocals", "instrumental"),
            )
        )

    def test_canonicalizer_renames_outputs_to_stable_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocal = root / "mapped-00-vocals.wav"
            inst = root / "mapped-01-instrumental.wav"
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


if __name__ == "__main__":
    unittest.main()
