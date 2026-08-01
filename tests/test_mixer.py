from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from audio_restoration_colab.mixer import (
    build_mix,
    build_mix_command,
    role_group,
)


class MixerTests(unittest.TestCase):
    def test_role_groups_cover_detailed_drums(self) -> None:
        self.assertEqual(role_group("kick"), "drums")
        self.assertEqual(role_group("vocals"), "vocals")
        self.assertEqual(role_group("unknown"), "other")

    def test_command_uses_group_gains_and_limiter(self) -> None:
        command = build_mix_command(
            resolved=[
                ("vocals", Path("/tmp/vocals.wav")),
                ("drums", Path("/tmp/drums.wav")),
            ],
            gains={"vocals": 0.5, "drums": 1.25},
            target=Path("/tmp/mix.wav"),
        )
        joined = " ".join(command)
        self.assertIn("volume=0.5000", joined)
        self.assertIn("volume=1.2500", joined)
        self.assertIn("amix=inputs=2", joined)
        self.assertIn("alimiter=limit=0.98", joined)

    def test_build_mix_validates_paths_and_accepts_fake_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            vocals = raw / "vocals.wav"
            drums = raw / "drums.wav"
            vocals.write_bytes(b"v")
            drums.write_bytes(b"d")

            def fake_ffmpeg(command: list[str]) -> None:
                Path(command[-1]).write_bytes(b"mix")

            result = build_mix(
                stem_paths={
                    "vocals": str(vocals),
                    "drums": str(drums),
                },
                selected_roles=["vocals", "drums"],
                gains={"vocals": 1.0, "drums": 1.0},
                ffmpeg_runner=fake_ffmpeg,
            )
            self.assertTrue(result.is_file())
            self.assertEqual(result.parent.name, "mixes")


if __name__ == "__main__":
    unittest.main()
