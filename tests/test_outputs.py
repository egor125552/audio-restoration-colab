from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from audio_restoration_colab.outputs import (
    build_ffmpeg_command,
    create_result_zip,
    output_extension,
    probe_audio_bitrate,
    safe_stem,
)


class OutputTests(unittest.TestCase):
    def test_safe_stem_removes_path_traversal_and_control_characters(self) -> None:
        self.assertEqual(
            safe_stem("../../Песня\nфинал?.mp3"),
            "Песня финал",
        )

    def test_original_format_is_kept_only_when_supported(self) -> None:
        self.assertEqual(output_extension("source", Path("трек.flac")), ".flac")
        self.assertEqual(output_extension("source", Path("трек.xyz")), ".wav")
        self.assertEqual(output_extension("mp3", Path("трек.wav")), ".mp3")
        self.assertEqual(output_extension("wav", Path("трек.mp3")), ".wav")

    def test_mp3_command_requests_320_kilobits(self) -> None:
        command = build_ffmpeg_command(
            Path("/tmp/input.wav"),
            Path("/tmp/output.mp3"),
        )

        self.assertEqual(command[0], "ffmpeg")
        self.assertIn("libmp3lame", command)
        self.assertIn("320000", command)
        self.assertNotIn("shell=True", command)

    def test_mp3_command_can_preserve_source_bitrate(self) -> None:
        command = build_ffmpeg_command(
            Path("/tmp/input.wav"),
            Path("/tmp/output.mp3"),
            source_bitrate=192_000,
        )

        self.assertEqual(
            command[command.index("-b:a") + 1],
            "192000",
        )

    @patch("audio_restoration_colab.outputs.subprocess.run")
    def test_probe_audio_bitrate_prefers_audio_stream(self, run: Mock) -> None:
        run.return_value = Mock(
            returncode=0,
            stdout=(
                '{"streams": [{"bit_rate": "192000"}], '
                '"format": {"bit_rate": "201000"}}'
            ),
        )

        self.assertEqual(probe_audio_bitrate(Path("song.mp3")), 192_000)

    @patch("audio_restoration_colab.outputs.subprocess.run")
    def test_probe_audio_bitrate_falls_back_to_container(self, run: Mock) -> None:
        run.return_value = Mock(
            returncode=0,
            stdout=(
                '{"streams": [{}], "format": {"bit_rate": "128000"}}'
            ),
        )

        self.assertEqual(probe_audio_bitrate(Path("song.m4a")), 128_000)

    def test_zip_contains_only_result_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "clean.wav"
            second = root / "noise.wav"
            first.write_bytes(b"clean")
            second.write_bytes(b"noise")

            archive = create_result_zip([first, second], root / "results.zip")

            with zipfile.ZipFile(archive) as result_zip:
                self.assertEqual(
                    result_zip.namelist(),
                    ["clean.wav", "noise.wav"],
                )


if __name__ == "__main__":
    unittest.main()
