from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audio_restoration_colab.runtime import (
    ModelResult,
    RuntimeLayout,
    build_worker_command,
    read_worker_manifest,
)


class RuntimeTests(unittest.TestCase):
    def test_worker_command_uses_arguments_instead_of_a_shell_string(self) -> None:
        layout = RuntimeLayout(
            project_root=Path("/project"),
            cache_root=Path("/cache"),
        )

        command = build_worker_command(
            layout=layout,
            model_id="denoise_normal",
            source=Path("/input/song.wav"),
            output_dir=Path("/output/job"),
            settings={"quality": "balanced"},
        )

        self.assertIsInstance(command, list)
        self.assertEqual(command[0], "/cache/envs/separator/bin/python")
        self.assertIn("/project/workers/separator_worker.py", command)
        self.assertEqual(json.loads(command[-1]), {"quality": "balanced"})

    def test_manifest_rejects_files_outside_the_job_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job_dir = Path(directory)
            manifest = job_dir / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "outputs": [
                            {
                                "role": "bad",
                                "path": str(job_dir / ".." / "secret.wav"),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "папки задания"):
                read_worker_manifest(job_dir)

    def test_manifest_returns_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job_dir = Path(directory)
            result = job_dir / "restored.wav"
            result.write_bytes(b"audio")
            (job_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "outputs": [
                            {"role": "restored", "path": str(result)}
                        ]
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                read_worker_manifest(job_dir),
                [ModelResult(role="restored", path=result.resolve())],
            )


if __name__ == "__main__":
    unittest.main()
