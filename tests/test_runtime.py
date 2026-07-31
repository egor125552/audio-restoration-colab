from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audio_restoration_colab.runtime import (
    JOB_LOG_ENV,
    ModelResult,
    RuntimeLayout,
    SubprocessWorker,
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

    def test_subprocess_worker_prepares_backend_then_reads_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            cache = root / "cache"
            output = root / "output"
            source = root / "input.wav"
            (project / "scripts").mkdir(parents=True)
            (project / "workers").mkdir()
            output.mkdir()
            source.write_bytes(b"input")
            calls: list[list[str]] = []
            environments: list[dict[str, str]] = []

            def fake_run(command: list[str], environment: dict[str, str]) -> None:
                calls.append(command)
                environments.append(environment)
                if command[0].endswith("python"):
                    result = output / "restored.wav"
                    result.write_bytes(b"audio")
                    (output / "manifest.json").write_text(
                        json.dumps(
                            {
                                "outputs": [
                                    {"role": "restored", "path": str(result)}
                                ]
                            }
                        ),
                        encoding="utf-8",
                    )

            messages: list[str] = []
            worker = SubprocessWorker(
                layout=RuntimeLayout(project_root=project, cache_root=cache),
                command_runner=fake_run,
            )

            results = worker.run(
                model_id="flashsr_medium",
                source=source,
                output_dir=output,
                settings={"lowpass": True},
                progress=lambda _fraction, message: messages.append(message),
            )

            self.assertEqual(len(calls), 2)
            self.assertEqual(
                calls[0],
                [
                    str(project / "scripts" / "prepare_backend.sh"),
                    "flashsr",
                    str(cache),
                ],
            )
            self.assertEqual(
                environments[0][JOB_LOG_ENV],
                str(root / "model.log"),
            )
            self.assertEqual(results[0].role, "restored")
            self.assertIn("Запускаю", messages[-1])


if __name__ == "__main__":
    unittest.main()
