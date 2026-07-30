from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKERS = PROJECT_ROOT / "workers"


def load_common():
    path = WORKERS / "common.py"
    spec = importlib.util.spec_from_file_location("worker_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Не удалось загрузить workers/common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_worker(filename: str):
    path = WORKERS / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Не удалось загрузить {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(WORKERS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class WorkerContractTests(unittest.TestCase):
    def test_common_writes_manifest_with_resolved_job_paths(self) -> None:
        common = load_common()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = output / "restored.wav"
            result.write_bytes(b"audio")

            manifest = common.write_manifest(
                output,
                [("restored", result)],
            )

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["outputs"][0]["role"], "restored")
            self.assertEqual(
                payload["outputs"][0]["path"],
                str(result.resolve()),
            )

    def test_common_rejects_result_outside_output_directory(self) -> None:
        common = load_common()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            outside = root / "outside.wav"
            output.mkdir()
            outside.write_bytes(b"audio")

            with self.assertRaisesRegex(ValueError, "рабочей папки"):
                common.write_manifest(
                    output,
                    [("restored", outside)],
                )

    def test_each_worker_help_runs_without_model_dependencies(self) -> None:
        for filename in (
            "separator_worker.py",
            "lavasr_worker.py",
            "flashsr_worker.py",
            "audiosr_worker.py",
        ):
            with self.subTest(worker=filename):
                completed = subprocess.run(
                    [sys.executable, str(WORKERS / filename), "--help"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("--settings-json", completed.stdout)

    def test_backend_installer_has_valid_shell_syntax(self) -> None:
        completed = subprocess.run(
            [
                "bash",
                "-n",
                str(PROJECT_ROOT / "scripts" / "prepare_backend.sh"),
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_denoise_worker_recognizes_no_noise_as_clean_layer(self) -> None:
        worker = load_worker("separator_worker.py")
        clean = Path("song_(No Noise).wav")
        noise = Path("song_(Noise).wav")

        self.assertEqual(
            worker._identify_denoise_outputs([noise, clean]),
            (clean, noise),
        )


if __name__ == "__main__":
    unittest.main()
