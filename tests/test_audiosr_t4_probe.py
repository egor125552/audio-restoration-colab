from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = PROJECT_ROOT / "scripts" / "probe_audiosr_t4.py"


def _load_probe_module():
    spec = importlib.util.spec_from_file_location("audiosr_t4_probe", PROBE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Не удалось загрузить probe-модуль.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeNamedModuleRoot:
    def __init__(self) -> None:
        self.wrapper = SimpleNamespace(diffusion_model=object())

    def named_modules(self):
        yield "", self
        yield "wrapper", self.wrapper
        yield "wrapper.diffusion_model", self.wrapper.diffusion_model


class AudioSrT4ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.probe = _load_probe_module()

    def test_probe_forces_headless_matplotlib_backend(self) -> None:
        self.assertEqual(os.environ.get("MPLBACKEND"), "Agg")

    def test_find_and_replace_diffusion_module(self) -> None:
        root = _FakeNamedModuleRoot()
        name, original = self.probe._find_diffusion_module(root)

        self.assertEqual(name, "wrapper.diffusion_model")
        self.assertIs(original, root.wrapper.diffusion_model)

        replacement = object()
        self.probe._replace_module(root, name, replacement)
        self.assertIs(root.wrapper.diffusion_model, replacement)

    def test_find_diffusion_module_fails_clearly(self) -> None:
        class EmptyRoot:
            def named_modules(self):
                yield "", self

        with self.assertRaisesRegex(RuntimeError, "diffusion_model"):
            self.probe._find_diffusion_module(EmptyRoot())

    def test_tensorrt_options_avoid_tiny_fragments_and_enable_fp16(self) -> None:
        fake_torch = SimpleNamespace(float32="fp32", float16="fp16")

        options = self.probe._tensorrt_options(fake_torch)

        self.assertEqual(options["enabled_precisions"], {"fp32", "fp16"})
        self.assertEqual(options["min_block_size"], 5)
        self.assertEqual(options["optimization_level"], 3)
        self.assertTrue(options["truncate_double"])
        self.assertNotIn("truncate_long_and_double", options)
        self.assertFalse(options["use_python_runtime"])
        self.assertTrue(options["pass_through_build_failures"])
        self.assertTrue(options["disable_tf32"])

    def test_export_adapter_hides_optional_audiosr_kwargs(self) -> None:
        class FakeModule:
            pass

        class FakeDiffusion:
            def __init__(self) -> None:
                self.calls = []

            def __call__(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return "result"

        diffusion = FakeDiffusion()
        fake_torch = SimpleNamespace(nn=SimpleNamespace(Module=FakeModule))
        adapter = self.probe._make_export_diffusion_adapter(
            diffusion_module=diffusion,
            torch=fake_torch,
        )

        result = adapter.forward("audio", "step")

        self.assertEqual(result, "result")
        self.assertEqual(
            diffusion.calls,
            [
                (
                    ("audio", "step"),
                    {
                        "y": None,
                        "context_list": [],
                        "context_attn_mask_list": [],
                    },
                )
            ],
        )

    def test_probe_uses_aot_export_instead_of_lazy_torch_compile(self) -> None:
        source = PROBE_PATH.read_text(encoding="utf-8")

        self.assertIn("torch.export.export(", source)
        self.assertIn("torch_tensorrt.dynamo.compile(", source)
        self.assertNotIn("torch.compile(", source)
        self.assertIn("_make_export_diffusion_adapter", source)
        self.assertNotIn("kwargs=export_kwargs", source)
        self.assertIn("Each real diffusion call therefore has batch size 1", source)
        self.assertIn("torch.tensor([999]", source)
        self.assertNotIn("torch.tensor([999, 500]", source)
        self.assertIn("граф уже полностью скомпилирован", source)

    def test_snr_is_infinite_for_identical_audio(self) -> None:
        import numpy as np

        reference = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        snr = self.probe._snr_db(reference, reference.copy(), np)

        self.assertTrue(math.isinf(snr))
        self.assertGreater(snr, 0)

    def test_snr_is_finite_for_changed_audio(self) -> None:
        import numpy as np

        reference = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        candidate = np.array([0.1, -0.1, 0.25], dtype=np.float32)
        snr = self.probe._snr_db(reference, candidate, np)

        self.assertTrue(math.isfinite(snr))

    def test_cli_rejects_single_tensorrt_run_before_gpu_imports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.wav"
            source.write_bytes(b"not-a-real-wave-but-exists")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROBE_PATH),
                    "--input",
                    str(source),
                    "--runs",
                    "1",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("--runs должен быть не меньше 2", completed.stderr)
        self.assertNotIn("torch_tensorrt", completed.stderr)


if __name__ == "__main__":
    unittest.main()
