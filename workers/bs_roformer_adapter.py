from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


class BSRoformerEngine:
    """Adapter around the registry-backed BS-RoFormer session API."""

    def __init__(self, *, model_slug: str, cache_dir: Path) -> None:
        import torch
        from bs_roformer import BSRoformerSession

        cache_dir.mkdir(parents=True, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.session = BSRoformerSession(
            model_name=model_slug,
            models_dir=cache_dir,
            device=device,
            backend="torch",
            progress=True,
        ).load()
        self.model_slug = model_slug

    def separate(self, source: Path, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="bs-roformer-input-",
            dir=output_dir,
        ) as temporary:
            input_dir = Path(temporary)
            staged = input_dir / "input.wav"
            shutil.copy2(source, staged)
            manifest = self.session.infer(
                input_dir,
                store_dir=output_dir,
                verbose=True,
            )
        paths = [Path(item.output_path).resolve() for item in manifest.outputs]
        paths = [path for path in paths if path.is_file()]
        if not paths:
            raise ValueError("BS-RoFormer не вернул ни одного WAV-файла.")
        return paths

    def release(self) -> None:
        self.session.release()
