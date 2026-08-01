from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any


class BSInferEngine:
    """Persistent adapter around bs-roformer-infer's lifecycle session."""

    def __init__(
        self,
        *,
        model_slug: str,
        models_dir: Path,
        output_dir: Path,
    ) -> None:
        import torch
        from bs_roformer import BSRoformerSession

        self.model_slug = model_slug
        self.models_dir = models_dir.resolve()
        self.output_dir = str(output_dir.resolve())
        self.models_dir.mkdir(parents=True, exist_ok=True)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.session = BSRoformerSession(
            model_name=model_slug,
            models_dir=self.models_dir,
            device=device,
            backend="torch",
            progress=True,
        ).load()

    def separate(self, source: str | Path) -> list[str]:
        source_path = Path(source).resolve()
        output_dir = Path(self.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="bsinfer-input-",
            dir=output_dir,
        ) as temporary:
            input_dir = Path(temporary)
            isolated_source = input_dir / "input.wav"
            try:
                isolated_source.symlink_to(source_path)
            except OSError:
                shutil.copy2(source_path, isolated_source)
            manifest = self.session.infer(
                input_dir,
                store_dir=output_dir,
                verbose=True,
            )
        paths = [
            str(Path(item.output_path).resolve())
            for item in manifest.outputs
            if Path(item.output_path).is_file()
        ]
        if not paths:
            raise ValueError("BS-RoFormer не вернул ни одной дорожки.")
        return paths

    def release(self) -> None:
        self.session.release()

    def cache_info(self) -> dict[str, Any]:
        return self.session.cache_info()


def parse_bsinfer_slug(model_filename: str) -> str:
    prefix = "bsinfer:"
    if not model_filename.startswith(prefix):
        raise ValueError("Ожидалась модель с префиксом bsinfer:.")
    slug = model_filename[len(prefix) :].strip()
    if not slug:
        raise ValueError("После bsinfer: не указан registry-slug модели.")
    return slug
