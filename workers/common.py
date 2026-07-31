from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

PROGRESS_PREFIX = "@@AUDIO_RESTORATION_PROGRESS@@"


def parse_worker_args(
    description: str,
    allowed_model_ids: set[str],
) -> tuple[argparse.Namespace, dict[str, Any]]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-id", required=True, choices=sorted(allowed_model_ids))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--settings-json", required=True)
    arguments = parser.parse_args()
    try:
        settings = json.loads(arguments.settings_json)
    except json.JSONDecodeError as error:
        parser.error(f"Некорректные настройки: {error}")
    if not isinstance(settings, dict):
        parser.error("Настройки должны быть объектом JSON.")
    arguments.input = Path(arguments.input).resolve()
    arguments.output_dir = Path(arguments.output_dir).resolve()
    if not arguments.input.is_file():
        parser.error("Входной аудиофайл не найден.")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    return arguments, settings


def report_progress(
    fraction: float,
    message: str,
    *,
    tqdm_span: float | None = None,
) -> None:
    payload: dict[str, Any] = {
        "fraction": max(0.0, min(1.0, float(fraction))),
        "message": str(message),
    }
    if tqdm_span is not None:
        payload["tqdm_span"] = max(0.0, min(1.0, float(tqdm_span)))
    print(
        PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False),
        flush=True,
    )


def model_cache_root() -> Path:
    return Path(
        os.environ.get(
            "AUDIO_RESTORATION_CACHE",
            "/content/audio-restoration-models",
        )
    ).expanduser().resolve()


def write_manifest(
    output_dir: Path,
    outputs: list[tuple[str, Path]],
) -> Path:
    root = output_dir.resolve()
    payload: list[dict[str, str]] = []
    for role, path in outputs:
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(
                "Результат модели оказался вне рабочей папки."
            ) from error
        if not resolved.is_file():
            raise ValueError(f"Результат не найден: {resolved.name}")
        payload.append({"role": role, "path": str(resolved)})
    if not payload:
        raise ValueError("Модель не создала результатов.")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps({"outputs": payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
