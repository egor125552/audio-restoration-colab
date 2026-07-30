from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import get_model


@dataclass(frozen=True)
class RuntimeLayout:
    project_root: Path
    cache_root: Path


@dataclass(frozen=True)
class ModelResult:
    role: str
    path: Path


BACKEND_LAYOUT = {
    "separator": ("separator", "separator_worker.py"),
    "lavasr": ("lavasr", "lavasr_worker.py"),
    "flashsr": ("flashsr", "flashsr_worker.py"),
    "audiosr": ("audiosr", "audiosr_worker.py"),
}


def build_worker_command(
    *,
    layout: RuntimeLayout,
    model_id: str,
    source: Path,
    output_dir: Path,
    settings: dict[str, Any],
) -> list[str]:
    model = get_model(model_id)
    environment_name, worker_name = BACKEND_LAYOUT[model.backend]
    python = layout.cache_root / "envs" / environment_name / "bin" / "python"
    worker = layout.project_root / "workers" / worker_name
    return [
        str(python),
        str(worker),
        "--model-id",
        model_id,
        "--input",
        str(source),
        "--output-dir",
        str(output_dir),
        "--settings-json",
        json.dumps(settings, ensure_ascii=False, sort_keys=True),
    ]


def read_worker_manifest(job_dir: Path) -> list[ModelResult]:
    job_root = job_dir.resolve()
    manifest_path = job_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Модель не создала список результатов.")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_outputs = payload["outputs"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Модель создала повреждённый список результатов.") from error
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ValueError("Модель не вернула ни одного результата.")

    results: list[ModelResult] = []
    for item in raw_outputs:
        if not isinstance(item, dict):
            raise ValueError("Модель вернула неправильное описание файла.")
        role = str(item.get("role", "")).strip()
        result_path = Path(str(item.get("path", ""))).resolve()
        try:
            result_path.relative_to(job_root)
        except ValueError as error:
            raise ValueError(
                "Результат модели оказался вне папки задания."
            ) from error
        if not role or not result_path.is_file():
            raise ValueError("Один из результатов модели отсутствует.")
        results.append(ModelResult(role=role, path=result_path))
    return results
