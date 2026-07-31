from __future__ import annotations

import json
import os
import shlex
import subprocess
from collections import deque
from collections.abc import Callable
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

JOB_LOG_ENV = "AUDIO_RESTORATION_JOB_LOG"
PROJECT_ROOT_ENV = "AUDIO_RESTORATION_PROJECT_ROOT"
CommandRunner = Callable[[list[str], dict[str, str]], None]


class SubprocessWorker:
    def __init__(
        self,
        *,
        layout: RuntimeLayout,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.layout = layout
        self.command_runner = command_runner or _run_command
        self._prepared_backends: set[str] = set()

    def run(
        self,
        *,
        model_id: str,
        source: Path,
        output_dir: Path,
        settings: dict[str, object],
        progress: Callable[[float, str], None],
    ) -> list[ModelResult]:
        model = get_model(model_id)
        project_root = _resolve_project_root(self.layout.project_root)
        resolved_layout = RuntimeLayout(
            project_root=project_root,
            cache_root=self.layout.cache_root,
        )
        log_path = output_dir.parent / "model.log"
        environment = {
            "AUDIO_RESTORATION_CACHE": str(self.layout.cache_root),
            PROJECT_ROOT_ENV: str(project_root),
            "PYTHONUNBUFFERED": "1",
            JOB_LOG_ENV: str(log_path),
        }
        print(
            f"[audio-restoration] {model.short_title}: лог запуска — {log_path}",
            flush=True,
        )
        print(
            f"[audio-restoration] Корень проекта: {project_root}",
            flush=True,
        )
        if model.backend not in self._prepared_backends:
            progress(
                0.10,
                "Проверяю среду модели. При первом запуске начнётся скачивание…",
            )
            prepare_command = [
                str(project_root / "scripts" / "prepare_backend.sh"),
                model.backend,
                str(self.layout.cache_root),
            ]
            self.command_runner(prepare_command, environment)
            self._prepared_backends.add(model.backend)

        progress(0.28, f"Запускаю: {model.short_title}…")
        command = build_worker_command(
            layout=resolved_layout,
            model_id=model_id,
            source=source,
            output_dir=output_dir,
            settings=settings,
        )
        self.command_runner(command, environment)
        return read_worker_manifest(output_dir)


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


def _resolve_project_root(configured_root: Path) -> Path:
    candidates: list[Path] = []
    override = os.environ.get(PROJECT_ROOT_ENV)
    if override:
        candidates.append(Path(override))
    candidates.extend([Path.cwd(), configured_root])

    checked: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in checked:
            continue
        checked.add(resolved)
        if (
            (resolved / "scripts" / "prepare_backend.sh").is_file()
            and (resolved / "workers").is_dir()
        ):
            return resolved

    raise ValueError(
        "Не найден корень проекта с scripts/prepare_backend.sh и workers. "
        "Перезапусти ячейки Colab сверху вниз."
    )


def _run_command(command: list[str], environment: dict[str, str]) -> None:
    log_value = environment.get(JOB_LOG_ENV)
    log_path = Path(log_value) if log_value else None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    header = f"\n$ {shlex.join(command)}\n"
    print(header, end="", flush=True)
    tail: deque[str] = deque(maxlen=80)

    log_file = (
        log_path.open("a", encoding="utf-8")
        if log_path is not None
        else None
    )
    try:
        if log_file is not None:
            log_file.write(header)
            log_file.flush()
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, **environment},
            )
        except FileNotFoundError as error:
            raise ValueError(
                "Не найден файл, необходимый для запуска выбранной модели."
            ) from error

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            if log_file is not None:
                log_file.write(line)
                log_file.flush()
            stripped = line.strip()
            if stripped:
                tail.append(stripped)

        return_code = process.wait()
        footer = f"\n[код завершения: {return_code}]\n"
        print(footer, end="", flush=True)
        if log_file is not None:
            log_file.write(footer)
            log_file.flush()
        if return_code != 0:
            if return_code < 0:
                detail = (
                    f"процесс остановлен сигналом {-return_code} "
                    f"(код {return_code})"
                )
            else:
                detail = tail[-1] if tail else f"код завершения {return_code}"
            raise ValueError(f"Модель завершилась с ошибкой: {detail}")
    finally:
        if log_file is not None:
            log_file.close()
