from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import threading
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
PROGRESS_PREFIX = "@@AUDIO_RESTORATION_PROGRESS@@"
SERVER_READY_PREFIX = "@@AUDIO_RESTORATION_SERVER_READY@@"
SERVER_RESULT_PREFIX = "@@AUDIO_RESTORATION_SERVER_RESULT@@"
TQDM_PERCENT = re.compile(r"(?<!\d)(100|[1-9]?\d)%\|")
LineHandler = Callable[[str], None]
CommandRunner = Callable[[list[str], dict[str, str], LineHandler | None], None]


class _WorkerProgressParser:
    def __init__(self, progress: Callable[[float, str], None]) -> None:
        self.progress = progress
        self.local_base = 0.0
        self.tqdm_span = 0.0
        self.message = "Модель обрабатывает аудио…"
        self.last_overall = 0.28

    def feed(self, text: str) -> None:
        for fragment in text.replace("\r", "\n").splitlines():
            stripped = fragment.strip()
            if not stripped:
                continue
            if stripped.startswith(PROGRESS_PREFIX):
                self._structured(stripped[len(PROGRESS_PREFIX) :])
                continue
            match = TQDM_PERCENT.search(stripped)
            if match is not None and self.tqdm_span > 0:
                percent = int(match.group(1))
                local = self.local_base + self.tqdm_span * (percent / 100.0)
                self._emit(local, f"{self.message} {percent}%")

    def _structured(self, raw_payload: str) -> None:
        try:
            payload = json.loads(raw_payload)
            fraction = float(payload["fraction"])
            message = str(payload["message"])
            tqdm_span = float(payload.get("tqdm_span", 0.0))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        self.local_base = max(0.0, min(1.0, fraction))
        self.tqdm_span = max(0.0, min(1.0 - self.local_base, tqdm_span))
        self.message = message
        self._emit(self.local_base, message)

    def _emit(self, local_fraction: float, message: str) -> None:
        local = max(0.0, min(1.0, local_fraction))
        overall = 0.28 + 0.48 * local
        if overall + 1e-9 < self.last_overall:
            return
        self.last_overall = overall
        self.progress(overall, message)


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
        environment = _worker_environment(
            layout=resolved_layout,
            log_path=log_path,
        )
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
            self.command_runner(prepare_command, environment, None)
            self._prepared_backends.add(model.backend)

        progress(0.28, f"Запускаю: {model.short_title}…")
        command = build_worker_command(
            layout=resolved_layout,
            model_id=model_id,
            source=source,
            output_dir=output_dir,
            settings=settings,
        )
        parser = _WorkerProgressParser(progress)
        self.command_runner(command, environment, parser.feed)
        return read_worker_manifest(output_dir)


class PersistentStemWorker:
    """Один separator-процесс на весь сеанс Gradio.

    Внутри процесса текущая модель остаётся в VRAM. При повторном запуске
    того же checkpoint веса не загружаются заново. При смене модели старый
    объект освобождается, но скачанные файлы остаются в дисковом кэше.
    """

    def __init__(
        self,
        *,
        layout: RuntimeLayout,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.layout = layout
        self.command_runner = command_runner or _run_command
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._prepared = False
        self._project_root: Path | None = None

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
        if model.backend != "stems":
            raise ValueError("PersistentStemWorker получил не stem-модель.")
        with self._lock:
            project_root = _resolve_project_root(self.layout.project_root)
            self._project_root = project_root
            log_path = output_dir.parent / "model.log"
            environment = _worker_environment(
                layout=RuntimeLayout(
                    project_root=project_root,
                    cache_root=self.layout.cache_root,
                ),
                log_path=log_path,
            )
            self._prepare(environment, progress)
            self._ensure_server(environment, log_path)
            progress(0.28, f"Запускаю: {model.short_title}…")
            parser = _WorkerProgressParser(progress)
            payload = {
                "model_id": model_id,
                "input": str(source),
                "output_dir": str(output_dir),
                "settings": settings,
            }
            result = self._request(
                payload=payload,
                log_path=log_path,
                parser=parser,
            )
            if not result.get("ok"):
                detail = str(result.get("error") or "неизвестная ошибка")
                raise ValueError(f"Разделитель завершился с ошибкой: {detail}")
            return read_worker_manifest(output_dir)

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            try:
                assert process.stdin is not None
                process.stdin.write(json.dumps({"action": "shutdown"}) + "\n")
                process.stdin.flush()
                process.wait(timeout=10)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()

    def _prepare(
        self,
        environment: dict[str, str],
        progress: Callable[[float, str], None],
    ) -> None:
        if self._prepared:
            return
        assert self._project_root is not None
        progress(
            0.10,
            "Проверяю постоянную среду разделителя и кэш моделей…",
        )
        command = [
            str(self._project_root / "scripts" / "prepare_backend.sh"),
            "separator",
            str(self.layout.cache_root),
        ]
        self.command_runner(command, environment, None)
        self._prepared = True

    def _ensure_server(
        self,
        environment: dict[str, str],
        log_path: Path,
    ) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        assert self._project_root is not None
        python = (
            self.layout.cache_root
            / "envs"
            / "separator"
            / "bin"
            / "python"
        )
        server = self._project_root / "workers" / "separator_server.py"
        command = [str(python), str(server)]
        header = f"\n$ {shlex.join(command)}\n"
        _append_runtime_log(log_path, header)
        print(header, end="", flush=True)
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, **environment},
        )
        self._wait_until_ready(log_path)

    def _wait_until_ready(self, log_path: Path) -> None:
        process = self._require_process()
        assert process.stdout is not None
        for line in process.stdout:
            self._relay_line(line, log_path)
            if line.startswith(SERVER_READY_PREFIX):
                return
        raise ValueError("Постоянный процесс разделителя не запустился.")

    def _request(
        self,
        *,
        payload: dict[str, object],
        log_path: Path,
        parser: _WorkerProgressParser,
    ) -> dict[str, object]:
        process = self._require_process()
        if process.poll() is not None:
            raise ValueError("Постоянный процесс разделителя неожиданно завершён.")
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()
        for line in process.stdout:
            self._relay_line(line, log_path)
            parser.feed(line)
            if line.startswith(SERVER_RESULT_PREFIX):
                raw = line[len(SERVER_RESULT_PREFIX) :].strip()
                try:
                    result = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Разделитель вернул повреждённый ответ."
                    ) from error
                if not isinstance(result, dict):
                    raise ValueError("Разделитель вернул неправильный ответ.")
                return result
        raise ValueError("Связь с постоянным разделителем оборвалась.")

    def _relay_line(self, line: str, log_path: Path) -> None:
        print(line, end="", flush=True)
        _append_runtime_log(log_path, line)

    def _require_process(self) -> subprocess.Popen[str]:
        if self._process is None:
            raise ValueError("Постоянный процесс разделителя не создан.")
        return self._process


class RouterWorker:
    def __init__(
        self,
        *,
        layout: RuntimeLayout,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.subprocess = SubprocessWorker(
            layout=layout,
            command_runner=command_runner,
        )
        self.stems = PersistentStemWorker(
            layout=layout,
            command_runner=command_runner,
        )

    def run(
        self,
        *,
        model_id: str,
        source: Path,
        output_dir: Path,
        settings: dict[str, object],
        progress: Callable[[float, str], None],
    ) -> list[ModelResult]:
        if get_model(model_id).backend == "stems":
            return self.stems.run(
                model_id=model_id,
                source=source,
                output_dir=output_dir,
                settings=settings,
                progress=progress,
            )
        return self.subprocess.run(
            model_id=model_id,
            source=source,
            output_dir=output_dir,
            settings=settings,
            progress=progress,
        )


def build_worker_command(
    *,
    layout: RuntimeLayout,
    model_id: str,
    source: Path,
    output_dir: Path,
    settings: dict[str, Any],
) -> list[str]:
    model = get_model(model_id)
    if model.backend == "stems":
        raise ValueError("Stem-модели запускаются постоянным worker-сервером.")
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


def _worker_environment(
    *,
    layout: RuntimeLayout,
    log_path: Path,
) -> dict[str, str]:
    project_root = layout.project_root.resolve()
    python_path = str(project_root / "src")
    existing = os.environ.get("PYTHONPATH")
    if existing:
        python_path = os.pathsep.join((python_path, existing))
    return {
        "AUDIO_RESTORATION_CACHE": str(layout.cache_root),
        PROJECT_ROOT_ENV: str(project_root),
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": python_path,
        "MPLBACKEND": "Agg",
        JOB_LOG_ENV: str(log_path),
    }


def _resolve_project_root(configured_root: Path) -> Path:
    candidates: list[Path] = []
    override = os.environ.get(PROJECT_ROOT_ENV)
    if override:
        candidates.append(Path(override))
    candidates.extend([configured_root, Path.cwd()])

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


def _append_runtime_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _run_command(
    command: list[str],
    environment: dict[str, str],
    line_handler: LineHandler | None = None,
) -> None:
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
            if line_handler is not None:
                line_handler(line)
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
