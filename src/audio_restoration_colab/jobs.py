from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from .catalog import normalize_settings
from .outputs import (
    build_ffmpeg_command,
    create_result_zip,
    output_extension,
    safe_stem,
)
from .runtime import ModelResult


JobProgress = Callable[[float, str], None]
FfmpegRunner = Callable[[list[str]], None]

SUPPORTED_INPUT_FORMATS = {
    ".aac",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
}

ROLE_TITLES = {
    "clean": "очищенный звук",
    "noise": "выделенный шум",
    "restored": "дорисованный звук",
}


class Worker(Protocol):
    def run(
        self,
        *,
        model_id: str,
        source: Path,
        output_dir: Path,
        settings: dict[str, object],
        progress: JobProgress,
    ) -> list[ModelResult]:
        ...


@dataclass(frozen=True)
class JobResult:
    files: list[Path]
    archive: Path
    primary_preview: Path | None
    secondary_preview: Path | None
    message: str


class AudioJobService:
    def __init__(
        self,
        *,
        jobs_root: Path,
        worker: Worker,
        ffmpeg_runner: FfmpegRunner | None = None,
    ) -> None:
        self.jobs_root = jobs_root
        self.worker = worker
        self.ffmpeg_runner = ffmpeg_runner or _run_ffmpeg

    def process(
        self,
        *,
        source: Path,
        model_id: str,
        format_choice: str,
        raw_settings: dict[str, object] | None,
        progress: JobProgress,
    ) -> JobResult:
        source = validate_source(source)
        settings = normalize_settings(model_id, raw_settings)
        self.cleanup_old_jobs()
        job_dir = self._new_job_dir()
        raw_dir = job_dir / "raw"
        formatted_dir = job_dir / "results"
        raw_dir.mkdir(parents=True)
        formatted_dir.mkdir()

        progress(0.05, "Подготавливаю выбранную модель…")
        raw_results = self.worker.run(
            model_id=model_id,
            source=source,
            output_dir=raw_dir,
            settings=settings,
            progress=progress,
        )
        if not raw_results:
            raise ValueError("Модель не вернула ни одного файла.")

        extension = output_extension(format_choice, source)
        source_name = safe_stem(source.name)
        files: list[Path] = []
        total = len(raw_results)
        for index, raw_result in enumerate(raw_results, start=1):
            role_title = ROLE_TITLES.get(raw_result.role, raw_result.role)
            filename = f"{source_name} - {role_title}{extension}"
            target = _unique_path(formatted_dir / filename)
            progress(
                0.78 + (0.12 * index / total),
                f"Сохраняю результат {index} из {total}…",
            )
            self.ffmpeg_runner(
                build_ffmpeg_command(raw_result.path, target)
            )
            if not target.is_file():
                raise ValueError("Не удалось сохранить готовый аудиофайл.")
            files.append(target)

        progress(0.93, "Собираю ZIP-архив…")
        archive = create_result_zip(
            files,
            job_dir / f"{source_name} - все результаты.zip",
        )
        progress(1.0, "Готово. Файлы можно слушать и скачивать.")
        return JobResult(
            files=files,
            archive=archive,
            primary_preview=files[0] if files else None,
            secondary_preview=files[1] if len(files) > 1 else None,
            message=(
                f"Готово: создано файлов — {len(files)}. "
                "Ниже можно скачать каждый файл или общий ZIP."
            ),
        )

    def cleanup_old_jobs(
        self,
        *,
        max_age_seconds: int = 21_600,
        now: float | None = None,
    ) -> None:
        if not self.jobs_root.is_dir():
            return
        current_time = time.time() if now is None else now
        for candidate in self.jobs_root.iterdir():
            if not candidate.is_dir():
                continue
            age = current_time - candidate.stat().st_mtime
            if age > max_age_seconds:
                shutil.rmtree(candidate)

    def _new_job_dir(self) -> Path:
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        job_dir = self.jobs_root / f"{timestamp}-{uuid.uuid4().hex[:8]}"
        job_dir.mkdir()
        return job_dir


def validate_source(source: Path) -> Path:
    resolved = source.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError("Загруженный аудиофайл не найден.")
    if resolved.suffix.lower() not in SUPPORTED_INPUT_FORMATS:
        raise ValueError(
            "Этот формат не поддерживается. Загрузи WAV, MP3, FLAC, "
            "M4A, AAC, OGG, OPUS, WMA или WEBM."
        )
    return resolved


def _run_ffmpeg(command: list[str]) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise ValueError("FFmpeg не найден в среде запуска.") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip().splitlines()
        short_detail = detail[-1] if detail else "неизвестная ошибка"
        raise ValueError(
            f"Не удалось преобразовать формат: {short_detail}"
        ) from error


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_stem(f"{path.stem} {index}")
        if not candidate.exists():
            return candidate
    raise ValueError("Слишком много файлов с одинаковым названием.")
