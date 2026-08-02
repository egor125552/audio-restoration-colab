from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
    "vocals": "вокал",
    "instrumental": "минусовка",
    "drums": "барабаны",
    "bass": "бас",
    "guitar": "гитара",
    "piano": "фортепиано",
    "other": "остальные инструменты",
    "dry": "сухой звук без реверберации",
    "reverb": "выделенная реверберация и эхо",
    "bleed": "удалённое просачивание",
    "breaths": "дыхание и придыхания",
    "kick": "бочка",
    "snare": "рабочий барабан",
    "toms": "томы",
    "cymbals": "тарелки",
    "hihat": "хай-хэт",
    "ride": "райд",
    "crash": "крэш",
    "speech": "речь и диалоги",
    "music": "музыка",
    "sfx": "звуковые эффекты",
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


class JobProcessingError(ValueError):
    def __init__(self, message: str, *, log_path: Path) -> None:
        super().__init__(message)
        self.log_path = log_path


@dataclass(frozen=True)
class JobResult:
    files: list[Path]
    archive: Path
    primary_preview: Path | None
    secondary_preview: Path | None
    message: str
    log_path: Path
    raw_results: tuple[ModelResult, ...]
    preview_results: tuple[ModelResult, ...]
    stem_manifest: Path | None


class AudioJobService:
    def __init__(
        self,
        *,
        jobs_root: Path,
        worker: Worker,
        ffmpeg_runner: FfmpegRunner | None = None,
    ) -> None:
        self.jobs_root = _gradio_safe_jobs_root(jobs_root)
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
        input_dir = job_dir / "input"
        raw_dir = job_dir / "raw"
        preview_dir = job_dir / "previews"
        formatted_dir = job_dir / "results"
        log_path = job_dir / "model.log"
        input_dir.mkdir(parents=True)
        raw_dir.mkdir()
        preview_dir.mkdir()
        formatted_dir.mkdir()
        log_path.write_text(
            (
                f"Время запуска: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Модель: {model_id}\n"
                f"Исходный файл: {source.name}\n"
            ),
            encoding="utf-8",
        )

        try:
            progress(0.02, "Декодирую аудио в WAV для модели…")
            model_input = input_dir / "model-input.wav"
            self.ffmpeg_runner(build_ffmpeg_command(source, model_input))
            if not model_input.is_file():
                raise ValueError("Не удалось подготовить WAV для выбранной модели.")

            progress(0.05, "Подготавливаю выбранную модель…")
            raw_results = self.worker.run(
                model_id=model_id,
                source=model_input,
                output_dir=raw_dir,
                settings=settings,
                progress=progress,
            )
            if not raw_results:
                raise ValueError("Модель не вернула ни одного файла.")

            progress(0.77, "Создаю облегчённые MP3-превью…")
            preview_results = _create_preview_results(
                raw_results=raw_results,
                preview_dir=preview_dir,
                ffmpeg_runner=self.ffmpeg_runner,
                log_path=log_path,
            )

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

            stem_manifest = _write_stem_manifest(
                job_dir=job_dir,
                model_id=model_id,
                source=source,
                raw_results=raw_results,
                formatted_files=files,
            )
            progress(0.93, "Собираю ZIP-архив…")
            archive_members = [*files]
            if stem_manifest is not None:
                archive_members.append(stem_manifest)
            archive = create_result_zip(
                archive_members,
                job_dir / f"{source_name} - все результаты.zip",
            )
            progress(1.0, "Готово. Файлы можно слушать и скачивать.")
            _append_log(log_path, "\nГотово: обработка завершена успешно.\n")
            return JobResult(
                files=files,
                archive=archive,
                primary_preview=(
                    preview_results[0].path if preview_results else None
                ),
                secondary_preview=(
                    preview_results[1].path
                    if len(preview_results) > 1
                    else None
                ),
                message=(
                    f"Готово: создано дорожек — {len(files)}. "
                    "Для прослушивания используются облегчённые MP3-превью; "
                    "полные файлы остаются без потери выбранного качества."
                ),
                log_path=log_path,
                raw_results=tuple(raw_results),
                preview_results=tuple(preview_results),
                stem_manifest=stem_manifest,
            )
        except ValueError as error:
            _append_log(log_path, f"\nОШИБКА: {error}\n")
            print(f"[audio-restoration] ОШИБКА: {error}", flush=True)
            print(
                f"[audio-restoration] Полный лог: {log_path}",
                flush=True,
            )
            raise JobProcessingError(
                str(error),
                log_path=log_path,
            ) from error

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


def _create_preview_results(
    *,
    raw_results: list[ModelResult],
    preview_dir: Path,
    ffmpeg_runner: FfmpegRunner,
    log_path: Path,
) -> list[ModelResult]:
    previews: list[ModelResult] = []
    for result in raw_results:
        target = preview_dir / f"{result.role}.mp3"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(result.path),
            "-vn",
            "-map_metadata",
            "-1",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "96k",
            str(target),
        ]
        try:
            ffmpeg_runner(command)
            if not target.is_file():
                raise ValueError("FFmpeg не создал MP3-превью.")
        except ValueError as error:
            _append_log(
                log_path,
                (
                    "\nПРЕДУПРЕЖДЕНИЕ: облегчённое превью "
                    f"для {result.role} не создано: {error}\n"
                ),
            )
            previews.append(result)
            continue
        previews.append(ModelResult(role=result.role, path=target))
    return previews


def _write_stem_manifest(
    *,
    job_dir: Path,
    model_id: str,
    source: Path,
    raw_results: list[ModelResult],
    formatted_files: list[Path],
) -> Path | None:
    if len(raw_results) < 2:
        return None
    path = job_dir / "stem-manifest.json"
    payload = {
        "model_id": model_id,
        "source": source.name,
        "stems": [
            {
                "role": result.role,
                "title": ROLE_TITLES.get(result.role, result.role),
                "raw_path": str(result.path),
                "download_path": str(download),
            }
            for result, download in zip(
                raw_results,
                formatted_files,
                strict=True,
            )
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


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


def _gradio_safe_jobs_root(configured_root: Path) -> Path:
    requested = configured_root.expanduser().resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    try:
        requested.relative_to(temp_root)
    except ValueError:
        fallback = temp_root / "audio-restoration-work"
        print(
            "[audio-restoration] Рабочая папка перенаправлена в "
            f"{fallback}, чтобы Gradio мог отдавать результаты и логи.",
            flush=True,
        )
        return fallback
    return requested


def _append_log(log_path: Path, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(text)


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_stem(f"{path.stem} {index}")
        if not candidate.exists():
            return candidate
    raise ValueError("Слишком много файлов с одинаковым названием.")
