from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from pathlib import Path

FfmpegRunner = Callable[[list[str]], None]

ROLE_GROUPS = {
    "vocals": "vocals",
    "clean": "vocals",
    "dry": "vocals",
    "breaths": "vocals",
    "speech": "vocals",
    "drums": "drums",
    "kick": "drums",
    "snare": "drums",
    "toms": "drums",
    "cymbals": "drums",
    "hihat": "drums",
    "ride": "drums",
    "crash": "drums",
    "bass": "bass",
    "guitar": "guitar",
    "piano": "piano",
}

GROUP_TITLES = {
    "vocals": "Вокал",
    "drums": "Барабаны",
    "bass": "Бас",
    "guitar": "Гитары",
    "piano": "Клавишные",
    "other": "Остальное",
}


def build_mix(
    *,
    stem_paths: dict[str, str],
    selected_roles: list[str],
    gains: dict[str, float],
    ffmpeg_runner: FfmpegRunner | None = None,
) -> Path:
    if not selected_roles:
        raise ValueError("Выбери хотя бы одну дорожку для микса.")
    resolved: list[tuple[str, Path]] = []
    for role in selected_roles:
        raw_path = stem_paths.get(role)
        if not raw_path:
            continue
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Дорожка «{role}» больше недоступна.")
        resolved.append((role, path))
    if not resolved:
        raise ValueError("Ни одна выбранная дорожка не найдена.")

    job_root = resolved[0][1].parent.parent
    for _, path in resolved:
        try:
            path.relative_to(job_root)
        except ValueError as error:
            raise ValueError("Дорожки относятся к разным заданиям.") from error

    mixes = job_root / "mixes"
    mixes.mkdir(exist_ok=True)
    target = mixes / f"mix-{int(time.time() * 1000)}.wav"
    command = build_mix_command(
        resolved=resolved,
        gains=gains,
        target=target,
    )
    runner = ffmpeg_runner or _run_ffmpeg
    runner(command)
    if not target.is_file():
        raise ValueError("FFmpeg не создал пользовательский микс.")
    return target


def build_mix_command(
    *,
    resolved: list[tuple[str, Path]],
    gains: dict[str, float],
    target: Path,
) -> list[str]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    filters: list[str] = []
    labels: list[str] = []
    for index, (role, path) in enumerate(resolved):
        command.extend(["-i", str(path)])
        group = role_group(role)
        gain = max(0.0, min(2.0, float(gains.get(group, 1.0))))
        label = f"a{index}"
        filters.append(f"[{index}:a]volume={gain:.4f}[{label}]")
        labels.append(f"[{label}]")
    filters.append(
        "".join(labels)
        + f"amix=inputs={len(labels)}:duration=longest:normalize=0,"
        "alimiter=limit=0.98[mix]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[mix]",
            "-c:a",
            "pcm_s24le",
            str(target),
        ]
    )
    return command


def role_group(role: str) -> str:
    return ROLE_GROUPS.get(role, "other")


def _run_ffmpeg(command: list[str]) -> None:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise ValueError("FFmpeg не найден.") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        raise ValueError(
            "Не удалось собрать микс: "
            + (detail[-1] if detail else "неизвестная ошибка")
        )
