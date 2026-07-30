from __future__ import annotations

import re
import zipfile
from pathlib import Path


SUPPORTED_SOURCE_FORMATS = {
    ".aac",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
}


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = "".join(
        character
        if character.isalnum() or character in {" ", "-", "_"}
        else " "
        for character in stem
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .-_")
    return cleaned[:120] or "audio"


def output_extension(choice: str, source: Path) -> str:
    if choice == "mp3":
        return ".mp3"
    if choice == "wav":
        return ".wav"
    if choice != "source":
        raise ValueError(f"Неизвестный формат результата: {choice}")
    suffix = source.suffix.lower()
    return suffix if suffix in SUPPORTED_SOURCE_FORMATS else ".wav"


def build_ffmpeg_command(source: Path, target: Path) -> list[str]:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
    ]
    suffix = target.suffix.lower()
    codec_options = {
        ".mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
        ".wav": ["-c:a", "pcm_s24le"],
        ".flac": ["-c:a", "flac"],
        ".m4a": ["-c:a", "aac", "-b:a", "320k"],
        ".aac": ["-c:a", "aac", "-b:a", "320k"],
        ".ogg": ["-c:a", "libvorbis", "-q:a", "8"],
        ".opus": ["-c:a", "libopus", "-b:a", "256k"],
        ".aiff": ["-c:a", "pcm_s24be"],
    }
    try:
        command.extend(codec_options[suffix])
    except KeyError as error:
        raise ValueError(f"Неподдерживаемый формат результата: {suffix}") from error
    command.append(str(target))
    return command


def create_result_zip(files: list[Path], target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        target,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for result in files:
            archive.write(result, arcname=result.name)
    return target
