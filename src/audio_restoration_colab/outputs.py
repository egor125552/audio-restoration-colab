from __future__ import annotations

import json
import re
import subprocess
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


def probe_audio_bitrate(source: Path) -> int | None:
    """Return the reported audio-stream bitrate in bits per second."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=bit_rate:format=bit_rate",
        "-of",
        "json",
        str(source),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return None

    candidates: list[object] = []
    streams = payload.get("streams")
    if isinstance(streams, list) and streams and isinstance(streams[0], dict):
        candidates.append(streams[0].get("bit_rate"))
    format_info = payload.get("format")
    if isinstance(format_info, dict):
        candidates.append(format_info.get("bit_rate"))

    for raw_value in candidates:
        try:
            bitrate = int(raw_value)
        except (TypeError, ValueError):
            continue
        if bitrate > 0:
            return bitrate
    return None


def _codec_bitrate(
    source_bitrate: int | None,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> str:
    bitrate = default if source_bitrate is None else int(source_bitrate)
    return str(max(minimum, min(maximum, bitrate)))


def build_ffmpeg_command(
    source: Path,
    target: Path,
    *,
    source_bitrate: int | None = None,
) -> list[str]:
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
        ".mp3": [
            "-c:a",
            "libmp3lame",
            "-b:a",
            _codec_bitrate(
                source_bitrate,
                default=320_000,
                minimum=32_000,
                maximum=320_000,
            ),
        ],
        ".wav": ["-c:a", "pcm_s24le"],
        ".flac": ["-c:a", "flac"],
        ".m4a": [
            "-c:a",
            "aac",
            "-b:a",
            _codec_bitrate(
                source_bitrate,
                default=320_000,
                minimum=32_000,
                maximum=512_000,
            ),
        ],
        ".aac": [
            "-c:a",
            "aac",
            "-b:a",
            _codec_bitrate(
                source_bitrate,
                default=320_000,
                minimum=32_000,
                maximum=512_000,
            ),
        ],
        ".opus": [
            "-c:a",
            "libopus",
            "-b:a",
            _codec_bitrate(
                source_bitrate,
                default=256_000,
                minimum=16_000,
                maximum=512_000,
            ),
        ],
        ".aiff": ["-c:a", "pcm_s24be"],
    }
    if suffix == ".ogg":
        if source_bitrate is None:
            codec_options[suffix] = ["-c:a", "libvorbis", "-q:a", "8"]
        else:
            codec_options[suffix] = [
                "-c:a",
                "libvorbis",
                "-b:a",
                _codec_bitrate(
                    source_bitrate,
                    default=256_000,
                    minimum=32_000,
                    maximum=500_000,
                ),
            ]
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
