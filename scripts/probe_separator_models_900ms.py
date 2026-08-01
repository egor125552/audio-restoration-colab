from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import struct
import subprocess
import sys
import time
import traceback
import wave
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "workers"))

from audio_restoration_colab.catalog import get_model  # noqa: E402
from separator_server import (  # noqa: E402
    SeparatorSession,
    _canonicalize_outputs,
    _resolve_paths,
)

PROBE_SECONDS = 0.9
SAMPLE_RATE = 44_100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Выполнить настоящий полный инференс separator-моделей "
            "на 900-мс WAV и проверить повторный запуск без перезагрузки."
        )
    )
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def create_probe_audio(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = round(SAMPLE_RATE * PROBE_SECONDS)
    frames = bytearray()
    for index in range(frame_count):
        t = index / SAMPLE_RATE
        envelope = min(1.0, t / 0.03, (PROBE_SECONDS - t) / 0.03)
        envelope = max(0.0, envelope)
        pulse = 0.0
        pulse_phase = t % 0.15
        if pulse_phase < 0.012:
            pulse = 0.28 * math.exp(-pulse_phase * 220.0)
        left = envelope * (
            0.28 * math.sin(2.0 * math.pi * 196.0 * t)
            + 0.18 * math.sin(2.0 * math.pi * 784.0 * t)
            + pulse
        )
        right = envelope * (
            0.26 * math.sin(2.0 * math.pi * 247.0 * t + 0.2)
            + 0.17 * math.sin(2.0 * math.pi * 988.0 * t)
            - pulse * 0.65
        )
        left_i16 = max(-32768, min(32767, round(left * 32767)))
        right_i16 = max(-32768, min(32767, round(right * 32767)))
        frames.extend(struct.pack("<hh", left_i16, right_i16))

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(frames)
    return path.resolve()


def audio_info(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"FFprobe не увидел аудиопоток в {path}")
    stream = streams[0]
    duration = float(stream.get("duration") or 0.0)
    if duration <= 0.0:
        raise ValueError(f"У результата нулевая длительность: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "duration_seconds": duration,
    }


def validate_outputs(
    *,
    output_dir: Path,
    expected_roles: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Не создан manifest.json: {output_dir}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = payload.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("В manifest.json нет массива outputs")
    by_role = {str(item.get("role")): Path(str(item.get("path"))) for item in outputs}
    missing = [role for role in expected_roles if role not in by_role]
    if missing:
        raise ValueError("Не созданы ожидаемые роли: " + ", ".join(missing))
    result: dict[str, dict[str, Any]] = {}
    for role in expected_roles:
        path = by_role[role]
        if not path.is_absolute():
            path = output_dir / path
        if not path.is_file():
            raise ValueError(f"Файл роли {role} отсутствует: {path}")
        result[role] = audio_info(path)
    return result


def set_separator_output_dir(separator: Any, output_dir: Path) -> None:
    separator.output_dir = str(output_dir)
    model_instance = getattr(separator, "model_instance", None)
    if model_instance is not None and hasattr(model_instance, "output_dir"):
        model_instance.output_dir = str(output_dir)


def run_plain_separator_model(
    *,
    model_id: str,
    source: Path,
    root: Path,
) -> dict[str, Any]:
    from audio_separator.separator import Separator

    model = get_model(model_id)
    if not model.model_filename:
        raise ValueError(f"Для {model_id} не указан checkpoint")
    cache_root = Path(
        os.environ.get("AUDIO_RESTORATION_CACHE", "/tmp/audio-restoration-models")
    )
    model_dir = cache_root / "weights" / "separator"
    model_dir.mkdir(parents=True, exist_ok=True)
    first_dir = root / "first"
    second_dir = root / "second"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)

    load_started = time.perf_counter()
    separator = Separator(
        output_dir=str(first_dir),
        model_file_dir=str(model_dir),
        output_format="WAV",
        use_soundfile=True,
        use_autocast=False,
        mdxc_params={
            "segment_size": 128,
            "override_model_segment_size": True,
            "batch_size": 1,
            "overlap": 2,
        },
    )
    separator.load_model(model_filename=model.model_filename)
    load_seconds = time.perf_counter() - load_started
    engine_identity = id(separator.model_instance)

    first_started = time.perf_counter()
    raw_first = separator.separate(str(source))
    first_paths = _resolve_paths(first_dir, raw_first)
    first_mapped = _canonicalize_outputs(
        paths=first_paths,
        output_dir=first_dir,
        expected_roles=model.output_roles,
    )
    from common import write_manifest

    write_manifest(first_dir, list(first_mapped.items()))
    first_seconds = time.perf_counter() - first_started

    set_separator_output_dir(separator, second_dir)
    second_started = time.perf_counter()
    raw_second = separator.separate(str(source))
    second_paths = _resolve_paths(second_dir, raw_second)
    second_mapped = _canonicalize_outputs(
        paths=second_paths,
        output_dir=second_dir,
        expected_roles=model.output_roles,
    )
    write_manifest(second_dir, list(second_mapped.items()))
    second_seconds = time.perf_counter() - second_started

    if id(separator.model_instance) != engine_identity:
        raise ValueError("Повторный проход пересоздал model_instance")
    first_outputs = validate_outputs(
        output_dir=first_dir,
        expected_roles=model.output_roles,
    )
    second_outputs = validate_outputs(
        output_dir=second_dir,
        expected_roles=model.output_roles,
    )
    release = getattr(separator, "release", None)
    if callable(release):
        release()
    return {
        "model_id": model_id,
        "checkpoint": model.model_filename,
        "load_seconds": load_seconds,
        "first_inference_seconds": first_seconds,
        "cached_inference_seconds": second_seconds,
        "cache_reused": True,
        "first_outputs": first_outputs,
        "second_outputs": second_outputs,
    }


def run_stem_model(
    *,
    model_id: str,
    source: Path,
    root: Path,
) -> dict[str, Any]:
    model = get_model(model_id)
    session = SeparatorSession()
    first_dir = root / "first"
    second_dir = root / "second"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    settings = {
        "quality": "maximum",
        "segment": 128,
        "overlap": 2,
        "chunk_minutes": 1,
        "keep_loaded": True,
    }

    first_started = time.perf_counter()
    session.run(
        {
            "model_id": model_id,
            "input": str(source),
            "output_dir": str(first_dir),
            "settings": settings,
        }
    )
    first_seconds = time.perf_counter() - first_started
    engine_identity = id(session.engine)
    if session.engine is None:
        raise ValueError("После первого прохода модель не осталась в кэше")

    second_started = time.perf_counter()
    session.run(
        {
            "model_id": model_id,
            "input": str(source),
            "output_dir": str(second_dir),
            "settings": settings,
        }
    )
    second_seconds = time.perf_counter() - second_started
    if session.engine is None or id(session.engine) != engine_identity:
        raise ValueError("Повторный проход перезагрузил модель")

    first_outputs = validate_outputs(
        output_dir=first_dir,
        expected_roles=model.output_roles,
    )
    second_outputs = validate_outputs(
        output_dir=second_dir,
        expected_roles=model.output_roles,
    )
    session.unload()
    return {
        "model_id": model_id,
        "checkpoint": model.model_filename,
        "ensemble_preset": model.ensemble_preset,
        "first_load_and_inference_seconds": first_seconds,
        "cached_inference_seconds": second_seconds,
        "cache_reused": True,
        "first_outputs": first_outputs,
        "second_outputs": second_outputs,
    }


def run_model(model_id: str, source: Path, root: Path) -> dict[str, Any]:
    model = get_model(model_id)
    started = time.perf_counter()
    try:
        if model.backend == "stems":
            details = run_stem_model(model_id=model_id, source=source, root=root)
        elif model.backend == "separator":
            details = run_plain_separator_model(
                model_id=model_id,
                source=source,
                root=root,
            )
        else:
            raise ValueError(
                f"{model_id}: backend {model.backend} не является разделителем"
            )
        details.update(
            {
                "status": "passed",
                "title": model.title,
                "purpose": model.purpose,
                "total_seconds": time.perf_counter() - started,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        )
        return details
    except Exception as error:  # noqa: BLE001
        return {
            "model_id": model_id,
            "title": model.title,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "total_seconds": time.perf_counter() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = create_probe_audio(output_dir / "probe-900ms.wav")
    source_info = audio_info(source)
    results: list[dict[str, Any]] = []

    for model_id in args.models:
        print(f"\n===== REAL 900 MS INFERENCE: {model_id} =====", flush=True)
        model_root = output_dir / model_id
        model_root.mkdir(parents=True, exist_ok=True)
        result = run_model(model_id, source, model_root)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    report = {
        "probe_seconds": PROBE_SECONDS,
        "source": source_info,
        "python": sys.version,
        "platform": sys.platform,
        "models": results,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    failed = [item["model_id"] for item in results if item["status"] != "passed"]
    if failed:
        raise SystemExit("Не прошли реальный инференс: " + ", ".join(failed))
    print(f"Все модели прошли. Отчёт: {report_path}", flush=True)


if __name__ == "__main__":
    main()
