from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "workers"))

from audio_restoration_colab.catalog import get_model  # noqa: E402


def create_probe_audio(path: Path) -> None:
    sample_rate = 44_100
    duration = 0.9
    frames = round(sample_rate * duration)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        payload = bytearray()
        for index in range(frames):
            t = index / sample_rate
            envelope = min(1.0, t / 0.02, (duration - t) / 0.02)
            bass = 0.20 * math.sin(2.0 * math.pi * 110.0 * t)
            mid = 0.14 * math.sin(2.0 * math.pi * 440.0 * t)
            high = 0.08 * math.sin(2.0 * math.pi * (900.0 + 500.0 * t) * t)
            click = 0.30 * math.exp(-90.0 * (t % 0.225))
            left = max(-0.98, min(0.98, envelope * (bass + mid + high + click)))
            right = max(-0.98, min(0.98, envelope * (bass - mid + high + click)))
            for sample in (left, right):
                value = int(sample * 32767)
                payload.extend(value.to_bytes(2, "little", signed=True))
        output.writeframes(payload)


def validate_manifest(
    output_dir: Path,
    expected_roles: tuple[str, ...],
) -> dict[str, object]:
    import soundfile as sf

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("manifest.json не создан")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise RuntimeError("manifest.json не содержит результатов")
    roles: list[str] = []
    files: list[dict[str, object]] = []
    for item in outputs:
        role = str(item["role"])
        path = Path(str(item["path"]))
        if not path.is_file():
            raise RuntimeError(f"не найден WAV для роли {role}: {path}")
        info = sf.info(str(path))
        if info.frames <= 0 or info.duration <= 0:
            raise RuntimeError(f"пустой WAV для роли {role}")
        roles.append(role)
        files.append(
            {
                "role": role,
                "path": str(path),
                "frames": info.frames,
                "samplerate": info.samplerate,
                "duration": info.duration,
                "channels": info.channels,
                "bytes": path.stat().st_size,
            }
        )
    missing = sorted(set(expected_roles) - set(roles))
    if missing:
        raise RuntimeError("не хватает ролей: " + ", ".join(missing))
    return {"roles": roles, "files": files}


def run_stem_model(
    *, model_id: str, source: Path, output_dir: Path, repeat: int
) -> list[dict[str, object]]:
    from separator_server import SeparatorSession

    session = SeparatorSession()
    runs: list[dict[str, object]] = []
    try:
        for index in range(1, repeat + 1):
            current_output = output_dir / f"run-{index}"
            current_output.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            session.run(
                {
                    "model_id": model_id,
                    "input": str(source),
                    "output_dir": str(current_output),
                    "settings": {
                        "quality": "balanced",
                        "segment": 256,
                        "overlap": 8,
                        "chunk_minutes": 10,
                        "keep_loaded": True,
                    },
                }
            )
            elapsed = time.perf_counter() - started
            validation = validate_manifest(
                current_output, get_model(model_id).output_roles
            )
            runs.append(
                {
                    "run": index,
                    "elapsed_seconds": elapsed,
                    "cached_model_after_run": session.model_id,
                    **validation,
                }
            )
    finally:
        session.unload()
    return runs


def run_denoise_model(
    *, model_id: str, source: Path, output_dir: Path
) -> list[dict[str, object]]:
    current_output = output_dir / "run-1"
    current_output.mkdir(parents=True, exist_ok=True)
    python = Path(sys.executable)
    command = [
        str(python),
        str(PROJECT_ROOT / "workers" / "separator_worker.py"),
        "--model-id",
        model_id,
        "--input",
        str(source),
        "--output-dir",
        str(current_output),
        "--settings-json",
        json.dumps({"quality": "balanced", "segment": 256}),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "workers"))
            ),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(
            f"separator_worker завершился с кодом {completed.returncode}"
        )
    validation = validate_manifest(current_output, get_model(model_id).output_roles)
    return [{"run": 1, "elapsed_seconds": elapsed, **validation}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--work-root", type=Path, default=Path("probe-artifacts"))
    arguments = parser.parse_args()

    model = get_model(arguments.model_id)
    if model.backend not in {"stems", "separator"}:
        raise SystemExit(f"Модель {arguments.model_id} не относится к разделителю")

    work_root = arguments.work_root.resolve()
    source = work_root / "input-900ms.wav"
    output_dir = work_root / arguments.model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    create_probe_audio(source)

    report: dict[str, object] = {
        "model_id": arguments.model_id,
        "title": model.title,
        "backend": model.backend,
        "model_filename": model.model_filename,
        "ensemble_preset": model.ensemble_preset,
        "expected_roles": list(model.output_roles),
        "input": str(source),
        "input_duration_seconds": 0.9,
        "python": sys.version,
        "platform": sys.platform,
        "success": False,
    }
    try:
        if model.backend == "stems":
            runs = run_stem_model(
                model_id=arguments.model_id,
                source=source,
                output_dir=output_dir,
                repeat=max(1, arguments.repeat),
            )
        else:
            runs = run_denoise_model(
                model_id=arguments.model_id,
                source=source,
                output_dir=output_dir,
            )
        report["runs"] = runs
        report["success"] = True
    except Exception as error:  # noqa: BLE001
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report_path = output_dir / "probe-report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("@@PROBE_REPORT@@" + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
