from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(
    os.environ.get("AUDIO_RESTORATION_PROJECT_ROOT", Path(__file__).parents[1])
).resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "workers"))

from audio_restoration_colab.catalog import get_model  # noqa: E402
from common import report_progress, write_manifest  # noqa: E402

READY_PREFIX = "@@AUDIO_RESTORATION_SERVER_READY@@"
RESULT_PREFIX = "@@AUDIO_RESTORATION_SERVER_RESULT@@"


class SeparatorSession:
    def __init__(self) -> None:
        self.engine = None
        self.engine_kind: str | None = None
        self.cache_key: tuple[object, ...] | None = None
        self.model_id: str | None = None

    def run(self, request: dict[str, Any]) -> None:
        model_id = str(request["model_id"])
        source = Path(str(request["input"])).resolve()
        output_dir = Path(str(request["output_dir"])).resolve()
        settings = dict(request.get("settings") or {})
        model = get_model(model_id)
        if model.backend != "stems":
            raise ValueError(f"{model_id} не относится к stem-разделителю.")
        if not source.is_file():
            raise ValueError(f"Входной файл не найден: {source}")
        output_dir.mkdir(parents=True, exist_ok=True)

        report_progress(0.02, "Разделитель: проверяю кэш модели…")
        engine = self._get_engine(
            model_id=model_id,
            output_dir=output_dir,
            settings=settings,
        )
        chunk_minutes = max(1, int(settings.get("chunk_minutes", 10)))
        duration = _probe_duration(source)
        if duration > chunk_minutes * 60 * 1.1:
            raw_results = self._separate_long(
                engine=engine,
                source=source,
                output_dir=output_dir,
                expected_roles=model.output_roles,
                chunk_seconds=chunk_minutes * 60,
            )
        else:
            raw_results = self._separate_once(
                engine=engine,
                source=source,
                output_dir=output_dir,
                expected_roles=model.output_roles,
            )

        write_manifest(
            output_dir,
            [(role, path) for role, path in raw_results.items()],
        )
        report_progress(1.0, "Разделитель: готово.")

        if not bool(settings.get("keep_loaded", True)):
            self.unload()

    def unload(self) -> None:
        if self.engine is None:
            return
        release = getattr(self.engine, "release", None)
        if callable(release):
            try:
                release()
            except Exception:  # noqa: BLE001
                pass
        self.engine = None
        self.engine_kind = None
        self.cache_key = None
        self.model_id = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        print("[stem-server] Активная модель выгружена.", flush=True)

    def _get_engine(
        self,
        *,
        model_id: str,
        output_dir: Path,
        settings: dict[str, Any],
    ):
        model = get_model(model_id)
        quality = str(settings.get("quality", "balanced"))
        segment = int(settings.get("segment", 256))
        overlap = int(settings.get("overlap", 8))
        use_autocast = quality != "maximum"
        engine_kind = (
            "demucs"
            if (model.model_filename or "").startswith("demucs:")
            else "audio_separator"
        )
        cache_key = (
            model_id,
            engine_kind,
            segment,
            overlap,
            use_autocast,
            model.model_filename,
            model.ensemble_preset,
        )
        if self.engine is not None and self.cache_key == cache_key:
            self._set_output_dir(self.engine, output_dir)
            report_progress(
                0.08,
                "Разделитель: использую уже загруженную модель из VRAM.",
            )
            return self.engine

        if self.engine is not None:
            report_progress(
                0.06,
                "Разделитель: освобождаю предыдущую модель для новой задачи…",
            )
            self.unload()

        report_progress(
            0.09,
            "Разделитель: создаю постоянный экземпляр модели…",
        )
        if engine_kind == "demucs":
            engine = self._load_demucs(model.model_filename or "")
        else:
            engine = self._load_audio_separator(
                model_id=model_id,
                output_dir=output_dir,
                quality=quality,
                segment=segment,
                overlap=overlap,
                use_autocast=use_autocast,
            )

        self.engine = engine
        self.engine_kind = engine_kind
        self.cache_key = cache_key
        self.model_id = model_id
        report_progress(
            0.20,
            "Разделитель: модель загружена и останется в памяти для повторов.",
        )
        return engine

    def _load_demucs(self, model_filename: str):
        model_name = model_filename.split(":", 1)[1]
        report_progress(
            0.12,
            "Разделитель: загружаю проверенный Demucs checkpoint; "
            "скачивание требуется только один раз…",
        )
        import torch
        from demucs_infer import DemucsSession

        cache_root = Path(
            os.environ.get(
                "AUDIO_RESTORATION_CACHE",
                "/content/audio-restoration-models",
            )
        )
        cache_dir = cache_root / "weights" / "demucs-infer"
        cache_dir.mkdir(parents=True, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        session = DemucsSession(
            model=model_name,
            device=device,
            cache_dir=cache_dir,
        )
        session.load()
        return session

    def _load_audio_separator(
        self,
        *,
        model_id: str,
        output_dir: Path,
        quality: str,
        segment: int,
        overlap: int,
        use_autocast: bool,
    ):
        from audio_separator.separator import Separator

        model = get_model(model_id)
        cache_root = Path(
            os.environ.get(
                "AUDIO_RESTORATION_CACHE",
                "/content/audio-restoration-models",
            )
        )
        model_dir = cache_root / "weights" / "separator"
        model_dir.mkdir(parents=True, exist_ok=True)
        demucs_overlap = min(0.75, max(0.1, overlap / 16.0))
        separator = Separator(
            output_dir=str(output_dir),
            model_file_dir=str(model_dir),
            output_format="WAV",
            use_soundfile=True,
            use_autocast=use_autocast,
            mdxc_params={
                "segment_size": segment,
                "override_model_segment_size": True,
                "batch_size": 1,
                "overlap": overlap,
            },
            demucs_params={
                "segment_size": "Default",
                "shifts": 2 if quality != "fast" else 1,
                "overlap": demucs_overlap,
                "segments_enabled": True,
            },
            ensemble_preset=model.ensemble_preset,
        )
        if model.ensemble_preset:
            report_progress(
                0.12,
                "Разделитель: загружаю модели ансамбля; "
                "скачивание требуется только один раз…",
            )
            separator.load_model()
        elif model.model_filename:
            report_progress(
                0.12,
                "Разделитель: загружаю веса; "
                "скачивание требуется только один раз…",
            )
            separator.load_model(model_filename=model.model_filename)
        else:
            raise ValueError(f"Для {model_id} не задан checkpoint или preset.")
        return separator

    def _separate_once(
        self,
        *,
        engine,
        source: Path,
        output_dir: Path,
        expected_roles: tuple[str, ...],
    ) -> dict[str, Path]:
        report_progress(
            0.24,
            "Разделитель: анализирую песню —",
            tqdm_span=0.68,
        )
        paths = self._run_engine(
            engine=engine,
            source=source,
            output_dir=output_dir,
        )
        return _canonicalize_outputs(
            paths=paths,
            output_dir=output_dir,
            expected_roles=expected_roles,
        )

    def _separate_long(
        self,
        *,
        engine,
        source: Path,
        output_dir: Path,
        expected_roles: tuple[str, ...],
        chunk_seconds: int,
    ) -> dict[str, Path]:
        report_progress(
            0.22,
            "Разделитель: длинный файл — режу его на безопасные куски…",
        )
        with tempfile.TemporaryDirectory(
            prefix="stem-chunks-",
            dir=output_dir,
        ) as temp:
            temp_root = Path(temp)
            chunks = _split_audio(
                source=source,
                output_dir=temp_root / "input",
                chunk_seconds=chunk_seconds,
            )
            if not chunks:
                raise ValueError("Не удалось разделить длинный файл на части.")
            by_role: dict[str, list[Path]] = {
                role: [] for role in expected_roles
            }
            for index, chunk in enumerate(chunks, start=1):
                fraction = 0.24 + 0.62 * ((index - 1) / len(chunks))
                report_progress(
                    fraction,
                    f"Разделитель: крупный кусок {index} из {len(chunks)} —",
                    tqdm_span=0.62 / len(chunks),
                )
                chunk_output = temp_root / f"result-{index:04d}"
                chunk_output.mkdir()
                paths = self._run_engine(
                    engine=engine,
                    source=chunk,
                    output_dir=chunk_output,
                )
                mapped = _canonicalize_outputs(
                    paths=paths,
                    output_dir=chunk_output,
                    expected_roles=expected_roles,
                )
                for role in expected_roles:
                    by_role[role].append(mapped[role])

            report_progress(0.90, "Разделитель: объединяю крупные куски…")
            results: dict[str, Path] = {}
            for role, parts in by_role.items():
                target = output_dir / f"{role}.wav"
                _concat_audio(parts, target)
                results[role] = target
            return results

    def _run_engine(
        self,
        *,
        engine,
        source: Path,
        output_dir: Path,
    ) -> list[Path]:
        if self.engine_kind == "demucs":
            return _run_demucs(
                session=engine,
                source=source,
                output_dir=output_dir,
            )
        self._set_output_dir(engine, output_dir)
        raw_paths = engine.separate(str(source))
        return _resolve_paths(output_dir, raw_paths)

    @staticmethod
    def _set_output_dir(engine, output_dir: Path) -> None:
        if not hasattr(engine, "output_dir"):
            return
        engine.output_dir = str(output_dir)
        model_instance = getattr(engine, "model_instance", None)
        if model_instance is not None and hasattr(model_instance, "output_dir"):
            model_instance.output_dir = str(output_dir)


def _run_demucs(
    *,
    session,
    source: Path,
    output_dir: Path,
) -> list[Path]:
    from demucs_infer.audio import save_audio

    output_dir.mkdir(parents=True, exist_ok=True)
    _, stems = session.infer(str(source))
    if not isinstance(stems, dict) or not stems:
        raise ValueError("Demucs не вернул дорожки.")
    paths: list[Path] = []
    for name, waveform in stems.items():
        target = output_dir / f"{name}.wav"
        tensor = waveform.detach().cpu()
        save_audio(tensor, str(target), session.samplerate)
        paths.append(target)
    return paths


def _resolve_paths(output_dir: Path, raw_paths: Any) -> list[Path]:
    if not raw_paths:
        paths = list(output_dir.glob("*.wav"))
    elif isinstance(raw_paths, (str, Path)):
        paths = [Path(raw_paths)]
    else:
        paths = [Path(item) for item in raw_paths]
    resolved: list[Path] = []
    for path in paths:
        candidate = path if path.is_absolute() else output_dir / path
        if not candidate.is_file():
            fallback = output_dir / path.name
            if fallback.is_file():
                candidate = fallback
        if candidate.is_file():
            resolved.append(candidate.resolve())
    if not resolved:
        raise ValueError("Модель не вернула ни одного WAV-файла.")
    return resolved


def _canonicalize_outputs(
    *,
    paths: list[Path],
    output_dir: Path,
    expected_roles: tuple[str, ...],
) -> dict[str, Path]:
    assignments: dict[str, Path] = {}
    unused = list(paths)
    for role in expected_roles:
        matched = next(
            (
                path
                for path in unused
                if _matches_role(
                    path=path,
                    role=role,
                    expected_roles=expected_roles,
                )
            ),
            None,
        )
        if matched is not None:
            assignments[role] = matched
            unused.remove(matched)

    for role in expected_roles:
        if role not in assignments and unused:
            assignments[role] = unused.pop(0)

    missing = [role for role in expected_roles if role not in assignments]
    if missing:
        raise ValueError(
            "Модель не вернула ожидаемые дорожки: " + ", ".join(missing)
        )

    canonical: dict[str, Path] = {}
    for role, source in assignments.items():
        target = output_dir / f"{role}.wav"
        if source.resolve() != target.resolve():
            if target.exists():
                target.unlink()
            shutil.copy2(source, target)
        canonical[role] = target.resolve()
    return canonical


def _matches_role(
    *,
    path: Path,
    role: str,
    expected_roles: tuple[str, ...],
) -> bool:
    name = path.stem.lower().replace("-", " ").replace("_", " ")
    no_reverb = "no reverb" in name or "no echo" in name
    aliases = {
        "vocals": ("vocals", "vocal"),
        "instrumental": ("instrumental", "karaoke", "no vocals", "novocals"),
        "drums": ("drums", "drum"),
        "bass": ("bass",),
        "guitar": ("guitar",),
        "piano": ("piano", "keys"),
        "other": ("other", "non vocals"),
        "dry": ("dry", "dereverb", "no reverb", "no echo"),
        "reverb": ("reverb", "echo"),
        "clean": ("clean", "dry", "no bleed", "no aspiration", "no noise"),
        "bleed": ("bleed",),
        "breaths": ("aspiration", "breath"),
        "kick": ("kick", "bombo"),
        "snare": ("snare", "redoblante"),
        "toms": ("toms", "tom"),
        "cymbals": ("cymbals", "platillos"),
        "hihat": ("hihat", "hi hat", "hh"),
        "ride": ("ride",),
        "crash": ("crash",),
        "speech": ("speech", "dialog", "dialogue"),
        "music": ("music",),
        "sfx": ("sfx", "effect"),
    }
    if role == "reverb" and no_reverb:
        return False
    if role == "instrumental" and "other" in name:
        return "other" not in expected_roles
    return any(alias in name for alias in aliases.get(role, (role,)))


def _probe_duration(source: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return 0.0


def _split_audio(
    *,
    source: Path,
    output_dir: Path,
    chunk_seconds: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "chunk-%04d.wav"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-reset_timestamps",
            "1",
            "-c:a",
            "pcm_f32le",
            str(pattern),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(completed.stderr.strip() or "FFmpeg split failed.")
    return sorted(output_dir.glob("chunk-*.wav"))


def _concat_audio(parts: list[Path], target: Path) -> None:
    if not parts:
        raise ValueError(f"Нет частей для объединения {target.name}.")
    list_path = target.with_suffix(".concat.txt")
    list_path.write_text(
        "\n".join(f"file '{path.as_posix()}'" for path in parts) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    list_path.unlink(missing_ok=True)
    if completed.returncode != 0 or not target.is_file():
        raise ValueError(
            completed.stderr.strip() or f"Не удалось собрать {target.name}."
        )


def _result(payload: dict[str, Any]) -> None:
    print(
        RESULT_PREFIX + json.dumps(payload, ensure_ascii=False),
        flush=True,
    )


def main() -> None:
    session = SeparatorSession()
    print(READY_PREFIX + "1", flush=True)
    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            request = json.loads(stripped)
            if request.get("action") == "shutdown":
                session.unload()
                _result({"ok": True, "shutdown": True})
                return
            session.run(request)
            _result(
                {
                    "ok": True,
                    "model_id": request.get("model_id"),
                    "cached_model": session.model_id,
                }
            )
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            _result(
                {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )


if __name__ == "__main__":
    main()
