from __future__ import annotations

from pathlib import Path

from common import model_cache_root, parse_worker_args, write_manifest

MODEL_FILES = {
    "denoise_normal": "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
    "denoise_aggressive": (
        "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt"
    ),
}

QUALITY_OVERLAP = {"fast": 2, "balanced": 8, "maximum": 16}


def main() -> None:
    arguments, settings = parse_worker_args(
        "Очистка аудио с помощью Mel-Band RoFormer.",
        set(MODEL_FILES),
    )
    from audio_separator.separator import Separator

    overlap = QUALITY_OVERLAP.get(str(settings.get("quality")), 8)
    segment = int(settings.get("segment", 256))
    separator = Separator(
        output_dir=str(arguments.output_dir),
        model_file_dir=str(model_cache_root() / "weights" / "separator"),
        output_format="WAV",
        use_autocast=True,
        mdxc_params={
            "segment_size": segment,
            "override_model_segment_size": True,
            "batch_size": 1,
            "overlap": overlap,
        },
    )
    separator.load_model(model_filename=MODEL_FILES[arguments.model_id])
    raw_paths = separator.separate(str(arguments.input))
    paths = [_resolve_output(arguments.output_dir, item) for item in raw_paths]
    if not paths:
        raise RuntimeError("Модель DeNoise не вернула результат.")
    if len(paths) == 1:
        clean = paths[0]
        noise = _create_noise_delta(arguments.input, clean, arguments.output_dir)
    else:
        clean, noise = _identify_denoise_outputs(paths)
    write_manifest(
        arguments.output_dir,
        [("clean", clean), ("noise", noise)],
    )


def _create_noise_delta(source: Path, clean: Path, output_dir: Path) -> Path:
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    original, original_rate = sf.read(str(source), dtype="float32", always_2d=True)
    restored, restored_rate = sf.read(str(clean), dtype="float32", always_2d=True)
    if restored_rate != original_rate:
        restored = resample_poly(restored, original_rate, restored_rate, axis=0)
    if restored.shape[1] != original.shape[1]:
        restored = np.repeat(restored[:, :1], original.shape[1], axis=1)
    length = min(len(original), len(restored))
    target = output_dir / "noise.wav"
    sf.write(str(target), original[:length] - restored[:length], original_rate)
    return target


def _resolve_output(output_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = output_dir / candidate
    if candidate.is_file():
        return candidate
    fallback = output_dir / Path(raw_path).name
    if fallback.is_file():
        return fallback
    raise RuntimeError(f"Не найден результат: {Path(raw_path).name}")


def _identify_denoise_outputs(paths: list[Path]) -> tuple[Path, Path]:
    clean = next(
        (
            path
            for path in paths
            if any(
                word in path.name.lower()
                for word in ("dry", "clean", "no noise", "no_noise")
            )
        ),
        None,
    )
    noise = next(
        (
            path
            for path in paths
            if any(word in path.name.lower() for word in ("noise", "other"))
            and "no noise" not in path.name.lower()
            and "no_noise" not in path.name.lower()
            and path != clean
        ),
        None,
    )
    if clean is None and noise is not None:
        clean = next(path for path in paths if path != noise)
    if noise is None and clean is not None:
        noise = next(path for path in paths if path != clean)
    return (clean or paths[0], noise or paths[1])


if __name__ == "__main__":
    main()
