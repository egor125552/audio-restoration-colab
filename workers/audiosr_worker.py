from __future__ import annotations

import tempfile
from pathlib import Path

from common import parse_worker_args, write_manifest


TARGET_RATE = 48_000
CHUNK_SECONDS = 5.12
OVERLAP_SECONDS = 0.5


def main() -> None:
    arguments, settings = parse_worker_args(
        "Тяжёлая диффузионная дорисовка с помощью AudioSR.",
        {"audiosr_large"},
    )
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from audiosr import build_model

    audio, _ = librosa.load(
        str(arguments.input),
        sr=TARGET_RATE,
        mono=False,
    )
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    if bool(settings.get("lowpass", True)):
        audio = np.stack([_clean_lowpass(channel, np) for channel in audio])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(
        model_name=str(settings.get("mode", "basic")),
        device=device,
    )
    channels = [
        _process_channel(
            model=model,
            channel=channel,
            steps=int(settings.get("steps", 50)),
            guidance=float(settings.get("guidance", 3.5)),
            seed=int(settings.get("seed", 42)) + index,
            np=np,
            sf=sf,
        )
        for index, channel in enumerate(audio)
    ]
    shortest = min(len(channel) for channel in channels)
    result = np.column_stack([channel[:shortest] for channel in channels])
    target = arguments.output_dir / "restored.wav"
    sf.write(str(target), result, TARGET_RATE)
    write_manifest(arguments.output_dir, [("restored", target)])


def _process_channel(
    *,
    model,
    channel,
    steps: int,
    guidance: float,
    seed: int,
    np,
    sf,
):
    from audiosr import super_resolution

    chunk_size = int(CHUNK_SECONDS * TARGET_RATE)
    overlap = int(OVERLAP_SECONDS * TARGET_RATE)
    hop = chunk_size - overlap
    assembled = None
    with tempfile.TemporaryDirectory(prefix="audiosr-") as directory:
        temp_root = Path(directory)
        for chunk_index, start in enumerate(range(0, len(channel), hop)):
            raw = channel[start : start + chunk_size]
            chunk_path = temp_root / f"chunk-{chunk_index}.wav"
            sf.write(str(chunk_path), raw, TARGET_RATE)
            generated = super_resolution(
                model,
                str(chunk_path),
                seed=seed + chunk_index,
                ddim_steps=steps,
                guidance_scale=guidance,
            )
            current = np.asarray(generated).squeeze().astype(np.float32)
            current = current[: len(raw)]
            if assembled is None:
                assembled = current
            else:
                actual_overlap = min(overlap, len(assembled), len(current))
                fade_in = np.linspace(
                    0.0,
                    1.0,
                    actual_overlap,
                    dtype=np.float32,
                )
                mixed = (
                    assembled[-actual_overlap:] * (1.0 - fade_in)
                    + current[:actual_overlap] * fade_in
                )
                assembled = np.concatenate(
                    [
                        assembled[:-actual_overlap],
                        mixed,
                        current[actual_overlap:],
                    ]
                )
    if assembled is None:
        raise RuntimeError("AudioSR получила пустой аудиофайл.")
    return np.clip(assembled, -1.0, 1.0)


def _clean_lowpass(channel, np):
    import librosa

    reduced = librosa.resample(
        channel,
        orig_sr=TARGET_RATE,
        target_sr=32_000,
    )
    restored = librosa.resample(
        reduced,
        orig_sr=32_000,
        target_sr=TARGET_RATE,
    )
    return np.asarray(restored[: len(channel)], dtype=np.float32)


if __name__ == "__main__":
    main()
