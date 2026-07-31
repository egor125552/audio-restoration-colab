from __future__ import annotations

import tempfile
from pathlib import Path

from common import parse_worker_args, report_progress, write_manifest

TARGET_RATE = 48_000
CHUNK_SECONDS = 5.12
OVERLAP_SECONDS = 0.5


def main() -> None:
    arguments, settings = parse_worker_args(
        "Тяжёлая диффузионная дорисовка с помощью AudioSR.",
        {"audiosr_large"},
    )
    report_progress(0.01, "AudioSR: загружаю библиотеки…")
    import audiosr.pipeline as audiosr_pipeline
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from audiosr import build_model

    original_lowpass = audiosr_pipeline.lowpass_filtering_prepare_inference

    def safe_lowpass(batch):
        try:
            return original_lowpass(batch)
        except ValueError as error:
            if "critical frequencies" not in str(error):
                raise
            return {"waveform_lowpass": batch["waveform"].clone()}

    audiosr_pipeline.lowpass_filtering_prepare_inference = safe_lowpass

    report_progress(0.04, "AudioSR: читаю и подготавливаю аудио…")
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
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(
            f"[audio-restoration] AudioSR GPU: {gpu_name}; VRAM всего: "
            f"{total_vram:.1f} ГБ",
            flush=True,
        )
    else:
        print("[audio-restoration] AudioSR: CUDA недоступна, использую CPU", flush=True)

    report_progress(0.07, f"AudioSR: загружаю модель на {device}…")
    model = build_model(
        model_name=str(settings.get("mode", "basic")),
        device=device,
    )
    if device == "cuda":
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(
            f"[audio-restoration] AudioSR VRAM после загрузки модели: "
            f"занято {allocated:.2f} ГБ, зарезервировано {reserved:.2f} ГБ",
            flush=True,
        )
    report_progress(0.18, "AudioSR: модель загружена, начинаю дорисовку…")

    chunk_count = _chunk_count(audio.shape[-1])
    total_passes = max(1, len(audio) * chunk_count)
    channels = []
    for channel_index, channel in enumerate(audio):
        channels.append(
            _process_channel(
                model=model,
                channel=channel,
                steps=int(settings.get("steps", 50)),
                guidance=float(settings.get("guidance", 3.5)),
                seed=int(settings.get("seed", 42)) + channel_index,
                pass_offset=channel_index * chunk_count,
                total_passes=total_passes,
                np=np,
                sf=sf,
            )
        )

    report_progress(0.94, "AudioSR: объединяю обработанные фрагменты…")
    shortest = min(len(channel) for channel in channels)
    result = np.column_stack([channel[:shortest] for channel in channels])
    target = arguments.output_dir / "restored.wav"
    report_progress(0.98, "AudioSR: сохраняю результат…")
    sf.write(str(target), result, TARGET_RATE)
    write_manifest(arguments.output_dir, [("restored", target)])
    report_progress(1.0, "AudioSR: готово.")


def _process_channel(
    *,
    model,
    channel,
    steps: int,
    guidance: float,
    seed: int,
    pass_offset: int,
    total_passes: int,
    np,
    sf,
):
    from audiosr import super_resolution

    chunk_size = int(CHUNK_SECONDS * TARGET_RATE)
    overlap = int(OVERLAP_SECONDS * TARGET_RATE)
    hop = chunk_size - overlap
    starts = list(range(0, len(channel), hop))
    assembled = None
    with tempfile.TemporaryDirectory(prefix="audiosr-") as directory:
        temp_root = Path(directory)
        for chunk_index, start in enumerate(starts):
            raw = channel[start : start + chunk_size]
            chunk_path = temp_root / f"chunk-{chunk_index}.wav"
            sf.write(str(chunk_path), raw, TARGET_RATE)
            pass_index = pass_offset + chunk_index
            span = 0.74 / total_passes
            base = 0.18 + span * pass_index
            report_progress(
                base,
                f"AudioSR: фрагмент {pass_index + 1} из {total_passes} —",
                tqdm_span=span,
            )
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


def _chunk_count(sample_count: int) -> int:
    chunk_size = int(CHUNK_SECONDS * TARGET_RATE)
    overlap = int(OVERLAP_SECONDS * TARGET_RATE)
    hop = chunk_size - overlap
    return max(1, len(range(0, sample_count, hop)))


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
