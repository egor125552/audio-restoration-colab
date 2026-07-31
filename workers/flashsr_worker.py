from __future__ import annotations

import sys

from common import model_cache_root, parse_worker_args, report_progress, write_manifest


def main() -> None:
    arguments, settings = parse_worker_args(
        "Однопроходная дорисовка аудио с помощью FlashSR.",
        {"flashsr_medium"},
    )
    report_progress(0.03, "FlashSR: загружаю библиотеки…")
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    repository = model_cache_root() / "repos" / "flashsr"
    sys.path.insert(0, str(repository))
    report_progress(0.06, "FlashSR: загружаю код модели…")
    from enhance import build_model, enhance

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(
            f"[audio-restoration] FlashSR GPU: {torch.cuda.get_device_name(0)}",
            flush=True,
        )
    else:
        print("[audio-restoration] FlashSR: CUDA недоступна, использую CPU", flush=True)

    report_progress(0.10, f"FlashSR: загружаю модель на {device.type}…")
    original_torch_load = torch.load

    def portable_torch_load(*args, **kwargs):
        kwargs.setdefault("map_location", device)
        return original_torch_load(*args, **kwargs)

    torch.load = portable_torch_load
    try:
        model = build_model(repository / "weights", device)
    finally:
        torch.load = original_torch_load
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        print(
            f"[audio-restoration] FlashSR VRAM после загрузки: {allocated:.2f} ГБ",
            flush=True,
        )

    report_progress(0.22, "FlashSR: читаю аудио…")
    audio, sample_rate = sf.read(
        str(arguments.input),
        dtype="float32",
        always_2d=True,
    )
    channels: list[np.ndarray] = []
    channel_count = audio.shape[1]
    span = 0.68 / max(1, channel_count)
    for channel_index in range(channel_count):
        channel = audio[:, channel_index]
        if sample_rate != 48_000:
            channel = resample_poly(
                channel,
                48_000,
                sample_rate,
            ).astype(np.float32)
        if bool(settings.get("lowpass", True)):
            channel = resample_poly(
                resample_poly(channel, 2, 3),
                3,
                2,
            )[: len(channel)].astype(np.float32)
        base = 0.24 + span * channel_index
        report_progress(
            base,
            f"FlashSR: канал {channel_index + 1} из {channel_count} —",
            tqdm_span=span,
        )
        restored = enhance(
            model,
            channel,
            device=device,
            lowpass=False,
        )
        channels.append(restored)
    report_progress(0.94, "FlashSR: сохраняю результат…")
    shortest = min(len(channel) for channel in channels)
    stereo_safe = np.column_stack(
        [channel[:shortest] for channel in channels]
    )
    target = arguments.output_dir / "restored.wav"
    sf.write(str(target), stereo_safe, 48_000)
    write_manifest(arguments.output_dir, [("restored", target)])
    report_progress(1.0, "FlashSR: готово.")


if __name__ == "__main__":
    main()
