from __future__ import annotations

import sys

from common import model_cache_root, parse_worker_args, write_manifest


def main() -> None:
    arguments, settings = parse_worker_args(
        "Однопроходная дорисовка аудио с помощью FlashSR.",
        {"flashsr_medium"},
    )
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    repository = model_cache_root() / "repos" / "flashsr"
    sys.path.insert(0, str(repository))
    from enhance import build_model, enhance

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(repository / "weights", device)
    audio, sample_rate = sf.read(
        str(arguments.input),
        dtype="float32",
        always_2d=True,
    )
    channels: list[np.ndarray] = []
    for channel_index in range(audio.shape[1]):
        channel = audio[:, channel_index]
        if sample_rate != 48_000:
            channel = resample_poly(
                channel,
                48_000,
                sample_rate,
            ).astype(np.float32)
        restored = enhance(
            model,
            channel,
            device=device,
            lowpass=bool(settings.get("lowpass", True)),
        )
        channels.append(restored)
    shortest = min(len(channel) for channel in channels)
    stereo_safe = np.column_stack(
        [channel[:shortest] for channel in channels]
    )
    target = arguments.output_dir / "restored.wav"
    sf.write(str(target), stereo_safe, 48_000)
    write_manifest(arguments.output_dir, [("restored", target)])


if __name__ == "__main__":
    main()
