from __future__ import annotations

from common import parse_worker_args, write_manifest


def main() -> None:
    arguments, settings = parse_worker_args(
        "Быстрая дорисовка речи с помощью LavaSR.",
        {"lavasr_small"},
    )
    import soundfile as sf
    from LavaSR.model import LavaEnhance2

    device = _device()
    model = LavaEnhance2("YatharthS/LavaSR", device)
    input_rate = settings.get("input_rate", "auto")
    if input_rate == "auto":
        audio, _ = model.load_audio(str(arguments.input))
    else:
        audio, _ = model.load_audio(
            str(arguments.input),
            input_sr=int(input_rate),
        )
    result = model.enhance(
        audio,
        denoise=bool(settings.get("denoise", False)),
        batch=bool(settings.get("batch", True)),
    )
    target = arguments.output_dir / "restored.wav"
    sf.write(
        str(target),
        result.detach().float().cpu().numpy().squeeze(),
        48_000,
    )
    write_manifest(arguments.output_dir, [("restored", target)])


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    main()
