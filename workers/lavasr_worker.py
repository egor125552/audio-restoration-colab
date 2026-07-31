from __future__ import annotations

from common import parse_worker_args, report_progress, write_manifest


def main() -> None:
    arguments, settings = parse_worker_args(
        "Быстрая дорисовка речи с помощью LavaSR.",
        {"lavasr_small"},
    )
    report_progress(0.03, "LavaSR: загружаю библиотеки…")
    import soundfile as sf
    import torch
    from LavaSR.model import LavaEnhance2

    device = _device()
    if device == "cuda":
        print(
            f"[audio-restoration] LavaSR GPU: {torch.cuda.get_device_name(0)}",
            flush=True,
        )
    else:
        print(f"[audio-restoration] LavaSR устройство: {device}", flush=True)

    report_progress(0.08, f"LavaSR: загружаю модель на {device}…")
    model = LavaEnhance2("YatharthS/LavaSR", device)
    if device == "cuda":
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        print(
            f"[audio-restoration] LavaSR VRAM после загрузки: {allocated:.2f} ГБ",
            flush=True,
        )

    report_progress(0.20, "LavaSR: читаю аудио…")
    input_rate = settings.get("input_rate", "auto")
    if input_rate == "auto":
        audio, _ = model.load_audio(str(arguments.input))
    else:
        audio, _ = model.load_audio(
            str(arguments.input),
            input_sr=int(input_rate),
        )
    report_progress(
        0.28,
        "LavaSR: дорисовываю частоты —",
        tqdm_span=0.62,
    )
    result = model.enhance(
        audio,
        denoise=bool(settings.get("denoise", False)),
        batch=bool(settings.get("batch", True)),
    )
    report_progress(0.94, "LavaSR: сохраняю результат…")
    target = arguments.output_dir / "restored.wav"
    sf.write(
        str(target),
        result.detach().float().cpu().numpy().squeeze(),
        48_000,
    )
    write_manifest(arguments.output_dir, [("restored", target)])
    report_progress(1.0, "LavaSR: готово.")


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    main()
