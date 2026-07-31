from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROBE_SECONDS = 5.12


def _find_diffusion_module(model):
    for name, module in model.named_modules():
        if name.endswith("diffusion_model"):
            return name, module
    raise RuntimeError("Не найден diffusion_model внутри AudioSR.")


def _replace_module(root, dotted_name: str, new_module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _prepare_probe_input(source: Path, output_dir: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg не найден в системе.")

    target = output_dir / "probe-input.wav"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-t",
        str(PROBE_SECONDS),
        "-ac",
        "1",
        "-ar",
        "48000",
        "-c:a",
        "pcm_s24le",
        str(target),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not target.is_file():
        details = completed.stderr.strip() or "ffmpeg не создал WAV."
        raise RuntimeError(f"Не удалось подготовить WAV для теста: {details}")
    print(
        f"Тестовый вход: первые {PROBE_SECONDS:.2f} с -> {target}",
        flush=True,
    )
    return target


def _install_lowpass_guard(audiosr_pipeline) -> None:
    original_lowpass = audiosr_pipeline.lowpass_filtering_prepare_inference

    def safe_lowpass(batch):
        try:
            return original_lowpass(batch)
        except ValueError as error:
            if "critical frequencies" not in str(error):
                raise
            print(
                "AudioSR lowpass не смог определить корректный срез; "
                "использую исходную waveform как безопасный fallback.",
                flush=True,
            )
            return {"waveform_lowpass": batch["waveform"].clone()}

    audiosr_pipeline.lowpass_filtering_prepare_inference = safe_lowpass


def _disable_training_checkpointing() -> None:
    import audiosr.latent_diffusion.modules.attention as attention
    import audiosr.latent_diffusion.modules.diffusionmodules.openaimodel as openaimodel

    def inference_checkpoint(func, inputs, _params, _flag):
        return func(*inputs)

    attention.checkpoint = inference_checkpoint
    openaimodel.checkpoint = inference_checkpoint
    print(
        "Inference-only: отключён training gradient checkpoint wrapper "
        "в UNet/attention для чистого графа TensorRT.",
        flush=True,
    )


def _run_once(*, model, source: Path, seed: int, steps: int, guidance: float, torch):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize()
    started = time.perf_counter()
    from audiosr import super_resolution

    with torch.inference_mode():
        generated = super_resolution(
            model,
            str(source),
            seed=seed,
            ddim_steps=steps,
            guidance_scale=guidance,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(0) / (1024**3)
    return generated, elapsed, peak


def _snr_db(reference, candidate, np) -> float:
    reference = np.asarray(reference).squeeze().astype(np.float64)
    candidate = np.asarray(candidate).squeeze().astype(np.float64)
    length = min(reference.size, candidate.size)
    if length == 0:
        return float("nan")
    reference = reference[:length]
    candidate = candidate[:length]
    signal_power = float(np.sum(reference * reference))
    error = reference - candidate
    error_power = float(np.sum(error * error))
    if error_power == 0.0:
        return float("inf")
    if signal_power == 0.0:
        return float("-inf")
    return 10.0 * math.log10(signal_power / error_power)


def _save_audio(value, target: Path, *, np, sf) -> None:
    audio = np.asarray(value).squeeze().astype(np.float32)
    sf.write(str(target), audio, 48_000)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Проверка ускорения AudioSR через Torch-TensorRT на NVIDIA T4."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Путь к WAV, MP3, M4A или другому аудиофайлу, который читает FFmpeg",
    )
    parser.add_argument("--output-dir", default="/tmp/audiosr-t4-probe")
    parser.add_argument("--mode", choices=("basic", "speech"), default="basic")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help=(
            "Число TensorRT-запусков. Первый включает компиляцию; "
            "последующие показывают установившуюся скорость."
        ),
    )
    args = parser.parse_args()

    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        parser.error(f"Файл не найден: {source}")
    if args.steps < 1:
        parser.error("--steps должен быть больше нуля")
    if args.runs < 2:
        parser.error("--runs должен быть не меньше 2")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_source = _prepare_probe_input(source, output_dir)

    import audiosr.pipeline as audiosr_pipeline
    import numpy as np
    import soundfile as sf
    import torch
    import torch_tensorrt
    from audiosr import build_model

    _install_lowpass_guard(audiosr_pipeline)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна. Нужен Colab с GPU.")

    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu_name}", flush=True)
    print(f"VRAM: {props.total_memory / (1024**3):.1f} ГБ", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"Torch-TensorRT: {torch_tensorrt.__version__}", flush=True)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    print(f"Загружаю AudioSR ({args.mode}) на CUDA…", flush=True)
    model = build_model(model_name=args.mode, device="cuda")
    model.eval()

    module_name, diffusion_module = _find_diffusion_module(model)
    param_count = sum(
        parameter.numel() for parameter in diffusion_module.parameters()
    )
    print(
        "Найден тяжёлый блок: "
        f"{module_name}, {param_count / 1_000_000:.1f} млн параметров",
        flush=True,
    )

    print("1/3: измеряю текущий PyTorch без изменений…", flush=True)
    baseline, baseline_time, baseline_peak = _run_once(
        model=model,
        source=probe_source,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
        torch=torch,
    )
    baseline_path = output_dir / "baseline-pytorch.wav"
    _save_audio(baseline, baseline_path, np=np, sf=sf)
    print(
        f"PyTorch current: {baseline_time:.2f} с, peak VRAM {baseline_peak:.2f} ГБ",
        flush=True,
    )

    _disable_training_checkpointing()
    print("2/3: измеряю PyTorch без training checkpoint wrapper…", flush=True)
    clean_baseline, clean_time, clean_peak = _run_once(
        model=model,
        source=probe_source,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
        torch=torch,
    )
    clean_path = output_dir / "baseline-pytorch-inference-clean.wav"
    _save_audio(clean_baseline, clean_path, np=np, sf=sf)
    checkpoint_speedup = baseline_time / clean_time if clean_time else float("inf")
    checkpoint_snr = _snr_db(baseline, clean_baseline, np)
    print(
        f"PyTorch clean: {clean_time:.2f} с, peak VRAM {clean_peak:.2f} ГБ, "
        f"ускорение {checkpoint_speedup:.2f}x, SNR {checkpoint_snr:.2f} dB",
        flush=True,
    )
    if checkpoint_snr < 60.0:
        raise RuntimeError(
            "Отключение training checkpoint wrapper изменило результат сильнее "
            "ожидаемого; TensorRT-тест остановлен."
        )

    print("3/3: подключаю torch.compile backend=\"torch_tensorrt\"…", flush=True)
    compiled_diffusion = torch.compile(
        diffusion_module,
        backend="torch_tensorrt",
        dynamic=False,
        options={
            "enabled_precisions": {torch.float32, torch.float16},
            "min_block_size": 3,
            "optimization_level": 4,
            "truncate_long_and_double": True,
            "use_python_runtime": False,
        },
    )
    _replace_module(model, module_name, compiled_diffusion)

    print(
        "Первый TensorRT-запуск может быть заметно медленнее: профилируется "
        "граф и собираются engine-блоки.",
        flush=True,
    )

    timings: list[float] = []
    final_generated = None
    for run_index in range(args.runs):
        generated, elapsed, peak = _run_once(
            model=model,
            source=probe_source,
            seed=args.seed,
            steps=args.steps,
            guidance=args.guidance,
            torch=torch,
        )
        final_generated = generated
        timings.append(elapsed)

        target = output_dir / f"tensorrt-run-{run_index + 1}.wav"
        _save_audio(generated, target, np=np, sf=sf)
        label = "компиляция + инференс" if run_index == 0 else "готовый engine"
        print(
            f"TensorRT {run_index + 1}: {elapsed:.2f} с ({label}), "
            f"peak VRAM {peak:.2f} ГБ",
            flush=True,
        )
        print(f"Результат: {target}", flush=True)

    steady_time = timings[-1]
    speedup_current = baseline_time / steady_time if steady_time else float("inf")
    speedup_clean = clean_time / steady_time if steady_time else float("inf")
    print(
        f"Итог: current PyTorch {baseline_time:.2f} с -> TensorRT "
        f"{steady_time:.2f} с = {speedup_current:.2f}x.",
        flush=True,
    )
    print(
        f"Чистый эффект TensorRT против inference-clean PyTorch: "
        f"{speedup_clean:.2f}x.",
        flush=True,
    )

    if final_generated is not None:
        quality_snr = _snr_db(baseline, final_generated, np)
        print(
            f"SNR TensorRT относительно текущего PyTorch: {quality_snr:.2f} dB",
            flush=True,
        )
        if quality_snr < 30.0:
            print(
                "Предупреждение: отличие от базового результата заметное; "
                "ускоренный режим пока не стоит встраивать в основной интерфейс.",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001
        print(
            f"T4 TensorRT probe failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise
