from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Проверка ускорения AudioSR через Torch-TensorRT на NVIDIA T4."
    )
    parser.add_argument("--input", required=True, help="Путь к короткому WAV/MP3/M4A")
    parser.add_argument("--output-dir", default="/tmp/audiosr-t4-probe")
    parser.add_argument("--mode", choices=("basic", "speech"), default="basic")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Первый запуск включает компиляцию, второй показывает установившуюся скорость.",
    )
    args = parser.parse_args()

    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        parser.error(f"Файл не найден: {source}")
    if args.steps < 1:
        parser.error("--steps должен быть больше нуля")
    if args.runs < 1:
        parser.error("--runs должен быть больше нуля")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import soundfile as sf
    import torch
    import torch_tensorrt  # noqa: F401
    from audiosr import build_model, super_resolution

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
    module_name, diffusion_module = _find_diffusion_module(model)
    param_count = sum(parameter.numel() for parameter in diffusion_module.parameters())
    print(
        f"Найден тяжёлый блок: {module_name}, {param_count / 1_000_000:.1f} млн параметров",
        flush=True,
    )

    print("Подключаю torch.compile backend=\"torch_tensorrt\"…", flush=True)
    compiled_diffusion = torch.compile(
        diffusion_module,
        backend="torch_tensorrt",
        dynamic=False,
        options={
            "precision": torch.float16,
            "min_block_size": 3,
            "optimization_level": 4,
            "truncate_long_and_double": True,
        },
    )
    _replace_module(model, module_name, compiled_diffusion)

    print(
        "Первый запуск может быть заметно медленнее: TensorRT профилирует и собирает engine.",
        flush=True,
    )

    timings: list[float] = []
    for run_index in range(args.runs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            generated = super_resolution(
                model,
                str(source),
                seed=args.seed,
                ddim_steps=args.steps,
                guidance_scale=args.guidance,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        timings.append(elapsed)

        audio = np.asarray(generated).squeeze().astype(np.float32)
        target = output_dir / f"tensorrt-run-{run_index + 1}.wav"
        sf.write(str(target), audio, 48_000)
        peak = torch.cuda.max_memory_allocated(0) / (1024**3)
        label = "компиляция + инференс" if run_index == 0 else "готовый engine"
        print(
            f"Запуск {run_index + 1}: {elapsed:.2f} с ({label}), peak VRAM {peak:.2f} ГБ",
            flush=True,
        )
        print(f"Результат: {target}", flush=True)

    if len(timings) >= 2:
        speedup = timings[0] / timings[-1] if timings[-1] else float("inf")
        print(
            f"Первый/последний запуск: {timings[0]:.2f} / {timings[-1]:.2f} с. "
            f"Отношение: {speedup:.2f}x.",
            flush=True,
        )
        print(
            "Для честного сравнения с обычной AudioSR сравни второй запуск с тем же файлом, "
            "seed, guidance и количеством шагов.",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001
        print(f"T4 TensorRT probe failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise
