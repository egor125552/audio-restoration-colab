from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any

import soundfile as sf
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from modelscope import snapshot_download as modelscope_snapshot_download
from omegaconf import OmegaConf

XVC_HOME = Path(os.environ.get("XVC_HOME", "/content/x-vc")).resolve()
SOURCE_DIR = XVC_HOME / "src"
ASSETS_DIR = XVC_HOME / "assets"
CHECKPOINT_DIR = XVC_HOME / "ckpts"
TOKENIZER_DIR = ASSETS_DIR / "glm-4-voice-tokenizer"
SPEAKER_DIR = ASSETS_DIR / "speech_eres2net_sv_en_voxceleb_16k"
RUNTIME_CONFIG = XVC_HOME / "xvc-colab.yaml"

MODEL_REPO = "chenxie95/X-VC"
TOKENIZER_REPO = "zai-org/glm-4-voice-tokenizer"
SPEAKER_REPO = "iic/speech_eres2net_sv_en_voxceleb_16k"

LATENT_HOP_LENGTH = 1280
MAX_OFFLINE_CHUNK_SECONDS = 20.0
OFFLINE_OVERLAP_SECONDS = 0.64
MAX_REFERENCE_SECONDS = 20.0

_MODEL: Any | None = None
_CFG: Any | None = None
_DEVICE: torch.device | None = None
_LOCK = Lock()


def _put_source_on_path() -> None:
    source = str(SOURCE_DIR)
    if source not in sys.path:
        sys.path.insert(0, source)


def _require_source() -> None:
    if not (SOURCE_DIR / "bins" / "infer_utils.py").is_file():
        raise RuntimeError(
            f"X-VC source was not found at {SOURCE_DIR}. Run install_xvc.sh first."
        )


def _seconds_to_aligned_samples(seconds: float, sample_rate: int) -> int:
    samples = int(seconds * sample_rate)
    samples = max(LATENT_HOP_LENGTH, samples)
    return max(
        LATENT_HOP_LENGTH,
        (samples // LATENT_HOP_LENGTH) * LATENT_HOP_LENGTH,
    )


def _plan_offline_chunks(total_samples: int, sample_rate: int) -> list[tuple[int, int]]:
    """Split long X-VC input into overlapping, hop-aligned windows.

    GLM-4 Voice Tokenizer used by X-VC has a 30 s Whisper-style feature window.
    We deliberately stay below it and keep every boundary aligned to X-VC's
    latent hop so semantic and acoustic timelines remain compatible.
    """
    if total_samples <= 0:
        return []

    max_chunk = _seconds_to_aligned_samples(MAX_OFFLINE_CHUNK_SECONDS, sample_rate)
    overlap = _seconds_to_aligned_samples(OFFLINE_OVERLAP_SECONDS, sample_rate)
    overlap = min(overlap, max_chunk - LATENT_HOP_LENGTH)

    if total_samples <= max_chunk:
        return [(0, total_samples)]

    total_hops = math.ceil(total_samples / LATENT_HOP_LENGTH)
    max_chunk_hops = max_chunk // LATENT_HOP_LENGTH
    overlap_hops = overlap // LATENT_HOP_LENGTH
    step_hops = max_chunk_hops - overlap_hops

    count = math.ceil((total_hops - overlap_hops) / step_hops)
    segment_hops = math.ceil(
        (total_hops + (count - 1) * overlap_hops) / count
    )
    if segment_hops > max_chunk_hops:
        raise RuntimeError("Не удалось безопасно разбить длинную запись X-VC.")

    step = (segment_hops - overlap_hops) * LATENT_HOP_LENGTH
    segment = segment_hops * LATENT_HOP_LENGTH
    ranges: list[tuple[int, int]] = []
    for index in range(count):
        start = index * step
        end = min(start + segment, total_samples)
        if start >= total_samples:
            break
        ranges.append((start, end))

    if not ranges or ranges[0][0] != 0 or ranges[-1][1] != total_samples:
        raise RuntimeError("Разбиение длинной записи X-VC получилось неполным.")
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] > previous[1]:
            raise RuntimeError("Между кусками X-VC образовался разрыв.")
        if current[1] - current[0] > max_chunk:
            raise RuntimeError("Кусок X-VC превысил безопасную длину.")
    return ranges


def _select_reference_range(
    target_wav: torch.Tensor,
    sample_rate: int,
) -> tuple[int, int, bool]:
    """Keep at most 20 s of the most active reference audio."""
    total = int(target_wav.shape[-1])
    maximum = _seconds_to_aligned_samples(MAX_REFERENCE_SECONDS, sample_rate)
    if total <= maximum:
        return 0, total, False

    hop = LATENT_HOP_LENGTH
    window_hops = maximum // hop
    with torch.no_grad():
        energy = target_wav.float().square().mean(dim=1, keepdim=True)
        hop_energy = F.avg_pool1d(energy, kernel_size=hop, stride=hop)
        if hop_energy.shape[-1] <= window_hops:
            start_hop = 0
        else:
            kernel = torch.ones(
                (1, 1, window_hops),
                device=hop_energy.device,
                dtype=hop_energy.dtype,
            )
            rolling = F.conv1d(hop_energy, kernel)
            start_hop = int(rolling.flatten().argmax().item())

    start = start_hop * hop
    end = min(start + maximum, total)
    if end - start < maximum:
        start = max(0, end - maximum)
    return start, end, True


def _match_audio_length(wav: torch.Tensor, wanted: int) -> torch.Tensor:
    current = int(wav.shape[-1])
    if current == wanted:
        return wav
    if current > wanted:
        return wav[..., :wanted]
    return F.pad(wav, (0, wanted - current))


def _overlap_add(
    assembled: torch.Tensor,
    new_chunk: torch.Tensor,
    overlap: int,
) -> torch.Tensor:
    overlap = min(overlap, assembled.shape[-1], new_chunk.shape[-1])
    if overlap <= 0:
        return torch.cat([assembled, new_chunk], dim=-1)

    phase = torch.linspace(
        0.0,
        1.0,
        overlap,
        device=new_chunk.device,
        dtype=new_chunk.dtype,
    )
    fade_in = 0.5 * (1.0 - torch.cos(torch.pi * phase))
    fade_out = 1.0 - fade_in
    mixed = assembled[..., -overlap:] * fade_out + new_chunk[..., :overlap] * fade_in
    return torch.cat(
        [assembled[..., :-overlap], mixed, new_chunk[..., overlap:]],
        dim=-1,
    )


def prepare_assets(progress=None) -> tuple[Path, Path]:
    """Download official X-VC assets and write a Colab-local config."""
    _require_source()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def note(value: float, text: str) -> None:
        if progress is not None:
            progress(value, desc=text)

    note(0.05, "Скачиваю checkpoint X-VC")
    checkpoint = Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename="xvc.pt",
            local_dir=str(CHECKPOINT_DIR),
        )
    )

    note(0.30, "Скачиваю GLM-4 Voice Tokenizer")
    snapshot_download(
        repo_id=TOKENIZER_REPO,
        local_dir=str(TOKENIZER_DIR),
    )

    note(0.58, "Скачиваю ERes2Net для голоса-референса")
    os.environ.setdefault("MODELSCOPE_DOMAIN", "www.modelscope.ai")
    modelscope_snapshot_download(
        SPEAKER_REPO,
        local_dir=str(SPEAKER_DIR),
    )

    note(0.75, "Готовлю конфигурацию")
    cfg = OmegaConf.load(SOURCE_DIR / "configs" / "xvc.yaml")
    cfg.model.generator.semantic_encoder.encoder.from_pretrained.hf_repo = TOKENIZER_REPO
    cfg.model.generator.semantic_encoder.encoder.from_pretrained.local_ckpt = str(TOKENIZER_DIR)
    cfg.model.generator.semantic_encoder.cfg.hf_repo = TOKENIZER_REPO
    cfg.model.generator.semantic_encoder.cfg.local_ckpt = str(TOKENIZER_DIR)
    cfg.model.generator.speaker_encoder.pretrained_dir = str(SPEAKER_DIR)
    OmegaConf.save(cfg, RUNTIME_CONFIG)

    if not checkpoint.is_file():
        raise RuntimeError(f"X-VC checkpoint was not downloaded: {checkpoint}")
    if not RUNTIME_CONFIG.is_file():
        raise RuntimeError(f"X-VC runtime config was not written: {RUNTIME_CONFIG}")

    note(0.90, "Файлы модели готовы")
    return checkpoint, RUNTIME_CONFIG


def load_model(progress=None) -> str:
    global _MODEL, _CFG, _DEVICE
    if _MODEL is not None:
        return f"X-VC уже загружена: {_DEVICE}"

    with _LOCK:
        if _MODEL is not None:
            return f"X-VC уже загружена: {_DEVICE}"

        checkpoint, config = prepare_assets(progress=progress)
        _put_source_on_path()
        from bins.infer_utils import load_xvc

        if progress is not None:
            progress(0.94, desc="Загружаю X-VC в память")
        _CFG, _MODEL, _DEVICE = load_xvc(
            str(config),
            str(checkpoint),
            0,
            False,
        )
        if progress is not None:
            progress(1.0, desc="X-VC готова")

    return f"X-VC загружена на {_DEVICE}"


def convert(
    source_path: str,
    reference_path: str,
    mode: str,
    current_ms: int,
    chunk_ms: int,
    future_ms: int,
    smooth_ms: int,
    progress=None,
) -> tuple[str, str]:
    if not source_path:
        raise ValueError("Добавьте исходную запись.")
    if not reference_path:
        raise ValueError("Добавьте голос-референс.")

    load_model(progress=progress)
    assert _MODEL is not None and _CFG is not None and _DEVICE is not None

    _put_source_on_path()
    from bins.infer_utils import (
        load_pair_as_tensors,
        precompute_conditions,
        run_offline,
        run_stream_chunk_forward,
        run_streaming,
        to_numpy_audio,
    )

    if progress is not None:
        progress(0.05, desc="Читаю исходную запись и референс")

    source_wav, target_wav, target_wav_cond = load_pair_as_tensors(
        source_wav_path=source_path,
        target_wav_path=reference_path,
        cfg=_CFG,
        device=_DEVICE,
        latent_hop_length=LATENT_HOP_LENGTH,
        mask_target_condition=False,
    )

    sample_rate = int(_CFG["sample_rate"])
    ref_start, ref_end, reference_trimmed = _select_reference_range(
        target_wav,
        sample_rate,
    )
    target_wav = target_wav[..., ref_start:ref_end]
    target_wav_cond = target_wav_cond[..., ref_start:ref_end]

    if mode == "Обычный":
        ranges = _plan_offline_chunks(int(source_wav.shape[-1]), sample_rate)
        if len(ranges) == 1:
            if progress is not None:
                progress(0.20, desc="Преобразую голос")
            try:
                recon = run_offline(_MODEL, source_wav, target_wav, target_wav_cond)
            except RuntimeError as exc:
                if "Sizes of tensors must match" in str(exc):
                    raise RuntimeError(
                        "X-VC не смогла совместить semantic и acoustic дорожки. "
                        "Попробуйте обновлённый блокнот: длинные записи в нём "
                        "автоматически делятся на безопасные части."
                    ) from exc
                raise
            mode_text = "обычный режим"
        else:
            if progress is not None:
                progress(0.18, desc=f"Длинная запись: делю на {len(ranges)} части")
            speaker_condition, frame_condition = precompute_conditions(
                _MODEL,
                target_wav,
                target_wav_cond,
            )
            assembled: torch.Tensor | None = None
            covered_end = 0
            for index, (start, end) in enumerate(ranges, start=1):
                if progress is not None:
                    progress(
                        0.20 + 0.68 * (index - 1) / len(ranges),
                        desc=f"Преобразую часть {index} из {len(ranges)}",
                    )
                chunk = source_wav[..., start:end]
                converted = run_stream_chunk_forward(
                    _MODEL,
                    chunk,
                    speaker_condition,
                    frame_condition,
                )
                converted = _match_audio_length(converted, end - start)
                if assembled is None:
                    assembled = converted
                    covered_end = end
                else:
                    overlap = max(0, covered_end - start)
                    assembled = _overlap_add(assembled, converted, overlap)
                    covered_end = max(covered_end, end)

            if assembled is None:
                raise RuntimeError("X-VC не создала ни одного куска результата.")
            recon = _match_audio_length(assembled, int(source_wav.shape[-1]))
            mode_text = f"обычный режим, длинная запись собрана из {len(ranges)} частей"
    else:
        if current_ms <= 0:
            raise ValueError("Для потокового режима current должен быть больше 0 мс.")
        if chunk_ms - current_ms - smooth_ms - future_ms < 0:
            raise ValueError(
                "Окно потокового режима некорректно: chunk должно быть не меньше "
                "current + future + smooth."
            )
        if chunk_ms >= 30000:
            raise ValueError("Окно X-VC должно быть короче 30 секунд.")
        if progress is not None:
            progress(0.20, desc="Подготавливаю голос-референс")
        speaker_condition, frame_condition = precompute_conditions(
            _MODEL,
            target_wav,
            target_wav_cond,
        )
        if progress is not None:
            progress(0.35, desc="Преобразую голос по частям")
        recon, latencies = run_streaming(
            model=_MODEL,
            source_wav=source_wav,
            speaker_condition=speaker_condition,
            frame_condition=frame_condition,
            sample_rate=sample_rate,
            chunk_ms=int(chunk_ms),
            current_ms=int(current_ms),
            future_ms=int(future_ms),
            smooth_ms=int(smooth_ms),
        )
        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
        mode_text = f"потоковый режим, средняя задержка чанка {mean_latency:.0f} мс"

    if progress is not None:
        progress(0.92, desc="Сохраняю результат")

    output = tempfile.NamedTemporaryFile(
        prefix="xvc-",
        suffix=".wav",
        delete=False,
    )
    output.close()
    sf.write(
        output.name,
        to_numpy_audio(recon),
        samplerate=sample_rate,
    )

    if reference_trimmed:
        mode_text += f"; референс автоматически ограничен до {MAX_REFERENCE_SECONDS:.0f} секунд"

    if progress is not None:
        progress(1.0, desc="Готово")
    return output.name, f"Готово: {mode_text}."
