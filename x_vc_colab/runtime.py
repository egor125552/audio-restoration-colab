from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any

import soundfile as sf
import torch
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
        latent_hop_length=1280,
        mask_target_condition=False,
    )

    if mode == "Обычный":
        if progress is not None:
            progress(0.20, desc="Преобразую голос")
        recon = run_offline(_MODEL, source_wav, target_wav, target_wav_cond)
        mode_text = "обычный режим"
    else:
        if current_ms <= 0:
            raise ValueError("Для потокового режима current должен быть больше 0 мс.")
        if chunk_ms - current_ms - smooth_ms - future_ms < 0:
            raise ValueError(
                "Окно потокового режима некорректно: chunk должно быть не меньше "
                "current + future + smooth."
            )
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
            sample_rate=int(_CFG["sample_rate"]),
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
        samplerate=int(_CFG["sample_rate"]),
    )

    if progress is not None:
        progress(1.0, desc="Готово")
    return output.name, f"Готово: {mode_text}."
