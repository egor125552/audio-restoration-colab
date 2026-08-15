from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

_MODEL_CACHE: dict[str, Any] = {"key": None, "model": None}


def _pick_dtype(torch):
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _release_cached_model() -> None:
    model = _MODEL_CACHE.get("model")
    if model is not None:
        del model
    _MODEL_CACHE["model"] = None
    _MODEL_CACHE["key"] = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def base_model_for_project(project_dir: str | Path, fallback: str) -> str:
    meta = Path(project_dir) / "adapter" / "training_meta.json"
    if meta.exists():
        try:
            value = json.loads(meta.read_text(encoding="utf-8")).get("model_name")
            if value:
                return str(value)
        except Exception:
            pass
    return fallback


def load_model(base_model: str, adapter_path: str | Path, attention_implementation: str = "eager"):
    import torch
    from peft import PeftModel
    from qwen_tts import Qwen3TTSModel

    adapter_path = str(Path(adapter_path))
    key = (base_model, adapter_path, attention_implementation)
    if _MODEL_CACHE.get("key") == key and _MODEL_CACHE.get("model") is not None:
        return _MODEL_CACHE["model"]

    _release_cached_model()
    dtype = _pick_dtype(torch)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Инференс: загружаю {base_model}, adapter {adapter_path}, "
        f"device {device}, dtype {str(dtype).replace('torch.', '')}, attention {attention_implementation}.",
        flush=True,
    )
    model = Qwen3TTSModel.from_pretrained(
        base_model,
        device_map="cuda:0" if device == "cuda" else None,
        dtype=dtype,
        attn_implementation=attention_implementation,
    )
    model.model.talker = PeftModel.from_pretrained(model.model.talker, adapter_path, is_trainable=False)
    model.model.talker.eval()
    _MODEL_CACHE["key"] = key
    _MODEL_CACHE["model"] = model
    return model


def synthesize(
    *,
    project_dir: str | Path,
    base_model: str,
    adapter_path: str | Path,
    checkpoint_label: str,
    text: str,
    language: str,
    ref_audio: str,
    ref_text: str,
    do_sample: bool = True,
    top_k: int = 50,
    top_p: float = 1.0,
    temperature: float = 0.9,
    repetition_penalty: float = 1.05,
    subtalker_dosample: bool = True,
    subtalker_top_k: int = 50,
    subtalker_top_p: float = 1.0,
    subtalker_temperature: float = 0.9,
    max_new_tokens: int = 2048,
    x_vector_only_mode: bool = False,
    non_streaming_mode: bool = True,
    attention_implementation: str = "eager",
) -> tuple[str, str]:
    import soundfile as sf

    text = (text or "").strip()
    ref_text = (ref_text or "").strip()
    if not text:
        raise ValueError("Введите текст для озвучивания.")
    if not ref_audio:
        raise ValueError("Не выбран референсный аудиофайл.")
    if not x_vector_only_mode and not ref_text:
        raise ValueError("В ICL-режиме нужен точный текст референсного аудио.")

    model = load_model(base_model, adapter_path, attention_implementation)
    generation_kwargs: dict[str, Any] = {
        "do_sample": bool(do_sample),
        "repetition_penalty": float(repetition_penalty),
        "max_new_tokens": int(max_new_tokens),
    }
    if do_sample:
        generation_kwargs.update(
            top_k=int(top_k),
            top_p=float(top_p),
            temperature=float(temperature),
        )
    generation_kwargs["subtalker_dosample"] = bool(subtalker_dosample)
    if subtalker_dosample:
        generation_kwargs.update(
            subtalker_top_k=int(subtalker_top_k),
            subtalker_top_p=float(subtalker_top_p),
            subtalker_temperature=float(subtalker_temperature),
        )

    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language or "Russian",
        ref_audio=str(ref_audio),
        ref_text=None if x_vector_only_mode else ref_text,
        x_vector_only_mode=bool(x_vector_only_mode),
        non_streaming_mode=bool(non_streaming_mode),
        **generation_kwargs,
    )
    if not wavs:
        raise RuntimeError("Qwen3-TTS не вернул аудио.")

    output_dir = Path(project_dir) / "generations"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    wav_path = output_dir / f"generation-{stamp}.wav"
    sf.write(wav_path, wavs[0], sr)

    metadata = {
        "checkpoint": checkpoint_label,
        "adapter_path": str(adapter_path),
        "base_model": base_model,
        "reference_audio": str(ref_audio),
        "reference_text": None if x_vector_only_mode else ref_text,
        "text": text,
        "language": language,
        "sample_rate": sr,
        "do_sample": bool(do_sample),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "temperature": float(temperature),
        "repetition_penalty": float(repetition_penalty),
        "subtalker_dosample": bool(subtalker_dosample),
        "subtalker_top_k": int(subtalker_top_k),
        "subtalker_top_p": float(subtalker_top_p),
        "subtalker_temperature": float(subtalker_temperature),
        "max_new_tokens": int(max_new_tokens),
        "x_vector_only_mode": bool(x_vector_only_mode),
        "non_streaming_mode": bool(non_streaming_mode),
        "attention_implementation": attention_implementation,
    }
    meta_path = wav_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(wav_path), f"Готово. {checkpoint_label}. Файл: {wav_path}"
