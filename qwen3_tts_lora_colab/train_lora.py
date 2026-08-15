#!/usr/bin/env python3
"""LoRA trainer for Qwen3-TTS.

The LoRA training core is adapted from Alexandria (MIT, 2026 Finrandojin)
and follows Qwen3-TTS's official teacher-forcing layout. This version adds
T4-friendly dtype selection, compact tqdm progress, per-epoch persistent
checkpoints, and resume support for Google Colab.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-TTS LoRA training")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--project_dir", required=True)
    p.add_argument("--work_dir", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_audio_seconds", type=float, default=20.0)
    p.add_argument("--language", default="russian")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def pick_dtype(torch):
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    candidates = []
    if checkpoints_dir.exists():
        for p in checkpoints_dir.glob("epoch-*"):
            try:
                n = int(p.name.split("-")[-1])
            except ValueError:
                continue
            if (p / "adapter_config.json").exists() and (p / "trainer_state.pt").exists():
                candidates.append((n, p))
    return max(candidates, default=(None, None))[1]


def load_dataset(data_dir, hf_model, processor, device, dtype, max_audio_seconds):
    import librosa
    import numpy as np
    import torch
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    data_dir = Path(data_dir)
    metadata_path = data_dir / "metadata.jsonl"
    entries = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not entries:
        raise RuntimeError("metadata.jsonl пуст.")

    ref_path = data_dir / "ref.wav"
    if not ref_path.exists():
        ref_rel = entries[0].get("ref_audio", "")
        ref_path = data_dir / ref_rel
    if not ref_path.exists():
        raise RuntimeError(f"Не найден reference: {ref_path}")

    print(f"Датасет: {len(entries)} фрагментов. Reference: {ref_path.name}", flush=True)
    ref_audio, _ = librosa.load(ref_path, sr=24000, mono=True)
    ref_audio = ref_audio.astype(np.float32)
    with torch.no_grad():
        ref_mels = mel_spectrogram(
            torch.from_numpy(ref_audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2).to(device=device, dtype=dtype)
        spk_embedding = hf_model.speaker_encoder(ref_mels).detach()

    samples = []
    for entry in entries:
        audio_rel = entry.get("audio_filepath") or entry.get("audio")
        audio_path = data_dir / audio_rel
        if not audio_path.exists():
            continue
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = len(audio) / sr
        if duration > max_audio_seconds:
            continue
        with torch.no_grad():
            codes = hf_model.speech_tokenizer.encode(audio, sr=sr).audio_codes[0]
        text = entry["text"].strip()
        assistant_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        text_ids = processor(text=assistant_text, return_tensors="pt", padding=True)["input_ids"].to(device)
        if text_ids.dim() == 1:
            text_ids = text_ids.unsqueeze(0)
        samples.append({"codec_ids": codes.to(device), "spk_embedding": spk_embedding, "text_ids": text_ids, "text": text})

    if not samples:
        raise RuntimeError("После проверки длительности не осталось обучающих фрагментов.")
    return samples, ref_path


def build_teacher_forcing_input(sample, hf_model, device, language="russian"):
    import torch

    talker = hf_model.talker
    config = hf_model.config
    tc = config.talker_config
    codec_ids = sample["codec_ids"]
    spk = sample["spk_embedding"]
    text_ids = sample["text_ids"]
    total_audio_steps = codec_ids.shape[0]
    num_groups = tc.num_code_groups

    special_ids = torch.tensor(
        [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
        device=device,
        dtype=text_ids.dtype,
    )
    tts_bos, tts_eos, tts_pad = talker.text_projection(talker.get_text_embeddings()(special_ids)).chunk(3, dim=1)
    role = talker.text_projection(talker.get_text_embeddings()(text_ids[:, :3]))

    language_id = tc.codec_language_id.get(language) if getattr(tc, "codec_language_id", None) else None
    if language_id is not None:
        codec_prefix_ids = [[tc.codec_think_id, tc.codec_think_bos_id, language_id, tc.codec_think_eos_id]]
    else:
        codec_prefix_ids = [[tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id]]
    codec_prefix = talker.get_input_embeddings()(torch.tensor(codec_prefix_ids, device=device, dtype=text_ids.dtype))
    codec_suffix = talker.get_input_embeddings()(torch.tensor([[tc.codec_pad_id, tc.codec_bos_id]], device=device, dtype=text_ids.dtype))
    codec_embed = torch.cat([codec_prefix, spk.view(1, 1, -1), codec_suffix], dim=1)
    prefix_len = codec_embed.shape[1]
    text_side_prefix = torch.cat([tts_pad.expand(-1, prefix_len - 2, -1), tts_bos], dim=1)
    prefix = text_side_prefix + codec_embed[:, :-1]

    content_ids = text_ids[:, 3:-5]
    content = talker.text_projection(talker.get_text_embeddings()(content_ids))
    text_with_eos = torch.cat([content, tts_eos], dim=1)
    pad_ids = torch.full((1, text_with_eos.shape[1]), tc.codec_pad_id, device=device, dtype=text_ids.dtype)
    text_portion = text_with_eos + talker.get_input_embeddings()(pad_ids)
    codec_bos = talker.get_input_embeddings()(torch.tensor([[tc.codec_bos_id]], device=device, dtype=text_ids.dtype))
    prefill = torch.cat([role, prefix, text_portion, tts_pad + codec_bos], dim=1)
    prefill_len = prefill.shape[1]

    group_embeds = [talker.get_input_embeddings()(codec_ids[:, :1])]
    for group in range(1, num_groups):
        group_embeds.append(talker.code_predictor.get_input_embeddings()[group - 1](codec_ids[:, group:group + 1]))
    audio_embeds = torch.cat(group_embeds, dim=1).sum(dim=1) + tts_pad.squeeze(0)
    full_input = torch.cat([prefill, audio_embeds.unsqueeze(0)], dim=1)

    labels = torch.full((1, prefill_len + total_audio_steps), -100, device=device, dtype=torch.long)
    labels[0, prefill_len:] = codec_ids[:, 0]
    return full_input, labels, codec_ids, prefill_len


def save_checkpoint(peft_talker, optimizer, epoch, best_loss, local_root: Path, persistent_root: Path) -> None:
    import torch

    local_dir = local_root / f"epoch-{epoch}"
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    peft_talker.save_pretrained(local_dir)
    torch.save({"epoch": epoch, "best_loss": best_loss, "optimizer": optimizer.state_dict()}, local_dir / "trainer_state.pt")

    persistent_dir = persistent_root / f"epoch-{epoch}"
    if persistent_dir.exists():
        shutil.rmtree(persistent_dir)
    shutil.copytree(local_dir, persistent_dir)
    print(f"Checkpoint эпохи {epoch} сохранён на Google Drive.", flush=True)


def train(args):
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, get_peft_model
    from qwen_tts import Qwen3TTSModel
    from tqdm.auto import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype(torch)
    print(f"Устройство: {device}. Тип вычислений: {str(dtype).replace('torch.', '')}.", flush=True)
    print(f"Загружаю {args.model_name}...", flush=True)

    model = Qwen3TTSModel.from_pretrained(
        args.model_name,
        device_map="cuda:0" if device == "cuda" else None,
        dtype=dtype,
        attn_implementation="eager",
    )
    hf_model = model.model
    processor = model.processor
    samples, ref_path = load_dataset(args.data_dir, hf_model, processor, device, dtype, args.max_audio_seconds)

    project_dir = Path(args.project_dir)
    persistent_checkpoints = project_dir / "checkpoints"
    persistent_adapter = project_dir / "adapter"
    local_checkpoints = Path(args.work_dir) / "checkpoints"
    persistent_checkpoints.mkdir(parents=True, exist_ok=True)
    persistent_adapter.mkdir(parents=True, exist_ok=True)
    local_checkpoints.mkdir(parents=True, exist_ok=True)

    resume_path = latest_checkpoint(persistent_checkpoints) if args.resume else None
    if resume_path:
        print(f"Продолжаю обучение с {resume_path.name}.", flush=True)
        peft_talker = PeftModel.from_pretrained(hf_model.talker, str(resume_path), is_trainable=True)
        state = torch.load(resume_path / "trainer_state.pt", map_location="cpu", weights_only=False)
        start_epoch = int(state["epoch"]) + 1
        best_loss = float(state.get("best_loss", "inf"))
    else:
        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        peft_talker = get_peft_model(hf_model.talker, lora)
        state = None
        start_epoch = 1
        best_loss = float("inf")

    hf_model.talker = peft_talker
    peft_talker.enable_input_require_grads()
    try:
        peft_talker.base_model.model.model.gradient_checkpointing_enable()
    except AttributeError:
        pass

    trainable = [p for p in peft_talker.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])

    use_scaler = device == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    base_talker = peft_talker.base_model.model
    transformer = base_talker.model
    peft_talker.train()

    if start_epoch > args.epochs:
        print("Запрошенное число эпох уже выполнено. Увеличьте число эпох, чтобы продолжить.", flush=True)
        return

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        random.shuffle(samples)
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        valid_steps = 0
        pbar = tqdm(
            enumerate(samples, 1),
            total=len(samples),
            desc=f"Эпоха {epoch}/{args.epochs}",
            unit="шаг",
            dynamic_ncols=False,
            ncols=100,
            leave=False,
            file=sys.stdout,
        )
        for step_idx, sample in pbar:
            try:
                full_input, labels, codec_ids, prefill_len = build_teacher_forcing_input(sample, hf_model, device, args.language)
                total_audio_steps = codec_ids.shape[0]
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=device == "cuda"):
                    output = transformer(inputs_embeds=full_input, use_cache=False)
                    hidden = output.last_hidden_state
                    logits = base_talker.codec_head(hidden)
                    talker_loss = F.cross_entropy(
                        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                        labels[:, 1:].contiguous().view(-1),
                        ignore_index=-100,
                    )
                    audio_hidden = hidden[0, prefill_len - 1:prefill_len + total_audio_steps - 1, :]
                    _, sub_loss = base_talker.forward_sub_talker_finetune(codec_ids, audio_hidden)
                    loss = talker_loss + 0.3 * sub_loss
                    scaled = loss / args.gradient_accumulation_steps

                if use_scaler:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

                step_loss = float(loss.detach().float().item())
                epoch_loss += step_loss
                valid_steps += 1
                pbar.set_postfix_str(f"loss {step_loss:.4f}", refresh=True)

                if step_idx % args.gradient_accumulation_steps == 0 or step_idx == len(samples):
                    if use_scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if device == "cuda":
                        torch.cuda.empty_cache()

                del full_input, labels, codec_ids, output, hidden, logits, audio_hidden, talker_loss, sub_loss, loss, scaled
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    pbar.write(f"Пропущен шаг {step_idx}: не хватило VRAM.")
                    optimizer.zero_grad(set_to_none=True)
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise
        pbar.close()

        avg_loss = epoch_loss / max(valid_steps, 1)
        best_loss = min(best_loss, avg_loss)
        elapsed = int(time.time() - started)
        print(f"Эпоха {epoch}/{args.epochs} завершена: loss {avg_loss:.4f}, время {elapsed} с.", flush=True)
        save_checkpoint(peft_talker, optimizer, epoch, best_loss, local_checkpoints, persistent_checkpoints)

        if persistent_adapter.exists():
            for child in persistent_adapter.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        peft_talker.save_pretrained(persistent_adapter)
        shutil.copy2(ref_path, persistent_adapter / "ref_sample.wav")
        ref_text = Path(args.data_dir, "ref_text.txt")
        if ref_text.exists():
            shutil.copy2(ref_text, persistent_adapter / "ref_text.txt")
        (persistent_adapter / "training_meta.json").write_text(
            json.dumps({
                "model_name": args.model_name,
                "epoch": epoch,
                "epochs_requested": args.epochs,
                "loss": avg_loss,
                "best_loss": best_loss,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lr": args.lr,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "samples": len(samples),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"Обучение завершено. Готовый LoRA: {persistent_adapter}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    try:
        train(args)
    except KeyboardInterrupt:
        print("Обучение остановлено пользователем.", flush=True)
        raise SystemExit(130)
