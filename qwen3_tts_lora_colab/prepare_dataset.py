from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from project import ProjectPaths, discover_audio


def _audio_segment_class():
    from pydub import AudioSegment
    return AudioSegment


def _split_on_silence():
    from pydub.silence import split_on_silence
    return split_on_silence


def segment_file(path: Path, out_dir: Path, prefix: str, min_ms: int = 2500, max_ms: int = 15000) -> list[Path]:
    AudioSegment = _audio_segment_class()
    split_on_silence = _split_on_silence()
    audio = AudioSegment.from_file(path).set_channels(1).set_frame_rate(24000)
    chunks = split_on_silence(
        audio,
        min_silence_len=450,
        silence_thresh=audio.dBFS - 16 if audio.dBFS != float("-inf") else -45,
        keep_silence=180,
    )
    if not chunks:
        chunks = [audio]

    normalized: list = []
    for chunk in chunks:
        if len(chunk) < min_ms:
            if normalized and len(normalized[-1]) + len(chunk) <= max_ms:
                normalized[-1] += chunk
            continue
        while len(chunk) > max_ms:
            normalized.append(chunk[:max_ms])
            chunk = chunk[max_ms:]
        if len(chunk) >= min_ms:
            normalized.append(chunk)

    result: list[Path] = []
    for i, chunk in enumerate(normalized):
        dst = out_dir / f"{prefix}-{i:04d}.wav"
        chunk.export(dst, format="wav", parameters=["-ac", "1", "-ar", "24000"])
        result.append(dst)
    return result


def choose_reference(records: list[dict]) -> dict:
    if not records:
        raise ValueError("Нет пригодных фрагментов для reference.")
    preferred = [r for r in records if 6.0 <= r.get("duration", 0) <= 14.0 and len(r.get("text", "")) >= 20]
    pool = preferred or records
    return max(pool, key=lambda r: (len(r.get("text", "")), r.get("duration", 0)))


def prepare(project_name: str, source_folder: str, asr_python: str, asr_model: str = "Qwen/Qwen3-ASR-0.6B") -> dict:
    from tqdm.auto import tqdm

    paths = ProjectPaths.for_name(project_name).ensure()
    source_files = discover_audio(source_folder)
    if not source_files:
        raise ValueError("В указанной папке нет поддерживаемых аудиофайлов.")

    clips_dir = paths.dataset / "clips"
    if clips_dir.exists():
        shutil.rmtree(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)

    clips: list[Path] = []
    for file_idx, src in enumerate(tqdm(source_files, desc="Нарезка", unit="файл", dynamic_ncols=False, ncols=100, leave=False), 1):
        clips.extend(segment_file(src, clips_dir, f"audio-{file_idx:03d}"))

    if not clips:
        raise ValueError("После нарезки не осталось фрагментов длительностью 2.5–15 секунд.")

    manifest = paths.transcripts / "asr-input.jsonl"
    asr_output = paths.transcripts / "asr-output.jsonl"
    manifest.write_text("\n".join(json.dumps({"audio": str(p)}, ensure_ascii=False) for p in clips) + "\n", encoding="utf-8")

    worker = Path(__file__).with_name("asr_worker.py")
    cmd = [asr_python, str(worker), "--manifest", str(manifest), "--output", str(asr_output), "--model", asr_model, "--language", "Russian"]
    subprocess.run(cmd, check=True)

    raw_records = [json.loads(line) for line in asr_output.read_text(encoding="utf-8").splitlines() if line.strip()]
    AudioSegment = _audio_segment_class()
    records: list[dict] = []
    for item in raw_records:
        text = item.get("text", "").strip()
        if len(text) < 2:
            continue
        p = Path(item["audio"])
        duration = len(AudioSegment.from_file(p)) / 1000.0
        records.append({"audio": str(p), "text": text, "duration": duration})

    if not records:
        raise ValueError("Qwen3-ASR не вернул пригодных расшифровок.")

    ref = choose_reference(records)
    ref_wav = paths.dataset / "ref.wav"
    shutil.copy2(ref["audio"], ref_wav)
    (paths.dataset / "ref_text.txt").write_text(ref["text"], encoding="utf-8")

    metadata_lines = []
    raw_lines = []
    for r in records:
        rel = os.path.relpath(r["audio"], paths.dataset)
        metadata_lines.append(json.dumps({"audio_filepath": rel, "text": r["text"], "ref_audio": "ref.wav"}, ensure_ascii=False))
        raw_lines.append(json.dumps({"audio": str(Path(r["audio"]).resolve()), "text": r["text"], "ref_audio": str(ref_wav.resolve()), "language": "Russian"}, ensure_ascii=False))

    (paths.dataset / "metadata.jsonl").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    (paths.dataset / "train_raw.jsonl").write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    return {
        "source_files": len(source_files),
        "clips": len(clips),
        "usable": len(records),
        "duration_seconds": round(sum(r["duration"] for r in records), 1),
        "dataset": str(paths.dataset),
        "reference": str(ref_wav),
    }
