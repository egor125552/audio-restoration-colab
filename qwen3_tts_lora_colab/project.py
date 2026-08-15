from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".webm"}
DEFAULT_DRIVE_ROOT = Path(os.environ.get("QWEN_TRAIN_DRIVE_ROOT", "/content/drive/MyDrive/Qwen3-TTS Training"))
LOCAL_FALLBACK_ROOT = Path(os.environ.get("QWEN_TRAIN_LOCAL_ROOT", "/content/Qwen3-TTS Training"))
WORK_ROOT = Path(os.environ.get("QWEN_TRAIN_WORK_ROOT", "/content/qwen3-tts-work"))


def safe_project_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("Введите имя проекта.")
    cleaned = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "-", name).strip(" .-")
    if not cleaned:
        raise ValueError("Имя проекта состоит только из недопустимых символов.")
    return cleaned[:80]


def persistent_root() -> Path:
    if Path("/content/drive/MyDrive").exists():
        return DEFAULT_DRIVE_ROOT
    return LOCAL_FALLBACK_ROOT


@dataclass(frozen=True)
class ProjectPaths:
    name: str
    root: Path
    source: Path
    dataset: Path
    transcripts: Path
    checkpoints: Path
    adapters: Path
    logs: Path
    work: Path

    @classmethod
    def for_name(cls, name: str) -> "ProjectPaths":
        safe = safe_project_name(name)
        root = persistent_root() / safe
        return cls(
            name=safe,
            root=root,
            source=root / "source",
            dataset=root / "dataset",
            transcripts=root / "transcripts",
            checkpoints=root / "checkpoints",
            adapters=root / "adapter",
            logs=root / "logs",
            work=WORK_ROOT / safe,
        )

    def ensure(self) -> "ProjectPaths":
        for path in (self.root, self.source, self.dataset, self.transcripts, self.checkpoints, self.adapters, self.logs, self.work):
            path.mkdir(parents=True, exist_ok=True)
        self.write_manifest()
        return self

    def write_manifest(self) -> None:
        payload = {
            "name": self.name,
            "root": str(self.root),
            "source": str(self.source),
            "dataset": str(self.dataset),
            "checkpoints": str(self.checkpoints),
            "adapter": str(self.adapters),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "project.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def discover_audio(folder: str | Path) -> list[Path]:
    root = Path(folder).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Папка не найдена: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Это не папка: {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def stage_directory(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
