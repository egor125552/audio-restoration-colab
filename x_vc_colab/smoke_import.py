from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

XVC_HOME = Path(os.environ.get("XVC_HOME", "/tmp/x-vc-colab-smoke")).resolve()
SOURCE_DIR = XVC_HOME / "src"

if not SOURCE_DIR.is_dir():
    raise SystemExit(f"Missing X-VC source: {SOURCE_DIR}")

sys.path.insert(0, str(SOURCE_DIR))

modules = [
    "torch",
    "torchaudio",
    "librosa",
    "omegaconf",
    "models.codec.sac.model",
    "bins.infer_utils",
]

for name in modules:
    importlib.import_module(name)
    print(f"IMPORT OK: {name}")

from models.codec.sac.model import XVC
from bins.infer_utils import load_xvc

if XVC is None or load_xvc is None:
    raise SystemExit("X-VC public model symbols were not imported.")

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "x_vc_colab"))
from app import build_demo

demo = build_demo()
if demo is None:
    raise SystemExit("Gradio demo was not constructed.")

print("X-VC MODEL IMPORT OK:", XVC)
print("GRADIO BUILD OK:", type(demo).__name__)
