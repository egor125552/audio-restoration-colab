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

import bootstrap_colab
import prepare_assets_cli
from app import build_demo
from runtime import (
    LATENT_HOP_LENGTH,
    MAX_OFFLINE_CHUNK_SECONDS,
    _plan_offline_chunks,
)

if not callable(bootstrap_colab.say):
    raise SystemExit("Colab setup progress helper is not callable.")
if not callable(prepare_assets_cli.TextProgress):
    raise SystemExit("Model-download progress helper was not imported.")
print("COLAB PROGRESS HELPERS OK")

demo = build_demo()
if demo is None:
    raise SystemExit("Gradio demo was not constructed.")

# Reproduce the user's failing shape in timeline units:
# 2232 acoustic frames at 50 Hz = 44.64 s of 16 kHz audio.
sample_rate = 16000
long_samples = 2232 * (sample_rate // 50)
assert long_samples % LATENT_HOP_LENGTH == 0
ranges = _plan_offline_chunks(long_samples, sample_rate)
assert len(ranges) >= 2, ranges
assert ranges[0][0] == 0, ranges
assert ranges[-1][1] == long_samples, ranges
max_samples = int(MAX_OFFLINE_CHUNK_SECONDS * sample_rate)
for start, end in ranges:
    assert end > start, ranges
    assert end - start <= max_samples, ranges
    assert start % LATENT_HOP_LENGTH == 0, ranges
    assert end == long_samples or end % LATENT_HOP_LENGTH == 0, ranges
for previous, current in zip(ranges, ranges[1:]):
    assert current[0] < previous[1], ranges

print("LONG AUDIO CHUNK PLAN OK:", ranges)
print("X-VC MODEL IMPORT OK:", XVC)
print("GRADIO BUILD OK:", type(demo).__name__)
