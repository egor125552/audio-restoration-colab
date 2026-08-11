from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"SMOKE FAILED: {message}")


home = Path(os.environ.get("SEED_VC_HOME", "/content/seed-vc")).resolve()
source = home / "src"
if not source.is_dir():
    fail(f"Seed-VC source directory does not exist: {source}")

os.chdir(source)
sys.path.insert(0, str(source))

# These imports deliberately use the real installed packages and real upstream
# Seed-VC modules. No mocks are allowed in this file.
import gradio as gr  # noqa: E402
import librosa  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
import transformers  # noqa: E402

import app as seed_app  # noqa: E402
from modules.commons import build_model  # noqa: E402,F401
from seed_vc_wrapper import SeedVCWrapper  # noqa: E402,F401


for name, value in {
    "create_v1_interface": getattr(seed_app, "create_v1_interface", None),
    "create_v2_interface": getattr(seed_app, "create_v2_interface", None),
    "convert_voice_v1_wrapper": getattr(seed_app, "convert_voice_v1_wrapper", None),
    "convert_voice_v2_wrapper": getattr(seed_app, "convert_voice_v2_wrapper", None),
}.items():
    if not callable(value):
        fail(f"upstream app is missing callable {name}")

# Construct the real Gradio interfaces from upstream code. This catches Gradio
# API incompatibilities that a plain `import app` would miss.
v1 = seed_app.create_v1_interface()
v2 = seed_app.create_v2_interface()

for version, interface in (("v1", v1), ("v2", v2)):
    if not isinstance(interface, gr.Interface):
        fail(f"{version} did not create a gradio.Interface")
    config = interface.get_config_file()
    text = json.dumps(config, ensure_ascii=False)
    for required_label in (
        "Source Audio / 源音频",
        "Reference Audio / 参考音频",
    ):
        if required_label not in text:
            fail(f"{version} interface is missing {required_label!r}")

v1_config = json.dumps(v1.get_config_file(), ensure_ascii=False)
for required in (
    "Diffusion Steps / 扩散步数",
    "Length Adjust / 长度调整",
    "Use F0 conditioned model / 启用F0输入",
    "Auto F0 adjust / 自动F0调整",
    "Pitch shift / 音调变换",
):
    if required not in v1_config:
        fail(f"v1 interface is missing {required!r}")

v2_config = json.dumps(v2.get_config_file(), ensure_ascii=False)
for required in (
    "Intelligibility CFG Rate",
    "Similarity CFG Rate",
    "Top-p",
    "Temperature",
    "Repetition Penalty",
    "convert style/emotion/accent",
    "anonymization only",
):
    if required not in v2_config:
        fail(f"v2 interface is missing {required!r}")

print("SMOKE IMPORT OK")
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torchaudio={torchaudio.__version__}")
print(f"transformers={transformers.__version__}")
print(f"gradio={gr.__version__}")
print(f"librosa={librosa.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
