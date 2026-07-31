#!/usr/bin/env bash
set -euo pipefail

CURRENT_STAGE="startup"
trap 'rc=$?; echo "" >&2; echo "T4 setup failed during: $CURRENT_STAGE" >&2; echo "Line $LINENO: $BASH_COMMAND" >&2; echo "Exit code: $rc" >&2; exit $rc' ERR

CACHE_ROOT="${1:-/content/audio-restoration-models}"
ENV_DIR="$CACHE_ROOT/envs/audiosr_trt"
READY_MARKER="$ENV_DIR/.audio-restoration-ready-v5"
STACK_CHECK_ONLY="${AUDIO_RESTORATION_T4_STACK_CHECK:-0}"

CURRENT_STAGE="detect NVIDIA GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  echo "GPU: $GPU_NAME"
  if [[ "$GPU_NAME" != *"T4"* ]]; then
    echo "Предупреждение: скрипт рассчитан прежде всего на Tesla T4." >&2
  fi
elif [[ "$STACK_CHECK_ONLY" == "1" ]]; then
  echo "Режим CI: GPU нет, проверяю установку, зависимости и CPU-безопасные импорты."
else
  echo "NVIDIA GPU не найдена. Запусти эту проверку в Colab с GPU." >&2
  exit 2
fi

CURRENT_STAGE="find uv"
if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif [[ -x /root/.local/bin/uv ]]; then
  UV_BIN="/root/.local/bin/uv"
else
  echo "uv не найден. Сначала выполни установочную ячейку проекта." >&2
  exit 3
fi

if [[ -f "$READY_MARKER" ]]; then
  echo "Среда AudioSR TensorRT уже готова: $ENV_DIR"
  exit 0
fi

CURRENT_STAGE="recreate isolated environment"
rm -rf "$ENV_DIR"
mkdir -p "$(dirname "$ENV_DIR")"

CURRENT_STAGE="install Python 3.10.20"
"$UV_BIN" python install 3.10.20
CURRENT_STAGE="create Python 3.10.20 venv"
"$UV_BIN" venv --python 3.10.20 "$ENV_DIR"

CURRENT_STAGE="install setuptools"
"$UV_BIN" pip install \
  --python "$ENV_DIR/bin/python" \
  "setuptools==80.9.0"

CURRENT_STAGE="install PyTorch CUDA 12.1 stack"
"$UV_BIN" pip install \
  --python "$ENV_DIR/bin/python" \
  --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.4.0" "torchaudio==2.4.0" "torchvision==0.19.0"

CURRENT_STAGE="install TensorRT and Torch-TensorRT"
"$UV_BIN" pip install \
  --python "$ENV_DIR/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  "tensorrt==10.1.0" \
  "tensorrt-cu12==10.1.0" \
  "tensorrt-cu12-bindings==10.1.0" \
  "tensorrt-cu12-libs==10.1.0" \
  "torch-tensorrt==2.4.0+cu121"

CURRENT_STAGE="install AudioSR and matplotlib"
"$UV_BIN" pip install \
  --python "$ENV_DIR/bin/python" \
  "git+https://github.com/haoheliu/versatile_audio_super_resolution.git@d312fbab9f0e94087d9f2802d03cf184353cc805" \
  "matplotlib==3.9.4"

CURRENT_STAGE="pip dependency check"
"$UV_BIN" pip check --python "$ENV_DIR/bin/python"

CURRENT_STAGE="validate AudioSR, CUDA and Torch-TensorRT imports"
AUDIO_RESTORATION_T4_STACK_CHECK="$STACK_CHECK_ONLY" \
  "$ENV_DIR/bin/python" - <<'PY'
import importlib.metadata
import os

import audiosr
import audiosr.utilities.tools as audiosr_tools
import matplotlib
import tensorrt
import torch

print("AudioSR импортирован:", audiosr.__file__)
print("AudioSR utilities импортирован:", audiosr_tools.__file__)
print("Matplotlib:", matplotlib.__version__)
print("PyTorch:", torch.__version__)
print("CUDA доступна:", torch.cuda.is_available())
print("Torch-TensorRT package:", importlib.metadata.version("torch-tensorrt"))
print("TensorRT:", tensorrt.__version__)

if os.environ.get("AUDIO_RESTORATION_T4_STACK_CHECK") == "1":
    print("CI без GPU: import torch_tensorrt пропущен, потому что 2.4.0 запрашивает CUDA device при импорте.")
else:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch внутри изолированной среды не видит CUDA, хотя nvidia-smi видит GPU.")

    import torch_tensorrt

    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch-TensorRT import:", torch_tensorrt.__version__)
PY

CURRENT_STAGE="write ready marker"
touch "$READY_MARKER"
echo "Среда AudioSR TensorRT готова: $ENV_DIR"
