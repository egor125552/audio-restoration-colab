#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT="${1:-/content/audio-restoration-models}"
ENV_DIR="$CACHE_ROOT/envs/audiosr_trt"
READY_MARKER="$ENV_DIR/.audio-restoration-ready-v3"
STACK_CHECK_ONLY="${AUDIO_RESTORATION_T4_STACK_CHECK:-0}"

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

rm -rf "$ENV_DIR"
mkdir -p "$(dirname "$ENV_DIR")"

"$UV_BIN" python install 3.10
"$UV_BIN" venv --python 3.10 "$ENV_DIR"

"$UV_BIN" pip install \
  --python "$ENV_DIR/bin/python" \
  "setuptools==80.9.0"

# Torch-TensorRT 2.4.0 официально таргетирует PyTorch 2.4.0 и TensorRT 10.1.
"$UV_BIN" pip install \
  --python "$ENV_DIR/bin/python" \
  --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.4.0" "torchaudio==2.4.0" "torchvision==0.19.0"

# Пинним весь TensorRT-стек одной версией. Иначе metapackage tensorrt==10.1.0
# может разрешить более новый tensorrt-cu12, что нам не нужно для воспроизводимости.
"$UV_BIN" pip install \
  --python "$ENV_DIR/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  "tensorrt==10.1.0" \
  "tensorrt-cu12==10.1.0" \
  "tensorrt-cu12-bindings==10.1.0" \
  "tensorrt-cu12-libs==10.1.0" \
  "torch-tensorrt==2.4.0+cu121"

"$UV_BIN" pip install \
  --python "$ENV_DIR/bin/python" \
  "git+https://github.com/haoheliu/versatile_audio_super_resolution.git@d312fbab9f0e94087d9f2802d03cf184353cc805"

"$UV_BIN" pip check --python "$ENV_DIR/bin/python"

# Не создаём READY_MARKER раньше этой проверки. Если реальный импорт TensorRT
# на T4 упадёт, следующий запуск обязан повторить установку/проверку, а не
# ошибочно сообщить, что среда уже готова.
AUDIO_RESTORATION_T4_STACK_CHECK="$STACK_CHECK_ONLY" \
  "$ENV_DIR/bin/python" - <<'PY'
import importlib.metadata
import os

import audiosr
import tensorrt
import torch

print("AudioSR импортирован:", audiosr.__file__)
print("PyTorch:", torch.__version__)
print("CUDA доступна:", torch.cuda.is_available())
print("Torch-TensorRT package:", importlib.metadata.version("torch-tensorrt"))
print("TensorRT:", tensorrt.__version__)

if os.environ.get("AUDIO_RESTORATION_T4_STACK_CHECK") == "1":
    print("CI без GPU: import torch_tensorrt пропущен, потому что 2.4.0 запрашивает CUDA device при импорте.")
else:
    import torch_tensorrt

    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch-TensorRT import:", torch_tensorrt.__version__)
PY

touch "$READY_MARKER"
echo "Среда AudioSR TensorRT готова: $ENV_DIR"
