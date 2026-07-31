#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT="${1:-/content/audio-restoration-models}"
ENV_DIR="$CACHE_ROOT/envs/audiosr_trt"
READY_MARKER="$ENV_DIR/.audio-restoration-ready-v1"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU не найдена. Запусти эту проверку в Colab с GPU." >&2
  exit 2
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo "GPU: $GPU_NAME"
if [[ "$GPU_NAME" != *"T4"* ]]; then
  echo "Предупреждение: скрипт рассчитан прежде всего на Tesla T4." >&2
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

# Torch-TensorRT 2.4 официально имеет CUDA 12.1 wheel и рассчитан на PyTorch 2.4.
"$UV_BIN" pip install \
  --python "$ENV_DIR/bin/python" \
  --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.4.1" "torchaudio==2.4.1" "torchvision==0.19.1"

"$UV_BIN" pip install \
  --python "$ENV_DIR/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  "tensorrt==10.1.0" "torch-tensorrt==2.4.0+cu121"

"$UV_BIN" pip install \
  --python "$ENV_DIR/bin/python" \
  "git+https://github.com/haoheliu/versatile_audio_super_resolution.git@d312fbab9f0e94087d9f2802d03cf184353cc805"

touch "$READY_MARKER"

echo "Среда AudioSR TensorRT готова: $ENV_DIR"
"$ENV_DIR/bin/python" - <<'PY'
import torch
import torch_tensorrt
import tensorrt

print("PyTorch:", torch.__version__)
print("CUDA доступна:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Torch-TensorRT:", torch_tensorrt.__version__)
print("TensorRT:", tensorrt.__version__)
PY
