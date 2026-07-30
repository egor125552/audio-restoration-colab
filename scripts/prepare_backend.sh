#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:-}"
CACHE_ROOT="${2:-/content/audio-restoration-models}"

case "$BACKEND" in
  separator|lavasr|flashsr|audiosr) ;;
  *)
    echo "Неизвестная среда модели: $BACKEND" >&2
    exit 2
    ;;
esac

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif [[ -x /root/.local/bin/uv ]]; then
  UV_BIN="/root/.local/bin/uv"
else
  echo "uv не найден. Сначала выполни установочную ячейку Colab." >&2
  exit 3
fi

ENV_ROOT="$CACHE_ROOT/envs"
REPO_ROOT="$CACHE_ROOT/repos"
ENV_DIR="$ENV_ROOT/$BACKEND"
READY_MARKER="$ENV_DIR/.audio-restoration-ready-v1"
mkdir -p "$ENV_ROOT" "$REPO_ROOT"

if [[ -f "$READY_MARKER" ]]; then
  exit 0
fi

install_torch_241() {
  echo "Устанавливаю PyTorch для видеокарты…"
  "$UV_BIN" pip install \
    --python "$ENV_DIR/bin/python" \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchaudio==2.4.1 torchvision==0.19.1
}

case "$BACKEND" in
  separator)
    "$UV_BIN" python install 3.10
    "$UV_BIN" venv --allow-existing --python 3.10 "$ENV_DIR"
    install_torch_241
    "$UV_BIN" pip install \
      --python "$ENV_DIR/bin/python" \
      "audio-separator[gpu]==0.44.5"
    ;;
  lavasr)
    "$UV_BIN" python install 3.10
    "$UV_BIN" venv --allow-existing --python 3.10 "$ENV_DIR"
    install_torch_241
    "$UV_BIN" pip install \
      --python "$ENV_DIR/bin/python" \
      "git+https://github.com/ysharma3501/LavaSR.git@33ac040892519c1bb4aed7eb32e79af51cc29e2a"
    ;;
  flashsr)
    echo "Создаю среду FlashSR…"
    "$UV_BIN" python install 3.10
    "$UV_BIN" venv --allow-existing --python 3.10 "$ENV_DIR"
    install_torch_241
    FLASH_REPO="$REPO_ROOT/flashsr"
    if [[ ! -d "$FLASH_REPO/.git" ]]; then
      echo "Получаю код FlashSR…"
      GIT_LFS_SKIP_SMUDGE=1 git clone \
        https://huggingface.co/laion/FlashSR_One-step_Versatile_Audio_Super-resolution \
        "$FLASH_REPO"
    fi
    echo "Закрепляю проверенную версию FlashSR…"
    git -C "$FLASH_REPO" fetch origin \
      02f023d307e5f17b915d60b731e78a9664d7029f --depth 1
    git -C "$FLASH_REPO" checkout --detach \
      02f023d307e5f17b915d60b731e78a9664d7029f
    echo "Скачиваю три файла весов FlashSR — около 3,2 ГБ…"
    git -C "$FLASH_REPO" lfs pull
    echo "Устанавливаю код FlashSR…"
    "$UV_BIN" pip install \
      --python "$ENV_DIR/bin/python" \
      --editable "$FLASH_REPO" \
      einops librosa soundfile tqdm scipy
    ;;
  audiosr)
    "$UV_BIN" python install 3.9
    "$UV_BIN" venv --allow-existing --python 3.9 "$ENV_DIR"
    "$UV_BIN" pip install \
      --python "$ENV_DIR/bin/python" \
      --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.1.2 torchaudio==2.1.2 torchvision==0.16.2
    "$UV_BIN" pip install \
      --python "$ENV_DIR/bin/python" \
      "git+https://github.com/haoheliu/versatile_audio_super_resolution.git@d312fbab9f0e94087d9f2802d03cf184353cc805"
    ;;
esac

touch "$READY_MARKER"
echo "Среда $BACKEND готова."
