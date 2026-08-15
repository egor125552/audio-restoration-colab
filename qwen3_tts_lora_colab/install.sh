#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-${QWEN_TRAIN_HOME:-/content/qwen3-tts-trainer}}"
TTS_VENV="${ROOT}/tts-env"
ASR_VENV="${ROOT}/asr-env"
PYTHON_BIN="${QWEN_TRAIN_PYTHON_BIN:-$(command -v python3)}"

log() { printf '\n[qwen3-trainer] %s\n' "$*"; }

ensure_command() {
  local cmd="$1" pkg="$2"
  if command -v "$cmd" >/dev/null 2>&1; then return; fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "Не найдена команда ${cmd}, а apt-get недоступен." >&2
    exit 1
  fi
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$pkg"
}

ensure_isolated_venv() {
  local venv="$1"
  if [[ -f "${venv}/pyvenv.cfg" ]] && grep -qi '^include-system-site-packages = true' "${venv}/pyvenv.cfg"; then
    log "Пересоздаю старое окружение без системных пакетов Colab: ${venv}"
    rm -rf "$venv"
  fi
  if [[ ! -x "${venv}/bin/python" ]]; then
    uv venv --seed --python "$PYTHON_BIN" "$venv"
  fi
}

mkdir -p "$ROOT"
ensure_command git git
ensure_command ffmpeg ffmpeg

if ! command -v uv >/dev/null 2>&1; then
  log "Устанавливаю uv"
  python3 -m pip install --quiet --disable-pip-version-check "uv==0.12.1"
fi

log "Использую Python: $(${PYTHON_BIN} --version 2>&1)"
ensure_isolated_venv "$TTS_VENV"
ensure_isolated_venv "$ASR_VENV"

TTS_PY="${TTS_VENV}/bin/python"
ASR_PY="${ASR_VENV}/bin/python"

log "Устанавливаю окружение Qwen3-TTS + LoRA"
uv pip install --python "$TTS_PY" --upgrade \
  "pip<26" "setuptools<81" wheel \
  "qwen-tts==0.1.1" \
  "transformers==4.57.3" \
  "peft==0.18.1" \
  "gradio==5.49.1" \
  "huggingface-hub==0.36.2" \
  "pydub==0.25.1" \
  "tqdm>=4.66,<5" \
  "psutil>=5.9,<8"

log "Устанавливаю отдельное окружение Qwen3-ASR"
uv pip install --python "$ASR_PY" --upgrade \
  "pip<26" "setuptools<81" wheel \
  "qwen-asr==0.0.6" \
  "transformers==4.57.6" \
  "huggingface-hub==0.36.2"

cat > "${ROOT}/INSTALLATION.txt" <<TXT
Qwen3-TTS environment: ${TTS_VENV}
Qwen3-ASR environment: ${ASR_VENV}
Python: $(${TTS_PY} --version 2>&1)
Qwen3-TTS: 0.1.1
Qwen3-ASR: 0.0.6
Isolation: system site-packages disabled
TXT

log "Готово: ${ROOT}"
