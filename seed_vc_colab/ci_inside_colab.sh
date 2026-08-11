#!/usr/bin/env bash
set -euo pipefail

export SEED_VC_HOME="${SEED_VC_HOME:-/tmp/seed-vc-colab-smoke}"
export SEED_VC_SMOKE_MARKER="${SEED_VC_SMOKE_MARKER:-/tmp/seed-vc-gradio-callbacks.jsonl}"
export SEED_VC_SMOKE_SOURCE="${SEED_VC_SMOKE_SOURCE:-/tmp/seed-vc-source.wav}"
export SEED_VC_SMOKE_REFERENCE="${SEED_VC_SMOKE_REFERENCE:-/tmp/seed-vc-reference.wav}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALLER="${ROOT}/seed_vc_colab/install_seed_vc.sh"
PYTHON="${SEED_VC_HOME}/.venv/bin/python"
SERVER_LOG="/tmp/seed-vc-gradio.log"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

printf '\n=== Colab runtime identity ===\n'
cat /etc/os-release || true
python3 --version

bash "${INSTALLER}" "${SEED_VC_HOME}"

printf '\n=== Real Seed-VC import / interface construction ===\n'
SEED_VC_HOME="${SEED_VC_HOME}" "${PYTHON}" "${ROOT}/seed_vc_colab/smoke_import.py"

printf '\n=== Install browser test dependency ===\n'
uv pip install --python "${PYTHON}" "playwright==1.54.0"
"${PYTHON}" -m playwright install --with-deps chromium

printf '\n=== Generate real WAV uploads ===\n'
"${PYTHON}" - <<'PY'
import math
import os
import struct
import wave
from pathlib import Path

for env_name, frequency in (
    ("SEED_VC_SMOKE_SOURCE", 220.0),
    ("SEED_VC_SMOKE_REFERENCE", 330.0),
):
    path = Path(os.environ[env_name])
    sample_rate = 16000
    frames = int(sample_rate * 0.6)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for index in range(frames):
            value = int(0.15 * 32767 * math.sin(2 * math.pi * frequency * index / sample_rate))
            wav.writeframesraw(struct.pack("<h", value))
    if path.stat().st_size <= 44:
        raise SystemExit(f"Generated WAV is empty: {path}")
    print(path, path.stat().st_size)
PY

printf '\n=== Launch actual upstream-built Gradio page ===\n'
SEED_VC_HOME="${SEED_VC_HOME}" \
SEED_VC_SMOKE_MARKER="${SEED_VC_SMOKE_MARKER}" \
"${PYTHON}" "${ROOT}/seed_vc_colab/serve_upstream_ui.py" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 90); do
  if curl --fail --silent --show-error http://127.0.0.1:7860/ >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "Gradio server exited before becoming ready" >&2
    cat "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 1
done
if [[ "${ready}" != "1" ]]; then
  echo "Gradio server did not become ready" >&2
  cat "${SERVER_LOG}" >&2 || true
  exit 1
fi

printf '\n=== Chromium end-to-end Gradio interaction ===\n'
SEED_VC_SMOKE_MARKER="${SEED_VC_SMOKE_MARKER}" \
SEED_VC_SMOKE_SOURCE="${SEED_VC_SMOKE_SOURCE}" \
SEED_VC_SMOKE_REFERENCE="${SEED_VC_SMOKE_REFERENCE}" \
"${PYTHON}" "${ROOT}/seed_vc_colab/e2e_gradio.py"

printf '\n=== Smoke marker ===\n'
cat "${SEED_VC_SMOKE_MARKER}"
printf '\n=== Seed-VC Colab smoke suite passed ===\n'
