#!/usr/bin/env bash
set -euo pipefail

export XVC_HOME="${XVC_HOME:-/tmp/x-vc-colab-smoke}"
export XVC_SMOKE_MARKER="${XVC_SMOKE_MARKER:-/tmp/xvc-gradio-callbacks.jsonl}"
export XVC_SMOKE_SOURCE="${XVC_SMOKE_SOURCE:-/tmp/xvc-source.wav}"
export XVC_SMOKE_REFERENCE="${XVC_SMOKE_REFERENCE:-/tmp/xvc-reference.wav}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALLER="${ROOT}/x_vc_colab/install_xvc.sh"
PYTHON="${XVC_HOME}/.venv/bin/python"
LOCAL_LOG="/tmp/xvc-gradio-local.log"
SHARE_LOG="/tmp/xvc-gradio-share.log"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

printf '\n=== Official Colab runtime identity ===\n'
cat /etc/os-release || true
python3 --version

bash "${INSTALLER}" "${XVC_HOME}"

printf '\n=== Real X-VC Python/model import + Gradio construction ===\n'
XVC_HOME="${XVC_HOME}" "${PYTHON}" "${ROOT}/x_vc_colab/smoke_import.py"

printf '\n=== Browser test dependency ===\n'
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
    ("XVC_SMOKE_SOURCE", 220.0),
    ("XVC_SMOKE_REFERENCE", 330.0),
):
    path = Path(os.environ[env_name])
    sample_rate = 16000
    frames = int(sample_rate * 0.7)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for index in range(frames):
            value = int(
                0.12 * 32767 * math.sin(2 * math.pi * frequency * index / sample_rate)
            )
            wav.writeframesraw(struct.pack("<h", value))
    if path.stat().st_size <= 44:
        raise SystemExit(f"Generated WAV is empty: {path}")
    print(path, path.stat().st_size)
PY

printf '\n=== Launch the actual X-VC Gradio app in CI-safe mode ===\n'
XVC_HOME="${XVC_HOME}" \
XVC_SMOKE_MODE=1 \
XVC_SMOKE_MARKER="${XVC_SMOKE_MARKER}" \
"${PYTHON}" -u "${ROOT}/x_vc_colab/app.py" --port 7860 >"${LOCAL_LOG}" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 90); do
  if curl --fail --silent --show-error http://127.0.0.1:7860/ >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "Gradio server exited before becoming ready" >&2
    cat "${LOCAL_LOG}" >&2 || true
    exit 1
  fi
  sleep 1
done

if [[ "${ready}" != "1" ]]; then
  echo "Gradio server did not become ready" >&2
  cat "${LOCAL_LOG}" >&2 || true
  exit 1
fi

printf '\n=== Chromium end-to-end interaction with every button ===\n'
XVC_SMOKE_MARKER="${XVC_SMOKE_MARKER}" \
XVC_SMOKE_SOURCE="${XVC_SMOKE_SOURCE}" \
XVC_SMOKE_REFERENCE="${XVC_SMOKE_REFERENCE}" \
"${PYTHON}" "${ROOT}/x_vc_colab/e2e_gradio.py"

cleanup
SERVER_PID=""

printf '\n=== Public gradio.live share-link smoke ===\n'
share_ok=0
for attempt in 1 2 3; do
  : > "${SHARE_LOG}"
  XVC_HOME="${XVC_HOME}" \
  XVC_SMOKE_MODE=1 \
  "${PYTHON}" -u "${ROOT}/x_vc_colab/app.py" --share --port 7861 >"${SHARE_LOG}" 2>&1 &
  SERVER_PID=$!

  share_url=""
  for _ in $(seq 1 90); do
    share_url="$(grep -Eo 'https://[A-Za-z0-9-]+\.gradio\.live' "${SHARE_LOG}" | head -n1 || true)"
    if [[ -n "${share_url}" ]]; then
      break
    fi
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  if [[ -n "${share_url}" ]]; then
    echo "PUBLIC SHARE URL: ${share_url}"
    if curl -L --retry 5 --retry-delay 2 --fail --silent --show-error \
      --max-time 30 "${share_url}" >/dev/null; then
      printf '%s\n' "${share_url}" > /tmp/xvc-gradio-share-url.txt
      share_ok=1
      cleanup
      SERVER_PID=""
      break
    fi
  fi

  echo "Share attempt ${attempt} failed" >&2
  cat "${SHARE_LOG}" >&2 || true
  cleanup
  SERVER_PID=""
  sleep 3
done

if [[ "${share_ok}" != "1" ]]; then
  echo "Could not create and reach a public gradio.live URL after 3 attempts." >&2
  exit 1
fi

printf '\n=== X-VC Colab smoke suite passed ===\n'
