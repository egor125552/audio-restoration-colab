#!/usr/bin/env bash
set -euo pipefail

ROOT="/tmp/qwen3-tts-trainer-ci"
export QWEN_TRAIN_HOME="$ROOT"
export QWEN_TRAIN_LOCAL_ROOT="/tmp/Qwen3-TTS Training"
export QWEN_TRAIN_WORK_ROOT="/tmp/qwen3-tts-work"
export QWEN_TRAIN_SMOKE=1
export QWEN_TRAIN_SMOKE_MARKER="/tmp/qwen3-trainer-smoke.jsonl"

bash qwen3_tts_lora_colab/install.sh "$ROOT"
TTS_PY="$ROOT/tts-env/bin/python"
ASR_PY="$ROOT/asr-env/bin/python"

"$TTS_PY" qwen3_tts_lora_colab/smoke_import.py

"$TTS_PY" -m pip install --quiet "playwright==1.55.0"
"$TTS_PY" -m playwright install --with-deps chromium

rm -f "$QWEN_TRAIN_SMOKE_MARKER"
QWEN_TRAIN_PORT=7860 "$TTS_PY" -u qwen3_tts_lora_colab/app.py >/tmp/qwen3-gradio.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:7860/ >/dev/null; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    cat /tmp/qwen3-gradio.log
    exit 1
  fi
  sleep 1
done

"$TTS_PY" qwen3_tts_lora_colab/e2e_gradio.py

"$TTS_PY" - <<'PY'
from pathlib import Path
text = Path('qwen3_tts_lora_colab/launch_colab.py').read_text(encoding='utf-8')
assert 'os.execv' in text
assert 'PYTHONUNBUFFERED' in text
print('COLAB LIVE OUTPUT LAUNCHER OK')
PY

cat /tmp/qwen3-gradio.log
printf '\nQWEN3 TTS LORA COLAB SMOKE OK\n'
