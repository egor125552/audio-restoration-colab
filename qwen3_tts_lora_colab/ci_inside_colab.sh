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
python3 qwen3_tts_lora_colab/colab_stream.py

HF_HOME="/tmp/qwen3-gradio-hf" HF_HUB_CACHE="/tmp/qwen3-drive-model-cache" "$TTS_PY" - <<'PY'
from gradio.tunneling import BINARY_PATH
from huggingface_hub.constants import HF_HUB_CACHE

assert BINARY_PATH.startswith("/tmp/qwen3-gradio-hf/"), BINARY_PATH
assert HF_HUB_CACHE == "/tmp/qwen3-drive-model-cache", HF_HUB_CACHE
print("GRADIO TUNNEL CACHE AND MODEL CACHE ARE SEPARATE")
PY

"$TTS_PY" -m pip install --quiet "playwright==1.55.0"
"$TTS_PY" -m playwright install --with-deps chromium

rm -f "$QWEN_TRAIN_SMOKE_MARKER"
QWEN_TRAIN_PORT=7860 "$TTS_PY" -u qwen3_tts_lora_colab/launch_colab.py >/tmp/qwen3-gradio.log 2>&1 &
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
import json

launcher = Path('qwen3_tts_lora_colab/launch_colab.py').read_text(encoding='utf-8')
assert 'server_name="127.0.0.1"' in launcher
assert 'share=not SMOKE' in launcher
assert 'quiet=False' in launcher
assert 'if not share_url:' in launcher
assert 'Gradio не смог создать публичную ссылку' in launcher
assert 'Встроенный прокси Colab' not in launcher
assert 'demo.block_thread()' in launcher
assert 'build_demo' in launcher

notebook = json.loads(Path('notebooks/Qwen3_TTS_LoRA_RU.ipynb').read_text(encoding='utf-8'))
drive_cell = ''.join(notebook['cells'][4]['source'])
last = ''.join(notebook['cells'][-1]['source'])
markdown = ''.join(notebook['cells'][-2]['source'])

assert "LOCAL_HF_HOME = Path('/content/qwen3-hf-home')" in drive_cell
assert "MODEL_CACHE = ROOT / '.cache' / 'huggingface' / 'hub'" in drive_cell
assert "os.environ['HF_HOME'] = str(LOCAL_HF_HOME)" in drive_cell
assert "os.environ['HF_HUB_CACHE'] = str(MODEL_CACHE)" in drive_cell
assert 'serve_kernel_port_as_iframe' not in last
assert 'ready_port=PORT' not in last
assert 'run_streamed(cmd)' in last
assert 'публичную ссылку `gradio.live`' in markdown
assert 'встроенный прокси' not in markdown.lower()
print('COLAB PUBLIC GRADIO LINK + LOCAL TUNNEL CACHE WIRED OK')
PY

cat /tmp/qwen3-gradio.log
printf '\nQWEN3 TTS LORA COLAB SMOKE OK\n'
