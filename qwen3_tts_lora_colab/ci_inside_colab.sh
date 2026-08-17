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
"$TTS_PY" -m py_compile \
  qwen3_tts_lora_colab/project.py \
  qwen3_tts_lora_colab/app.py \
  qwen3_tts_lora_colab/prepare_dataset.py \
  qwen3_tts_lora_colab/asr_worker.py \
  qwen3_tts_lora_colab/train_lora.py \
  qwen3_tts_lora_colab/inference_lora.py \
  qwen3_tts_lora_colab/e2e_gradio.py

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

stream = Path('qwen3_tts_lora_colab/colab_stream.py').read_text(encoding='utf-8')
assert 'ready_port' not in stream
assert 'on_ready' not in stream
assert 'port-proxy' not in stream

project = Path('qwen3_tts_lora_colab/project.py').read_text(encoding='utf-8')
assert 'def list_projects()' in project
assert 'not path.name.startswith(".")' in project
assert '(path / "project.json").is_file()' in project

app = Path('qwen3_tts_lora_colab/app.py').read_text(encoding='utf-8')
assert 'label="Существующий проект"' in app
assert 'label="Новый проект"' in app
assert 'Обновить список проектов' in app
assert 'project_name.change(' in app
assert 'project_changed' in app
assert 'gr.Dropdown(choices=choices, value=paths.name)' in app
assert 'label="Имя проекта"' not in app
assert 'label="Batch size"' in app
assert 'label="Накопление градиентов"' in app
assert 'Gradient checkpointing' in app
assert 'with gr.Tab("Инференс")' in app
assert 'Checkpoint для озвучивания' in app
assert 'Референс голоса' in app
assert 'generate_inference' in app
assert 'PREPARING_PROJECT' in app
assert 'OP_LOCK = threading.Lock()' in app
assert 'ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"' in app

trainer = Path('qwen3_tts_lora_colab/train_lora.py').read_text(encoding='utf-8')
assert 'p.add_argument("--batch_size", type=int, default=1)' in trainer
assert 'p.add_argument("--gradient_accumulation_steps", type=int, default=4)' in trainer
assert 'argparse.BooleanOptionalAction' in trainer
assert 'gradient_checkpointing' in trainer
assert 'attention_implementation' in trainer

inference = Path('qwen3_tts_lora_colab/inference_lora.py').read_text(encoding='utf-8')
assert 'generate_voice_clone(' in inference
assert 'x_vector_only_mode' in inference
assert 'non_streaming_mode' in inference
assert 'subtalker_top_k' in inference
assert 'repetition_penalty' in inference
assert 'PeftModel.from_pretrained' in inference

dataset_prep = Path('qwen3_tts_lora_colab/prepare_dataset.py').read_text(encoding='utf-8')
assert 'seek_step=10' in dataset_prep
assert 'chunk.export(dst, format="wav")' in dataset_prep
assert 'asr_model: str = "Qwen/Qwen3-ASR-1.7B"' in dataset_prep
assert '"--batch-size",' in dataset_prep
assert '"2",' in dataset_prep

asr_worker = Path('qwen3_tts_lora_colab/asr_worker.py').read_text(encoding='utf-8')
assert 'p.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")' in asr_worker
assert 'p.add_argument("--batch-size", type=int, default=2)' in asr_worker
assert 'max_inference_batch_size=batch_size' in asr_worker
assert 'audio_batch = [item["audio"] for item in batch]' in asr_worker
assert 'model.transcribe(audio=audio_batch, language=args.language)' in asr_worker
assert 'progress.update(len(batch))' in asr_worker
assert 'TRANSFORMERS_VERBOSITY' in asr_worker

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
print('COLAB PUBLIC GRADIO LINK + ADVANCED TRAINING + INFERENCE UI WIRED OK')
PY

cat /tmp/qwen3-gradio.log
printf '\nQWEN3 TTS LORA COLAB SMOKE OK\n'
