from __future__ import annotations

import os

from app import build_demo

os.environ.setdefault("PYTHONUNBUFFERED", "1")
PORT = int(os.environ.get("QWEN_TRAIN_PORT", "7860"))
SMOKE = os.environ.get("QWEN_TRAIN_SMOKE") == "1"

print("Запускаю интерфейс Qwen3-TTS LoRA.", flush=True)
print("Встроенный прокси Colab работает независимо от публичной интернет-ссылки.", flush=True)
print("Параллельно пробую создать обычную ссылку gradio.live.", flush=True)

demo = build_demo()
_, local_url, share_url = demo.queue().launch(
    server_name="127.0.0.1",
    server_port=PORT,
    share=not SMOKE,
    inline=False,
    show_error=True,
    prevent_thread_lock=True,
    quiet=True,
)

if not SMOKE:
    if share_url:
        print(f"Публичная интернет-ссылка: {share_url}", flush=True)
    else:
        print(
            "Публичная интернет-ссылка не работает. Используйте интерфейс внутри Colab — он продолжает работать через встроенный прокси.",
            flush=True,
        )

print("Интерфейс запущен. Ячейка останется занятой, а вывод и прогресс обучения будут видны ниже.", flush=True)
demo.block_thread()
