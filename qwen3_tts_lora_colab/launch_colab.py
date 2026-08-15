from __future__ import annotations

import os

from app import build_demo

os.environ.setdefault("PYTHONUNBUFFERED", "1")
PORT = int(os.environ.get("QWEN_TRAIN_PORT", "7860"))

print("Запускаю интерфейс без внешнего Gradio-туннеля.", flush=True)
print("Colab откроет локальный порт через свой встроенный прокси.", flush=True)
print("Ячейка останется занятой, а вывод и прогресс обучения будут видны ниже.", flush=True)

demo = build_demo()
demo.queue().launch(
    server_name="127.0.0.1",
    server_port=PORT,
    share=False,
    show_error=True,
    prevent_thread_lock=False,
    quiet=True,
)
