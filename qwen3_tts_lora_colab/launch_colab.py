from __future__ import annotations

import os

from app import build_demo

os.environ.setdefault("PYTHONUNBUFFERED", "1")
PORT = int(os.environ.get("QWEN_TRAIN_PORT", "7860"))
SMOKE = os.environ.get("QWEN_TRAIN_SMOKE") == "1"

print("Запускаю интерфейс Qwen3-TTS LoRA.", flush=True)
if not SMOKE:
    print("Gradio сейчас создаст публичную ссылку gradio.live.", flush=True)
    print("Если туннель не поднимется, точная причина будет напечатана ниже.", flush=True)

demo = build_demo()
_, local_url, share_url = demo.queue().launch(
    server_name="127.0.0.1",
    server_port=PORT,
    share=not SMOKE,
    inline=False,
    show_error=True,
    prevent_thread_lock=True,
    quiet=False,
)

if not SMOKE:
    if not share_url:
        demo.close()
        raise RuntimeError(
            "Gradio не смог создать публичную ссылку. Посмотрите сообщение об ошибке туннеля выше."
        )
    print(f"Публичная интернет-ссылка: {share_url}", flush=True)

print("Интерфейс запущен. Ячейка останется занятой, а вывод и прогресс обучения будут видны ниже.", flush=True)
demo.block_thread()
