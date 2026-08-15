from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    import gradio as gr
    import peft
    import transformers
    from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    import app

    demo = app.build_demo()
    config = demo.get_config_file()
    serialized = str(config)
    required = [
        "Имя проекта",
        "Папка с исходными аудиофайлами",
        "Создать или открыть проект",
        "Проверить папку",
        "Скачать модели",
        "Подготовить датасет",
        "Начать обучение",
        "Остановить обучение",
        "Обновить состояние",
    ]
    missing = [x for x in required if x not in serialized]
    if missing:
        raise AssertionError(f"В Gradio-конфиге нет элементов: {missing}")

    root = Path(os.environ.get("QWEN_TRAIN_HOME", "/content/qwen3-tts-trainer"))
    asr_python = root / "asr-env/bin/python"
    if not asr_python.exists():
        raise AssertionError(f"Не найден ASR Python: {asr_python}")
    subprocess.run(
        [str(asr_python), "-c", "from qwen_asr import Qwen3ASRModel; import transformers; print('QWEN3-ASR IMPORT OK', transformers.__version__)"],
        check=True,
    )

    print("QWEN3-TTS IMPORT OK", transformers.__version__)
    print("PEFT IMPORT OK", peft.__version__)
    print("GRADIO UI BUILD OK", gr.__version__)
    print("MODEL CLASSES OK", Qwen3TTSModel.__name__, Qwen3TTSTokenizer.__name__)


if __name__ == "__main__":
    main()
