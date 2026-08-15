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
    import inference_lora

    demo = app.build_demo()
    config = demo.get_config_file()
    serialized = str(config)
    required = [
        "Существующий проект",
        "Новый проект",
        "Создать проект",
        "Обновить список проектов",
        "Папка с исходными аудиофайлами",
        "Проверить папку",
        "Скачать модели",
        "Подготовить датасет",
        "Всего эпох",
        "Скорость обучения",
        "Размер адаптера LoRA",
        "Множитель LoRA",
        "Batch size",
        "Накопление градиентов",
        "Gradient checkpointing",
        "Attention при обучении",
        "Продолжить с последнего сохранения",
        "Начать обучение",
        "Остановить обучение",
        "Обновить состояние",
        "Checkpoint для озвучивания",
        "Референс голоса",
        "Референсное аудио",
        "Точный текст референса",
        "Текст для озвучивания",
        "Расширенные настройки генерации",
        "Top K",
        "Temperature",
        "Repetition penalty",
        "Только x-vector",
        "Сгенерировать голос",
    ]
    missing = [x for x in required if x not in serialized]
    if missing:
        raise AssertionError(f"В Gradio-конфиге нет новых элементов: {missing}")

    assert app.recommend_epochs(30) in range(10, 16)
    assert app.recommend_epochs(60) in range(5, 9)
    assert app.recommend_epochs(120) in range(2, 4)
    print("EPOCH RECOMMENDATION OK", app.recommend_epochs(30), app.recommend_epochs(60), app.recommend_epochs(120))

    assert callable(inference_lora.synthesize)
    assert callable(inference_lora.load_model)
    print("LORA INFERENCE MODULE IMPORT OK")

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
