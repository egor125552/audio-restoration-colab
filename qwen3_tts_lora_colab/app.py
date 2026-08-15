from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import gradio as gr

from project import ProjectPaths, discover_audio, stage_directory
from prepare_dataset import prepare as prepare_dataset

ROOT = Path(os.environ.get("QWEN_TRAIN_HOME", "/content/qwen3-tts-trainer"))
ASR_PYTHON = ROOT / "asr-env/bin/python"
SMOKE = os.environ.get("QWEN_TRAIN_SMOKE") == "1"
SMOKE_MARKER = Path(os.environ.get("QWEN_TRAIN_SMOKE_MARKER", "/tmp/qwen3-trainer-smoke.jsonl"))
TRAIN_PROCESS: subprocess.Popen | None = None
TRAIN_PROJECT: str | None = None

MODELS = {
    "Qwen3-TTS 0.6B Base — рекомендуется для T4": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen3-TTS 1.7B Base — тяжелее": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}
ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"


def _record(event: str, **payload) -> None:
    if not SMOKE:
        return
    SMOKE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    with SMOKE_MARKER.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n")


def create_project(name: str):
    _record("create_project", name=name)
    try:
        paths = ProjectPaths.for_name(name).ensure()
        return f"Проект «{paths.name}» готов. Постоянная папка: {paths.root}"
    except Exception as exc:
        return f"Ошибка: {exc}"


def inspect_source(folder: str):
    _record("inspect_source", folder=folder)
    try:
        files = discover_audio(folder)
        if not files:
            return "Аудиофайлы не найдены."
        names = ", ".join(p.name for p in files[:5])
        tail = "" if len(files) <= 5 else f" и ещё {len(files)-5}"
        return f"Найдено файлов: {len(files)}. Первые: {names}{tail}."
    except Exception as exc:
        return f"Ошибка: {exc}"


def download_models(model_label: str):
    _record("download_models", model=model_label)
    if SMOKE:
        return "SMOKE: загрузка моделей пропущена."
    try:
        from huggingface_hub import snapshot_download
        model_id = MODELS[model_label]
        print(f"Скачиваю базовую TTS-модель: {model_id}", flush=True)
        snapshot_download(model_id)
        print(f"Скачиваю модель распознавания: {ASR_MODEL}", flush=True)
        snapshot_download(ASR_MODEL)
        return "Модели загружены и находятся в кэше Hugging Face."
    except Exception as exc:
        return f"Ошибка загрузки: {exc}"


def recommend_epochs(sample_count: int) -> int:
    """Keep total dataset exposure near Alexandria's tested 250–400 range."""
    if sample_count <= 0:
        return 10
    return max(1, min(15, round(320 / sample_count)))


def prepare_project_dataset(name: str, source_folder: str):
    _record("prepare_dataset", name=name, source=source_folder)
    if SMOKE:
        return "SMOKE: подготовка датасета вызвана.", 3
    try:
        if not ASR_PYTHON.exists():
            raise RuntimeError(f"Не найдено окружение ASR: {ASR_PYTHON}")
        result = prepare_dataset(name, source_folder, str(ASR_PYTHON), ASR_MODEL)
        minutes = result["duration_seconds"] / 60
        suggested = recommend_epochs(result["usable"])
        status = (
            f"Датасет готов. Исходных файлов: {result['source_files']}. "
            f"Фрагментов после нарезки: {result['clips']}. Пригодных: {result['usable']}. "
            f"Длительность: {minutes:.1f} мин. Для такого объёма я поставил {suggested} эпох. "
            f"Папка: {result['dataset']}"
        )
        return status, suggested
    except Exception as exc:
        return f"Ошибка подготовки: {exc}", gr.skip()


def _latest_checkpoint(paths: ProjectPaths) -> str:
    found = []
    for p in paths.checkpoints.glob("epoch-*"):
        try:
            found.append((int(p.name.split("-")[-1]), p))
        except ValueError:
            pass
    if not found:
        return "сохранений пока нет"
    epoch, path = max(found)
    return f"последнее сохранение: эпоха {epoch} ({path})"


def training_status(name: str):
    _record("refresh_status", name=name)
    global TRAIN_PROCESS
    try:
        paths = ProjectPaths.for_name(name).ensure()
    except Exception as exc:
        return f"Ошибка: {exc}"
    if TRAIN_PROCESS is not None and TRAIN_PROCESS.poll() is None:
        state = f"обучение идёт, номер процесса {TRAIN_PROCESS.pid}"
    elif TRAIN_PROCESS is not None:
        state = f"процесс завершён, код {TRAIN_PROCESS.returncode}"
    else:
        state = "обучение сейчас не запущено"
    return f"{state}; {_latest_checkpoint(paths)}."


def start_training(name: str, model_label: str, epochs: int, lr: float, rank: int, alpha: int, grad_accum: int, resume: bool):
    global TRAIN_PROCESS, TRAIN_PROJECT
    _record(
        "start_training", name=name, model=model_label, epochs=epochs, lr=lr,
        rank=rank, alpha=alpha, grad_accum=grad_accum, resume=resume,
    )
    if SMOKE:
        return "SMOKE: кнопка обучения дошла до Python."
    if TRAIN_PROCESS is not None and TRAIN_PROCESS.poll() is None:
        return f"Обучение уже идёт, номер процесса {TRAIN_PROCESS.pid}."
    try:
        paths = ProjectPaths.for_name(name).ensure()
        if not (paths.dataset / "metadata.jsonl").exists():
            raise RuntimeError("Сначала подготовьте датасет.")

        local_dataset = paths.work / "dataset"
        stage_directory(paths.dataset, local_dataset)
        local_work = paths.work / "training"
        local_work.mkdir(parents=True, exist_ok=True)

        trainer = Path(__file__).with_name("train_lora.py")
        cmd = [
            sys.executable, "-u", str(trainer),
            "--data_dir", str(local_dataset),
            "--project_dir", str(paths.root),
            "--work_dir", str(local_work),
            "--model_name", MODELS[model_label],
            "--epochs", str(int(epochs)),
            "--lr", str(float(lr)),
            "--lora_r", str(int(rank)),
            "--lora_alpha", str(int(alpha)),
            "--gradient_accumulation_steps", str(int(grad_accum)),
        ]
        if resume:
            cmd.append("--resume")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        print("\nЗапускаю обучение. Прогресс будет обновляться в этой консоли.", flush=True)
        TRAIN_PROCESS = subprocess.Popen(cmd, env=env, stdout=None, stderr=None)
        TRAIN_PROJECT = paths.name
        return f"Обучение запущено, номер процесса {TRAIN_PROCESS.pid}. Прогресс смотрите в консоли Colab."
    except Exception as exc:
        return f"Ошибка запуска: {exc}"


def stop_training():
    global TRAIN_PROCESS
    _record("stop_training")
    if SMOKE:
        return "SMOKE: остановка вызвана."
    if TRAIN_PROCESS is None or TRAIN_PROCESS.poll() is not None:
        return "Активного обучения нет."
    TRAIN_PROCESS.send_signal(signal.SIGINT)
    try:
        TRAIN_PROCESS.wait(timeout=10)
    except subprocess.TimeoutExpired:
        TRAIN_PROCESS.terminate()
    return "Команда остановки отправлена. Последнее завершённое сохранение на Google Drive не потеряется."


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Qwen3-TTS LoRA — обучение голоса") as demo:
        gr.Markdown(
            "# Qwen3-TTS LoRA — обучение голоса\n"
            "Русский Colab-интерфейс. Длинные записи можно указать папкой; программа сама нарежет их, "
            "расшифрует Qwen3-ASR и подготовит датасет."
        )

        with gr.Tab("Проект"):
            gr.Markdown("Сначала назовите голос и укажите папку с записями. Всё, что относится к этому голосу, будет храниться в отдельной папке проекта на Google Drive.")
            project_name = gr.Textbox(
                label="Имя проекта",
                placeholder="Например: Путин",
                info="Просто понятное имя голоса. По нему создаётся отдельная папка с датасетом, сохранениями и готовой LoRA.",
            )
            source_folder = gr.Textbox(
                label="Папка с исходными аудиофайлами",
                placeholder="/content/drive/MyDrive/Голоса/Путин",
                info="Путь к папке с вашими длинными WAV, FLAC, MP3 и другими записями. Тексты заранее готовить не нужно.",
            )
            model = gr.Dropdown(
                label="Базовая модель",
                choices=list(MODELS),
                value=list(MODELS)[0],
                info="0.6B легче и безопаснее для бесплатной T4. 1.7B крупнее и потенциально сильнее, но требует заметно больше видеопамяти.",
            )
            with gr.Row():
                create_btn = gr.Button("Создать или открыть проект")
                inspect_btn = gr.Button("Проверить папку")
                models_btn = gr.Button("Скачать модели")
            project_status = gr.Textbox(label="Состояние проекта", interactive=False)

        with gr.Tab("Датасет"):
            gr.Markdown(
                "Нажмите «Подготовить датасет». Программа сама найдёт речь, нарежет длинные записи на удобные фрагменты, "
                "переведёт их в 24 кГц mono и расшифрует русский текст через Qwen3-ASR 0.6B. "
                "Во время работы в консоли будет одна перерисовывающаяся полоска прогресса."
            )
            prepare_btn = gr.Button("Подготовить датасет", variant="primary")
            dataset_status = gr.Textbox(label="Состояние датасета", interactive=False, lines=4)

        with gr.Tab("Обучение"):
            gr.Markdown(
                "### Что означают настройки\n"
                "Если не хочется разбираться, оставьте предложенные значения. После подготовки датасета число эпох "
                "подбирается автоматически по количеству фрагментов."
            )
            epochs = gr.Number(
                label="Всего эпох",
                value=10,
                minimum=1,
                precision=0,
                info="Эпоха — один полный проход по всем фрагментам. Слишком много эпох может переобучить голос и вызвать кашу. После подготовки датасета это число подставится автоматически; его можно изменить вручную.",
            )
            lr = gr.Number(
                label="Скорость обучения",
                value=0.000001,
                info="Насколько сильно LoRA меняется на каждом шаге. Слишком большое значение может быстро испортить голос. Для первого обучения оставьте 0.000001.",
            )
            rank = gr.Dropdown(
                label="Размер адаптера LoRA",
                choices=[8, 16, 32, 64],
                value=64,
                info="Сколько возможностей даём адаптеру для запоминания нового голоса. Больше — больше ёмкость и расход памяти. Значение 64 проверено Alexandria как хороший стандарт для одного голоса.",
            )
            alpha = gr.Dropdown(
                label="Множитель LoRA",
                choices=[16, 32, 64, 128],
                value=128,
                info="Масштаб влияния адаптера. Его используют вместе с размером LoRA. Для размера 64 проверенная пара — 128, поэтому без причины менять её не нужно.",
            )
            grad_accum = gr.Dropdown(
                label="Накопление градиентов",
                choices=[1, 2, 4, 8],
                value=4,
                info="Объединяет несколько маленьких шагов перед изменением весов. Это помогает экономить видеопамять. Для T4 значение 4 — нормальный безопасный вариант.",
            )
            resume = gr.Checkbox(
                label="Продолжить с последнего сохранения",
                value=True,
                info="Если Colab отключился или вы остановили обучение, продолжить с последней полностью сохранённой эпохи. Обычно лучше оставить включённым.",
            )
            gr.Markdown(
                "Полоска в консоли показывает текущую эпоху, шаг и loss. Loss — это оценка ошибки модели. "
                "Она должна в целом снижаться, но слишком низкий loss тоже может означать переобучение. "
                "Для LoRA Alexandria на поддерживаемом языке считает область около 4.1–4.2 хорошей целью; ниже примерно 4.1 возрастает риск зажёванной речи. "
                "После каждой эпохи создаётся сохранение на Google Drive."
            )
            with gr.Row():
                train_btn = gr.Button("Начать обучение", variant="primary")
                stop_btn = gr.Button("Остановить обучение", variant="stop")
                refresh_btn = gr.Button("Обновить состояние")
            train_status = gr.Textbox(label="Состояние обучения", interactive=False, lines=3)

        create_btn.click(create_project, project_name, project_status, api_name="create_project")
        inspect_btn.click(inspect_source, source_folder, project_status, api_name="inspect_source")
        models_btn.click(download_models, model, project_status, api_name="download_models")
        prepare_btn.click(prepare_project_dataset, [project_name, source_folder], [dataset_status, epochs], api_name="prepare_dataset")
        train_btn.click(start_training, [project_name, model, epochs, lr, rank, alpha, grad_accum, resume], train_status, api_name="start_training")
        stop_btn.click(stop_training, None, train_status, api_name="stop_training")
        refresh_btn.click(training_status, project_name, train_status, api_name="refresh_status")

    return demo


def main() -> None:
    demo = build_demo()
    if SMOKE:
        demo.queue().launch(server_name="127.0.0.1", server_port=int(os.environ.get("QWEN_TRAIN_PORT", "7860")), share=False, show_error=True)
    else:
        print("Запускаю русский интерфейс Qwen3-TTS LoRA. Эта ячейка будет работать постоянно — так и задумано.", flush=True)
        print("Вывод обучения остаётся видимым; полоска прогресса обновляется на месте.", flush=True)
        demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True, prevent_thread_lock=False)


if __name__ == "__main__":
    main()
