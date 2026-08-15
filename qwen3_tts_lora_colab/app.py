from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr

from inference_lora import base_model_for_project, synthesize
from project import ProjectPaths, discover_audio, list_projects, stage_directory
from prepare_dataset import prepare as prepare_dataset

ROOT = Path(os.environ.get("QWEN_TRAIN_HOME", "/content/qwen3-tts-trainer"))
ASR_PYTHON = ROOT / "asr-env/bin/python"
SMOKE = os.environ.get("QWEN_TRAIN_SMOKE") == "1"
SMOKE_MARKER = Path(os.environ.get("QWEN_TRAIN_SMOKE_MARKER", "/tmp/qwen3-trainer-smoke.jsonl"))
TRAIN_PROCESS: subprocess.Popen | None = None
TRAIN_PROJECT: str | None = None
PREPARING_PROJECT: str | None = None
OP_LOCK = threading.Lock()

MODELS = {
    "Qwen3-TTS 0.6B Base — рекомендуется для T4": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen3-TTS 1.7B Base — тяжелее": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}
ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"
LATEST_ADAPTER = "Готовая LoRA — последняя сохранённая"
AUTO_REFERENCE = "__AUTO_REFERENCE__"
LANGUAGES = ["Russian", "Auto", "English", "German", "French", "Spanish", "Italian", "Portuguese", "Chinese", "Japanese", "Korean"]


def _record(event: str, **payload) -> None:
    if not SMOKE:
        return
    SMOKE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    with SMOKE_MARKER.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n")


def _training_running() -> bool:
    return TRAIN_PROCESS is not None and TRAIN_PROCESS.poll() is None


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


def _checkpoint_choices(name: str | None) -> list[str]:
    if not name:
        return []
    try:
        paths = ProjectPaths.for_name(name)
    except Exception:
        return []
    choices: list[str] = []
    if (paths.adapters / "adapter_config.json").exists():
        choices.append(LATEST_ADAPTER)
    epochs = []
    for p in paths.checkpoints.glob("epoch-*"):
        if not (p / "adapter_config.json").exists():
            continue
        try:
            epochs.append((int(p.name.split("-")[-1]), p.name))
        except ValueError:
            pass
    choices.extend(name for _, name in sorted(epochs, reverse=True))
    return choices


def _resolve_adapter(paths: ProjectPaths, choice: str | None) -> Path:
    if choice == LATEST_ADAPTER:
        path = paths.adapters
    elif choice:
        path = paths.checkpoints / choice
    else:
        raise RuntimeError("Выберите checkpoint для инференса.")
    if not (path / "adapter_config.json").exists():
        raise RuntimeError(f"В выбранном checkpoint нет LoRA adapter: {path}")
    return path


def _reference_choices(name: str | None):
    choices = [("Автоматический ref.wav из датасета", AUTO_REFERENCE)]
    if not name:
        return choices
    try:
        paths = ProjectPaths.for_name(name)
        metadata = paths.dataset / "metadata.jsonl"
        if not metadata.exists():
            return choices
        for line in metadata.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            rel = item.get("audio_filepath") or item.get("audio")
            text = (item.get("text") or "").replace("\n", " ").strip()
            if not rel:
                continue
            short = text[:80] + ("…" if len(text) > 80 else "")
            choices.append((f"{Path(rel).name}: {short}", str(rel)))
    except Exception:
        pass
    return choices


def _load_reference(name: str | None, choice: str | None):
    if not name:
        return None, "", "Сначала выберите проект."
    try:
        paths = ProjectPaths.for_name(name)
        if choice in (None, AUTO_REFERENCE):
            audio = paths.dataset / "ref.wav"
            text_file = paths.dataset / "ref_text.txt"
            if not audio.exists():
                return None, "", "В проекте пока нет автоматического reference. Сначала подготовьте датасет."
            text = text_file.read_text(encoding="utf-8").strip() if text_file.exists() else ""
            return str(audio), text, f"Автоматический reference: {audio.name}. Текст подставлен из ref_text.txt."

        metadata = paths.dataset / "metadata.jsonl"
        text = ""
        if metadata.exists():
            for line in metadata.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                rel = item.get("audio_filepath") or item.get("audio")
                if str(rel) == str(choice):
                    text = (item.get("text") or "").strip()
                    break
        audio = paths.dataset / str(choice)
        if not audio.exists():
            return None, text, f"Файл reference не найден: {audio}"
        return str(audio), text, f"Reference из датасета: {audio.name}. Расшифровка подставлена автоматически."
    except Exception as exc:
        return None, "", f"Ошибка reference: {exc}"


def project_summary(name: str | None) -> str:
    if not name:
        return "Существующих проектов пока нет. Создайте новый проект ниже."
    try:
        paths = ProjectPaths.for_name(name)
        if not paths.root.exists():
            return f"Проект «{paths.name}» не найден на Google Drive. Обновите список проектов."

        metadata = paths.dataset / "metadata.jsonl"
        if metadata.exists():
            samples = sum(1 for line in metadata.read_text(encoding="utf-8").splitlines() if line.strip())
            dataset = f"датасет готов, фрагментов: {samples}"
        else:
            dataset = "датасет ещё не подготовлен"

        training_meta = paths.adapters / "training_meta.json"
        if training_meta.exists():
            try:
                meta = json.loads(training_meta.read_text(encoding="utf-8"))
                epoch = meta.get("epoch", "?")
                loss = meta.get("loss")
                batch = meta.get("batch_size")
                accum = meta.get("gradient_accumulation_steps")
                adapter = f"готовая LoRA есть, эпоха {epoch}"
                if isinstance(loss, (int, float)):
                    adapter += f", loss {loss:.4f}"
                if batch and accum:
                    adapter += f", batch {batch} × accumulation {accum}"
            except Exception:
                adapter = "готовая LoRA есть"
        else:
            adapter = "готовой LoRA пока нет"

        return f"Проект «{paths.name}»: {dataset}; {_latest_checkpoint(paths)}; {adapter}."
    except Exception as exc:
        return f"Ошибка: {exc}"


def project_changed(name: str | None):
    checkpoints = _checkpoint_choices(name)
    checkpoint_value = checkpoints[0] if checkpoints else None
    refs = _reference_choices(name)
    ref_audio, ref_text, ref_status = _load_reference(name, AUTO_REFERENCE)
    return (
        project_summary(name),
        gr.Dropdown(choices=checkpoints, value=checkpoint_value),
        gr.Dropdown(choices=refs, value=AUTO_REFERENCE),
        ref_audio,
        ref_text,
        ref_status,
    )


def create_project(name: str):
    _record("create_project", name=name)
    try:
        paths = ProjectPaths.for_name(name).ensure()
        choices = list_projects()
        return gr.Dropdown(choices=choices, value=paths.name), "", project_summary(paths.name)
    except Exception as exc:
        return gr.skip(), gr.skip(), f"Ошибка: {exc}"


def refresh_projects(selected: str | None):
    _record("refresh_projects", selected=selected)
    choices = list_projects()
    value = selected if selected in choices else (choices[0] if choices else None)
    return gr.Dropdown(choices=choices, value=value), project_summary(value)


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
    if sample_count <= 0:
        return 10
    return max(1, min(15, round(320 / sample_count)))


def prepare_project_dataset(name: str, source_folder: str):
    global PREPARING_PROJECT
    _record("prepare_dataset", name=name, source=source_folder)
    if SMOKE:
        return "SMOKE: подготовка датасета вызвана.", 3
    with OP_LOCK:
        if _training_running():
            return "Нельзя готовить датасет, пока идёт обучение. Сначала остановите обучение.", gr.skip()
        if PREPARING_PROJECT is not None:
            return f"Подготовка датасета уже идёт для проекта «{PREPARING_PROJECT}».", gr.skip()
        PREPARING_PROJECT = name
    try:
        if not name:
            raise RuntimeError("Сначала выберите проект.")
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
    finally:
        with OP_LOCK:
            PREPARING_PROJECT = None


def training_status(name: str):
    _record("refresh_status", name=name)
    try:
        paths = ProjectPaths.for_name(name)
        if not paths.root.exists():
            raise RuntimeError("Выбранный проект не найден. Обновите список проектов.")
    except Exception as exc:
        return f"Ошибка: {exc}"
    if _training_running():
        state = f"обучение идёт, номер процесса {TRAIN_PROCESS.pid}"
    elif TRAIN_PROCESS is not None:
        state = f"процесс завершён, код {TRAIN_PROCESS.returncode}"
    else:
        state = "обучение сейчас не запущено"
    if PREPARING_PROJECT:
        state += f"; одновременно идёт подготовка проекта {PREPARING_PROJECT}"
    return f"{state}; {_latest_checkpoint(paths)}."


def start_training(
    name: str,
    model_label: str,
    epochs: int,
    lr: float,
    rank: int,
    alpha: int,
    batch_size: int,
    grad_accum: int,
    gradient_checkpointing: bool,
    attention_implementation: str,
    resume: bool,
):
    global TRAIN_PROCESS, TRAIN_PROJECT
    _record(
        "start_training",
        name=name,
        model=model_label,
        epochs=epochs,
        lr=lr,
        rank=rank,
        alpha=alpha,
        batch_size=batch_size,
        grad_accum=grad_accum,
        gradient_checkpointing=gradient_checkpointing,
        attention_implementation=attention_implementation,
        resume=resume,
    )
    if SMOKE:
        return "SMOKE: кнопка обучения дошла до Python."
    with OP_LOCK:
        if _training_running():
            return f"Обучение уже идёт, номер процесса {TRAIN_PROCESS.pid}."
        if PREPARING_PROJECT is not None:
            return f"Сейчас идёт подготовка датасета проекта «{PREPARING_PROJECT}». Дождитесь её окончания."
        try:
            paths = ProjectPaths.for_name(name)
            if not paths.root.exists():
                raise RuntimeError("Выбранный проект не найден. Обновите список проектов.")
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
                "--batch_size", str(int(batch_size)),
                "--gradient_accumulation_steps", str(int(grad_accum)),
                "--attention_implementation", str(attention_implementation),
            ]
            cmd.append("--gradient-checkpointing" if gradient_checkpointing else "--no-gradient-checkpointing")
            if resume:
                cmd.append("--resume")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            effective = int(batch_size) * int(grad_accum)
            print(
                f"\nЗапускаю обучение: batch {int(batch_size)}, accumulation {int(grad_accum)}, "
                f"effective batch {effective}, gradient checkpointing "
                f"{'ON' if gradient_checkpointing else 'OFF'}.",
                flush=True,
            )
            TRAIN_PROCESS = subprocess.Popen(cmd, env=env, stdout=None, stderr=None)
            TRAIN_PROJECT = paths.name
            return (
                f"Обучение запущено, номер процесса {TRAIN_PROCESS.pid}. "
                f"Batch {int(batch_size)} × accumulation {int(grad_accum)} = effective batch {effective}. "
                "Прогресс смотрите в консоли Colab."
            )
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


def generate_inference(
    name: str,
    checkpoint: str,
    model_label: str,
    text: str,
    language: str,
    ref_audio: str,
    ref_text: str,
    do_sample: bool,
    top_k: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    subtalker_dosample: bool,
    subtalker_top_k: int,
    subtalker_top_p: float,
    subtalker_temperature: float,
    max_new_tokens: int,
    x_vector_only_mode: bool,
    non_streaming_mode: bool,
    attention_implementation: str,
):
    _record(
        "generate_inference",
        name=name,
        checkpoint=checkpoint,
        language=language,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        subtalker_dosample=subtalker_dosample,
        subtalker_top_k=subtalker_top_k,
        subtalker_top_p=subtalker_top_p,
        subtalker_temperature=subtalker_temperature,
        max_new_tokens=max_new_tokens,
        x_vector_only_mode=x_vector_only_mode,
        non_streaming_mode=non_streaming_mode,
        attention_implementation=attention_implementation,
    )
    if SMOKE:
        return None, "SMOKE: параметры инференса дошли до Python."
    if _training_running():
        return None, "Нельзя запускать инференс, пока обучение занимает GPU. Сначала остановите обучение."
    if PREPARING_PROJECT is not None:
        return None, f"Сейчас GPU занят подготовкой датасета проекта «{PREPARING_PROJECT}»."
    try:
        paths = ProjectPaths.for_name(name)
        adapter = _resolve_adapter(paths, checkpoint)
        base_model = base_model_for_project(paths.root, MODELS[model_label])
        audio_path, status = synthesize(
            project_dir=paths.root,
            base_model=base_model,
            adapter_path=adapter,
            checkpoint_label=checkpoint,
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            do_sample=bool(do_sample),
            top_k=int(top_k),
            top_p=float(top_p),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            subtalker_dosample=bool(subtalker_dosample),
            subtalker_top_k=int(subtalker_top_k),
            subtalker_top_p=float(subtalker_top_p),
            subtalker_temperature=float(subtalker_temperature),
            max_new_tokens=int(max_new_tokens),
            x_vector_only_mode=bool(x_vector_only_mode),
            non_streaming_mode=bool(non_streaming_mode),
            attention_implementation=attention_implementation,
        )
        return audio_path, status
    except Exception as exc:
        return None, f"Ошибка инференса: {exc}"


def build_demo() -> gr.Blocks:
    initial_projects = list_projects()
    initial_project = initial_projects[0] if initial_projects else None
    initial_checkpoints = _checkpoint_choices(initial_project)
    initial_checkpoint = initial_checkpoints[0] if initial_checkpoints else None
    initial_refs = _reference_choices(initial_project)
    initial_ref_audio, initial_ref_text, initial_ref_status = _load_reference(initial_project, AUTO_REFERENCE)

    with gr.Blocks(title="Qwen3-TTS LoRA — обучение голоса") as demo:
        gr.Markdown(
            "# Qwen3-TTS LoRA — обучение и проверка голоса\n"
            "Проекты, датасет, LoRA-обучение и инференс обученных checkpoint в одном интерфейсе."
        )

        with gr.Tab("Проект"):
            gr.Markdown(
                "Выберите уже сохранённый проект с Google Drive. Если нужен новый голос, создайте новый проект ниже. "
                "Старые названия вручную вводить больше не нужно."
            )
            project_name = gr.Dropdown(
                label="Существующий проект",
                choices=initial_projects,
                value=initial_project,
                allow_custom_value=False,
                info="Список проектов в папке Qwen3-TTS Training на Google Drive.",
            )
            refresh_projects_btn = gr.Button("Обновить список проектов")
            new_project_name = gr.Textbox(
                label="Новый проект",
                placeholder="Например: Путин",
                info="Заполняйте только когда хотите создать новый голос.",
            )
            create_btn = gr.Button("Создать проект", variant="primary")
            project_status = gr.Textbox(
                label="Состояние проекта",
                value=project_summary(initial_project),
                interactive=False,
                lines=4,
            )
            source_folder = gr.Textbox(
                label="Папка с исходными аудиофайлами",
                placeholder="/content/drive/MyDrive/Голоса/Путин",
                info="Путь к папке с длинными WAV, FLAC, MP3 и другими записями.",
            )
            model = gr.Dropdown(
                label="Базовая модель",
                choices=list(MODELS),
                value=list(MODELS)[0],
                info="0.6B легче для T4. 1.7B заметно тяжелее.",
            )
            with gr.Row():
                inspect_btn = gr.Button("Проверить папку")
                models_btn = gr.Button("Скачать модели")

        with gr.Tab("Датасет"):
            gr.Markdown(
                "«Подготовить датасет» нужен только когда исходные записи изменились или датасета ещё нет. "
                "Во время подготовки обучение того же Colab теперь запускаться не сможет."
            )
            prepare_btn = gr.Button("Подготовить датасет", variant="primary")
            dataset_status = gr.Textbox(label="Состояние датасета", interactive=False, lines=4)

        with gr.Tab("Обучение"):
            gr.Markdown(
                "Batch size — сколько фрагментов модель считает одновременно. Он сильнее всего влияет на VRAM. "
                "Accumulation — сколько таких batch накопить перед обновлением весов. "
                "Effective batch = Batch size × Accumulation."
            )
            epochs = gr.Number(label="Всего эпох", value=10, minimum=1, precision=0)
            lr = gr.Number(
                label="Скорость обучения",
                value=0.000001,
                info="Для первого обучения оставьте 0.000001.",
            )
            rank = gr.Dropdown(label="Размер адаптера LoRA", choices=[8, 16, 32, 64], value=64)
            alpha = gr.Dropdown(label="Множитель LoRA", choices=[16, 32, 64, 128], value=128)
            batch_size = gr.Dropdown(
                label="Batch size",
                choices=[1, 2, 4, 8],
                value=1,
                info="Настоящий GPU batch. На вашей T4 можно пробовать 2, затем 4 и следить за VRAM.",
            )
            grad_accum = gr.Dropdown(
                label="Накопление градиентов",
                choices=[1, 2, 4, 8],
                value=4,
                info="1 означает, что накопление фактически выключено.",
            )
            gradient_checkpointing = gr.Checkbox(
                label="Gradient checkpointing — экономить VRAM ценой скорости",
                value=True,
                info="Можно выключить и проверить, станет ли обучение быстрее. VRAM при этом вырастет.",
            )
            training_attention = gr.Dropdown(
                label="Attention при обучении",
                choices=["eager", "sdpa"],
                value="eager",
                info="eager — текущий проверенный режим. sdpa — встроенный оптимизированный PyTorch attention.",
            )
            resume = gr.Checkbox(
                label="Продолжить с последнего сохранения",
                value=True,
                info="Продолжает с последней полностью сохранённой эпохи.",
            )
            with gr.Row():
                train_btn = gr.Button("Начать обучение", variant="primary")
                stop_btn = gr.Button("Остановить обучение", variant="stop")
                refresh_btn = gr.Button("Обновить состояние")
            train_status = gr.Textbox(label="Состояние обучения", interactive=False, lines=4)

        with gr.Tab("Инференс"):
            gr.Markdown(
                "Выберите сохранённую LoRA или конкретную эпоху и сразу послушайте результат. "
                "По умолчанию используется автоматический ref.wav и его точная расшифровка из датасета."
            )
            checkpoint = gr.Dropdown(
                label="Checkpoint для озвучивания",
                choices=initial_checkpoints,
                value=initial_checkpoint,
                allow_custom_value=False,
            )
            reference_choice = gr.Dropdown(
                label="Референс голоса",
                choices=initial_refs,
                value=AUTO_REFERENCE,
                allow_custom_value=False,
                info="Можно оставить автоматический или выбрать любой расшифрованный фрагмент датасета.",
            )
            ref_audio = gr.Audio(
                label="Референсное аудио",
                value=initial_ref_audio,
                type="filepath",
                interactive=True,
            )
            ref_text = gr.Textbox(
                label="Точный текст референса",
                value=initial_ref_text,
                lines=3,
                info="В ICL-режиме этот текст должен совпадать с тем, что произнесено в референсе.",
            )
            reference_status = gr.Textbox(
                label="Состояние референса",
                value=initial_ref_status,
                interactive=False,
            )
            synth_text = gr.Textbox(
                label="Текст для озвучивания",
                placeholder="Введите фразу, которую должен сказать обученный голос",
                lines=4,
            )
            language = gr.Dropdown(label="Язык", choices=LANGUAGES, value="Russian")

            gr.Markdown("### Расширенные настройки генерации")
            do_sample = gr.Checkbox(label="Sampling", value=True)
            top_k = gr.Number(label="Top K", value=50, minimum=0, precision=0)
            top_p = gr.Number(label="Top P", value=1.0, minimum=0.0, maximum=1.0)
            temperature = gr.Number(label="Temperature", value=0.9, minimum=0.01)
            repetition_penalty = gr.Number(label="Repetition penalty", value=1.05, minimum=0.1)
            subtalker_dosample = gr.Checkbox(label="Subtalker sampling", value=True)
            subtalker_top_k = gr.Number(label="Subtalker Top K", value=50, minimum=0, precision=0)
            subtalker_top_p = gr.Number(label="Subtalker Top P", value=1.0, minimum=0.0, maximum=1.0)
            subtalker_temperature = gr.Number(label="Subtalker Temperature", value=0.9, minimum=0.01)
            max_new_tokens = gr.Number(label="Max new tokens", value=2048, minimum=32, precision=0)
            x_vector_only = gr.Checkbox(
                label="Только x-vector (без ICL)",
                value=False,
                info="Использует только speaker embedding; ref-текст не нужен, но качество клонирования может снизиться.",
            )
            non_streaming = gr.Checkbox(
                label="Полный non-streaming режим",
                value=True,
                info="Для готового текста используем полный non-streaming ввод, а не имитацию потокового текста.",
            )
            inference_attention = gr.Dropdown(
                label="Attention при инференсе",
                choices=["eager", "sdpa"],
                value="eager",
            )
            generate_btn = gr.Button("Сгенерировать голос", variant="primary")
            generated_audio = gr.Audio(label="Результат", type="filepath", interactive=False)
            inference_status = gr.Textbox(label="Состояние инференса", interactive=False, lines=3)

        project_name.change(
            project_changed,
            project_name,
            [project_status, checkpoint, reference_choice, ref_audio, ref_text, reference_status],
            api_name="project_status",
        )
        create_btn.click(
            create_project,
            new_project_name,
            [project_name, new_project_name, project_status],
            api_name="create_project",
        )
        refresh_projects_btn.click(
            refresh_projects,
            project_name,
            [project_name, project_status],
            api_name="refresh_projects",
        )
        reference_choice.change(
            _load_reference,
            [project_name, reference_choice],
            [ref_audio, ref_text, reference_status],
            api_name="reference_changed",
        )
        inspect_btn.click(inspect_source, source_folder, project_status, api_name="inspect_source")
        models_btn.click(download_models, model, project_status, api_name="download_models")
        prepare_btn.click(prepare_project_dataset, [project_name, source_folder], [dataset_status, epochs], api_name="prepare_dataset")
        train_btn.click(
            start_training,
            [
                project_name, model, epochs, lr, rank, alpha, batch_size, grad_accum,
                gradient_checkpointing, training_attention, resume,
            ],
            train_status,
            api_name="start_training",
        )
        stop_btn.click(stop_training, None, train_status, api_name="stop_training")
        refresh_btn.click(training_status, project_name, train_status, api_name="refresh_status")
        generate_btn.click(
            generate_inference,
            [
                project_name, checkpoint, model, synth_text, language, ref_audio, ref_text,
                do_sample, top_k, top_p, temperature, repetition_penalty,
                subtalker_dosample, subtalker_top_k, subtalker_top_p, subtalker_temperature,
                max_new_tokens, x_vector_only, non_streaming, inference_attention,
            ],
            [generated_audio, inference_status],
            api_name="generate_inference",
        )

    return demo


def main() -> None:
    demo = build_demo()
    if SMOKE:
        demo.queue().launch(
            server_name="127.0.0.1",
            server_port=int(os.environ.get("QWEN_TRAIN_PORT", "7860")),
            share=False,
            show_error=True,
        )
    else:
        print("Запускаю русский интерфейс Qwen3-TTS LoRA.", flush=True)
        print("Вывод подготовки, обучения и инференса остаётся видимым в Colab.", flush=True)
        demo.queue().launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            show_error=True,
            prevent_thread_lock=False,
        )


if __name__ == "__main__":
    main()
