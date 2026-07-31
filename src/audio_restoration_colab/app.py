from __future__ import annotations

import argparse
import html
import os
import shutil
from pathlib import Path
from typing import Any

from .catalog import MODEL_SPECS, default_browser_settings, get_model
from .jobs import AudioJobService, JobProcessingError, JobProgress
from .runtime import ModelResult, RuntimeLayout, SubprocessWorker
from .ui_state import (
    DEFAULT_MODEL_ID,
    merge_active_settings,
    selection_view,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CHOICES = [
    (model.short_title, model_id) for model_id, model in MODEL_SPECS.items()
]

CONTROL_KEYS = (
    "denoise_quality",
    "denoise_segment",
    "lavasr_input_rate",
    "lavasr_denoise",
    "lavasr_batch",
    "flashsr_lowpass",
    "audiosr_mode",
    "audiosr_steps",
    "audiosr_guidance",
    "audiosr_seed",
    "audiosr_lowpass",
)

CSS = """
:root {
  font-size: 18px;
}
.gradio-container {
  max-width: 1080px !important;
  margin: 0 auto !important;
}
#model-information {
  border-left: 0.3rem solid #2563eb;
  padding-left: 1rem;
}
#job-status {
  border: 1px solid #64748b;
  padding: 0.75rem 1rem;
  min-height: 3rem;
  background: #f8fafc;
}
button {
  min-height: 3rem;
}
@media (max-width: 640px) {
  :root {
    font-size: 16px;
  }
}
"""


class DemoWorker:
    """Лёгкая замена моделей только для проверки интерфейса в CI."""

    def run(
        self,
        *,
        model_id: str,
        source: Path,
        output_dir: Path,
        settings: dict[str, object],
        progress: JobProgress,
    ) -> list[ModelResult]:
        del settings
        progress(0.55, "Демонстрационный режим: создаю тестовый результат…")
        model = get_model(model_id)
        results: list[ModelResult] = []
        for role in model.output_roles:
            target = output_dir / f"{role}{source.suffix.lower()}"
            shutil.copy2(source, target)
            results.append(ModelResult(role=role, path=target))
        return results


def build_app(*, demo_mode: bool = False):
    import gradio as gr

    service = _build_service(demo_mode=demo_mode)
    initial_view = selection_view(
        DEFAULT_MODEL_ID,
        default_browser_settings(),
    )
    demo_notice = (
        "<div role='note'><strong>Режим проверки:</strong> "
        "тяжёлые модели отключены.</div>"
        if demo_mode
        else ""
    )

    with gr.Blocks(
        title="Восстановление и очистка аудио",
        theme="default",
        css=CSS,
        delete_cache=(21_600, 21_600),
        analytics_enabled=False,
    ) as app:
        gr.Markdown(
            "# Восстановление и очистка аудио\n\n"
            "Загрузи файл, выбери модель и нажми «Начать обработку». "
            "Модель скачивается только при первом запуске."
        )
        if demo_notice:
            gr.HTML(demo_notice)
        gr.Markdown(
            "**Важно:** дорисовка создаёт правдоподобные верхние частоты, "
            "но не может буквально вернуть удалённый оригинал."
        )

        settings_state = gr.BrowserState(
            default_browser_settings(),
            storage_key="audio-restoration-settings-v1",
            secret="audio-restoration-colab-settings",
        )
        selected_model_state = gr.BrowserState(
            DEFAULT_MODEL_ID,
            storage_key="audio-restoration-selected-model-v1",
            secret="audio-restoration-colab-model",
        )
        output_format_state = gr.BrowserState(
            "wav",
            storage_key="audio-restoration-format-v1",
            secret="audio-restoration-colab-format",
        )

        with gr.Group():
            input_file = gr.File(
                label="1. Аудиофайл",
                file_types=["audio"],
                type="filepath",
            )
            model_dropdown = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=DEFAULT_MODEL_ID,
                label="2. Модель",
                info="Настройки ниже изменятся под выбранную модель.",
                allow_custom_value=False,
            )
            model_information = gr.Markdown(
                initial_view.information,
                elem_id="model-information",
            )

        with gr.Group(
            visible=initial_view.visible_panels["denoise"],
        ) as denoise_group:
            gr.Markdown("### Настройки DeNoise")
            denoise_quality = gr.Radio(
                choices=[
                    ("Быстро", "fast"),
                    ("Сбалансированно", "balanced"),
                    ("Максимальное качество", "maximum"),
                ],
                value=initial_view.values["denoise_quality"],
                label="Качество и скорость",
                info="Большее качество требует больше времени.",
            )
            denoise_segment = gr.Slider(
                minimum=128,
                maximum=352,
                step=32,
                value=initial_view.values["denoise_segment"],
                label="Размер обрабатываемого фрагмента",
                info="256 — безопасное значение для Tesla T4.",
            )

        with gr.Group(
            visible=initial_view.visible_panels["lavasr"],
        ) as lavasr_group:
            gr.Markdown("### Настройки маленькой LavaSR")
            lavasr_input_rate = gr.Dropdown(
                choices=[
                    ("Определить автоматически", "auto"),
                    ("8 кГц", 8000),
                    ("16 кГц", 16000),
                    ("24 кГц", 24000),
                    ("32 кГц", 32000),
                    ("44,1 кГц", 44100),
                    ("48 кГц", 48000),
                ],
                value=initial_view.values["lavasr_input_rate"],
                label="Ожидаемая полоса исходника",
                allow_custom_value=False,
            )
            lavasr_denoise = gr.Checkbox(
                value=initial_view.values["lavasr_denoise"],
                label="Одновременно убрать шум",
                info="Включай только для речи с постоянным шумом.",
            )
            lavasr_batch = gr.Checkbox(
                value=initial_view.values["lavasr_batch"],
                label="Режим длинного файла",
                info="Уменьшает расход памяти.",
            )

        with gr.Group(
            visible=initial_view.visible_panels["flashsr"],
        ) as flashsr_group:
            gr.Markdown("### Настройки средней FlashSR")
            flashsr_lowpass = gr.Checkbox(
                value=initial_view.values["flashsr_lowpass"],
                label="Подготовить неровный частотный срез",
                info=(
                    "Полезно после MP3 и нейросетевого разделения. "
                    "Отключи для исходника с уже ровным срезом."
                ),
            )

        with gr.Group(
            visible=initial_view.visible_panels["audiosr"],
        ) as audiosr_group:
            gr.Markdown("### Настройки большой AudioSR")
            audiosr_mode = gr.Radio(
                choices=[
                    ("Обычный звук и музыка", "basic"),
                    ("Речь", "speech"),
                ],
                value=initial_view.values["audiosr_mode"],
                label="Тип звука",
            )
            audiosr_steps = gr.Slider(
                minimum=10,
                maximum=100,
                step=10,
                value=initial_view.values["audiosr_steps"],
                label="Количество шагов",
                info="50 обычно достаточно. Больше — медленнее.",
            )
            audiosr_guidance = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                step=0.1,
                value=initial_view.values["audiosr_guidance"],
                label="Сила обработки",
                info="Слишком большое значение может добавить артефакты.",
            )
            audiosr_seed = gr.Number(
                value=initial_view.values["audiosr_seed"],
                precision=0,
                minimum=0,
                maximum=2_147_483_647,
                label="Случайное зерно",
                info="Поменяй число, чтобы получить другой вариант.",
            )
            audiosr_lowpass = gr.Checkbox(
                value=initial_view.values["audiosr_lowpass"],
                label="Подготовить неровный частотный срез",
                info="Рекомендуется для MP3 и разделённых файлов.",
            )

        output_format = gr.Radio(
            choices=[
                ("Как у исходного файла", "source"),
                ("MP3, максимальное качество 320 кбит/с", "mp3"),
                ("WAV без сжатия", "wav"),
            ],
            value="wav",
            label="3. Формат результата",
        )
        run_button = gr.Button(
            "4. Начать обработку",
            variant="primary",
        )
        status = gr.HTML(
            "<div role='status' aria-live='polite'>"
            "Ожидаю аудиофайл и запуск.</div>",
            elem_id="job-status",
        )

        gr.Markdown("## Результаты")
        with gr.Row():
            primary_preview = gr.Audio(
                label="Основной результат",
                type="filepath",
                interactive=False,
            )
            secondary_preview = gr.Audio(
                label="Второй результат, например выделенный шум",
                type="filepath",
                interactive=False,
            )
        result_files = gr.File(
            label="Скачать отдельные файлы",
            file_count="multiple",
            interactive=False,
        )
        result_zip = gr.File(
            label="Скачать все результаты одним ZIP",
            interactive=False,
        )
        diagnostic_log = gr.File(
            label="Скачать лог последнего запуска",
            interactive=False,
        )

        controls = [
            denoise_quality,
            denoise_segment,
            lavasr_input_rate,
            lavasr_denoise,
            lavasr_batch,
            flashsr_lowpass,
            audiosr_mode,
            audiosr_steps,
            audiosr_guidance,
            audiosr_seed,
            audiosr_lowpass,
        ]
        groups = [
            denoise_group,
            lavasr_group,
            flashsr_group,
            audiosr_group,
        ]
        selection_outputs = [
            model_information,
            *groups,
            *controls,
        ]

        def apply_selection(model_id: str, saved_settings: dict[str, Any]):
            view = selection_view(model_id, saved_settings)
            return (
                view.information,
                gr.update(visible=view.visible_panels["denoise"]),
                gr.update(visible=view.visible_panels["lavasr"]),
                gr.update(visible=view.visible_panels["flashsr"]),
                gr.update(visible=view.visible_panels["audiosr"]),
                *(view.values[key] for key in CONTROL_KEYS),
            )

        def change_model(model_id: str, saved_settings: dict[str, Any]):
            selected = (
                model_id if model_id in MODEL_SPECS else DEFAULT_MODEL_ID
            )
            return selected, *apply_selection(selected, saved_settings)

        model_dropdown.change(
            change_model,
            inputs=[model_dropdown, settings_state],
            outputs=[selected_model_state, *selection_outputs],
        )

        def save_settings(
            model_id: str,
            saved_settings: dict[str, Any],
            *values: Any,
        ):
            all_values = dict(zip(CONTROL_KEYS, values, strict=True))
            return merge_active_settings(
                saved_settings,
                model_id,
                _active_values(model_id, all_values),
            )

        gr.on(
            triggers=[control.change for control in controls],
            fn=save_settings,
            inputs=[model_dropdown, settings_state, *controls],
            outputs=settings_state,
        )

        output_format.change(
            lambda value: value,
            inputs=output_format,
            outputs=output_format_state,
        )

        def load_browser_state(
            saved_model: str,
            saved_settings: dict[str, Any],
        ):
            selected = (
                saved_model
                if saved_model in MODEL_SPECS
                else DEFAULT_MODEL_ID
            )
            return selected, *apply_selection(selected, saved_settings)

        app.load(
            load_browser_state,
            inputs=[selected_model_state, settings_state],
            outputs=[model_dropdown, *selection_outputs],
        )
        app.load(
            lambda saved: saved
            if saved in {"source", "mp3", "wav"}
            else "wav",
            inputs=output_format_state,
            outputs=output_format,
        )

        def process_audio(
            input_path: str | None,
            model_id: str,
            format_choice: str,
            saved_settings: dict[str, Any],
            gradio_progress=gr.Progress(track_tqdm=False),  # noqa: B008
        ):
            if not input_path:
                raise gr.Error("Сначала загрузи аудиофайл.")
            raw_settings = (saved_settings or {}).get(model_id, {})

            def report(fraction: float, message: str) -> None:
                gradio_progress(fraction, desc=message)

            try:
                result = service.process(
                    source=Path(input_path),
                    model_id=model_id,
                    format_choice=format_choice,
                    raw_settings=raw_settings,
                    progress=report,
                )
            except JobProcessingError as error:
                status_html = (
                    "<div role='status' aria-live='assertive'><strong>Ошибка: "
                    + html.escape(str(error))
                    + "</strong><br>Полный лог сохранён ниже и одновременно "
                    "выведен в консоль Colab.</div>"
                )
                return (
                    status_html,
                    None,
                    None,
                    [],
                    None,
                    str(error.log_path),
                )
            except ValueError as error:
                raise gr.Error(str(error)) from error
            status_html = (
                "<div role='status' aria-live='polite'><strong>"
                + html.escape(result.message)
                + "</strong></div>"
            )
            return (
                status_html,
                _path_or_none(result.primary_preview),
                _path_or_none(result.secondary_preview),
                [str(path) for path in result.files],
                str(result.archive),
                str(result.log_path),
            )

        run_button.click(
            process_audio,
            inputs=[
                input_file,
                model_dropdown,
                output_format,
                settings_state,
            ],
            outputs=[
                status,
                primary_preview,
                secondary_preview,
                result_files,
                result_zip,
                diagnostic_log,
            ],
            concurrency_limit=1,
            show_progress="full",
        )

    app.queue(default_concurrency_limit=1, max_size=8)
    return app


def _build_service(*, demo_mode: bool) -> AudioJobService:
    cache_root = Path(
        os.environ.get(
            "AUDIO_RESTORATION_CACHE",
            "/content/audio-restoration-models",
        )
    )
    jobs_root = Path(
        os.environ.get(
            "AUDIO_RESTORATION_WORKDIR",
            "/content/audio-restoration-work",
        )
    )
    worker = (
        DemoWorker()
        if demo_mode
        else SubprocessWorker(
            layout=RuntimeLayout(
                project_root=PROJECT_ROOT,
                cache_root=cache_root,
            )
        )
    )
    return AudioJobService(jobs_root=jobs_root, worker=worker)


def _active_values(
    model_id: str,
    all_values: dict[str, Any],
) -> dict[str, Any]:
    if model_id.startswith("denoise_"):
        return {
            "quality": all_values["denoise_quality"],
            "segment": all_values["denoise_segment"],
        }
    if model_id == "lavasr_small":
        return {
            "input_rate": all_values["lavasr_input_rate"],
            "denoise": all_values["lavasr_denoise"],
            "batch": all_values["lavasr_batch"],
        }
    if model_id == "flashsr_medium":
        return {"lowpass": all_values["flashsr_lowpass"]}
    if model_id == "audiosr_large":
        return {
            "mode": all_values["audiosr_mode"],
            "steps": all_values["audiosr_steps"],
            "guidance": all_values["audiosr_guidance"],
            "seed": all_values["audiosr_seed"],
            "lowpass": all_values["audiosr_lowpass"],
        }
    return {}


def _path_or_none(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Русский интерфейс очистки и дорисовки аудио.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Создать временную публичную ссылку Gradio.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Не запускать и не скачивать модели; используется в проверках.",
    )
    parser.add_argument(
        "--server-name",
        default="0.0.0.0",
        help="Адрес сервера. По умолчанию: 0.0.0.0.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Порт сервера. По умолчанию: 7860.",
    )
    arguments = parser.parse_args()
    app = build_app(demo_mode=arguments.demo)
    app.launch(
        share=arguments.share,
        server_name=arguments.server_name,
        server_port=arguments.server_port,
        show_error=True,
        max_file_size="2gb",
        enable_monitoring=False,
    )


if __name__ == "__main__":
    main()
