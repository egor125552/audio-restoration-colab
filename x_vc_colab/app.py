from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import gradio as gr

SMOKE_MODE = os.environ.get("XVC_SMOKE_MODE") == "1"
SMOKE_MARKER = Path(
    os.environ.get("XVC_SMOKE_MARKER", "/tmp/xvc-gradio-callbacks.jsonl")
)


def _record(event: str, **payload) -> None:
    if not SMOKE_MODE:
        return
    SMOKE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    with SMOKE_MARKER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n")


def load_model_ui(progress=gr.Progress()) -> str:
    if SMOKE_MODE:
        _record("load_model")
        return "Тест: импорт X-VC уже проверен, тяжёлые веса в CI не загружаются."

    from runtime import load_model

    return load_model(progress=progress)


def convert_ui(
    source_path: str | None,
    reference_path: str | None,
    mode: str,
    current_ms: int,
    chunk_ms: int,
    future_ms: int,
    smooth_ms: int,
    progress=gr.Progress(),
):
    if not source_path:
        raise gr.Error("Добавьте исходную запись.")
    if not reference_path:
        raise gr.Error("Добавьте голос-референс.")

    controls = {
        "mode": mode,
        "current_ms": int(current_ms),
        "chunk_ms": int(chunk_ms),
        "future_ms": int(future_ms),
        "smooth_ms": int(smooth_ms),
    }

    if SMOKE_MODE:
        _record("convert", controls=controls)
        output = Path("/tmp/xvc-smoke-output.wav")
        shutil.copyfile(source_path, output)
        return str(output), "Тест: кнопка преобразования дошла до Python."

    from runtime import convert

    return convert(
        source_path=source_path,
        reference_path=reference_path,
        mode=mode,
        current_ms=int(current_ms),
        chunk_ms=int(chunk_ms),
        future_ms=int(future_ms),
        smooth_ms=int(smooth_ms),
        progress=progress,
    )


def clear_ui():
    _record("clear")
    return None, None, "Обычный", 300, 2400, 100, 20, None, "Очищено."


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="X-VC — преобразование голоса") as demo:
        gr.Markdown(
            "# X-VC\n"
            "Загрузите исходную речь и короткий пример нужного голоса. "
            "Для первого запуска нажмите «Загрузить модель»."
        )

        source = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Исходная речь",
        )
        reference = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Голос-референс",
        )

        mode = gr.Radio(
            choices=["Обычный", "Потоковый"],
            value="Обычный",
            label="Режим",
        )

        with gr.Accordion("Настройки потокового режима", open=False):
            current_ms = gr.Slider(
                100,
                1000,
                value=300,
                step=20,
                label="Текущий фрагмент, мс",
            )
            chunk_ms = gr.Slider(
                500,
                4000,
                value=2400,
                step=100,
                label="Общее окно, мс",
            )
            future_ms = gr.Slider(
                0,
                500,
                value=100,
                step=20,
                label="Будущий звук, мс",
            )
            smooth_ms = gr.Slider(
                0,
                200,
                value=20,
                step=10,
                label="Сглаживание, мс",
            )

        load_button = gr.Button("Загрузить модель", variant="secondary")
        convert_button = gr.Button("Преобразовать голос", variant="primary")
        clear_button = gr.Button("Очистить")

        output = gr.Audio(label="Результат", type="filepath")
        status = gr.Textbox(label="Состояние", value="Модель ещё не загружена.")

        load_button.click(
            fn=load_model_ui,
            outputs=status,
            api_name="load_model",
        )
        convert_button.click(
            fn=convert_ui,
            inputs=[
                source,
                reference,
                mode,
                current_ms,
                chunk_ms,
                future_ms,
                smooth_ms,
            ],
            outputs=[output, status],
            api_name="convert",
        )
        clear_button.click(
            fn=clear_ui,
            outputs=[
                source,
                reference,
                mode,
                current_ms,
                chunk_ms,
                future_ms,
                smooth_ms,
                output,
                status,
            ],
            api_name="clear",
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_demo()
    demo.queue(default_concurrency_limit=1).launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=args.port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
