# ruff: noqa: I001
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr


home = Path(os.environ.get("SEED_VC_HOME", "/content/seed-vc")).resolve()
source = home / "src"
marker = Path(
    os.environ.get(
        "SEED_VC_SMOKE_MARKER",
        "/tmp/seed-vc-gradio-callbacks.jsonl",
    )
)

os.chdir(source)
sys.path.insert(0, str(source))
import app as seed_app  # noqa: E402


def record(
    version: str,
    source_audio: str | None,
    reference_audio: str | None,
    controls: dict[str, object],
) -> None:
    if not source_audio or not Path(source_audio).is_file():
        raise gr.Error("Smoke source audio did not reach the backend")
    if not reference_audio or not Path(reference_audio).is_file():
        raise gr.Error("Smoke reference audio did not reach the backend")
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "version": version,
                    "source": Path(source_audio).name,
                    "reference": Path(reference_audio).name,
                    "controls": controls,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def smoke_v1(
    source_audio_path,
    target_audio_path,
    diffusion_steps=10,
    length_adjust=1.0,
    inference_cfg_rate=0.7,
    f0_condition=False,
    auto_f0_adjust=True,
    pitch_shift=0,
    stream_output=True,
):
    record(
        "v1",
        source_audio_path,
        target_audio_path,
        {
            "diffusion_steps": diffusion_steps,
            "length_adjust": length_adjust,
            "inference_cfg_rate": inference_cfg_rate,
            "f0_condition": f0_condition,
            "auto_f0_adjust": auto_f0_adjust,
            "pitch_shift": pitch_shift,
            "stream_output": stream_output,
        },
    )
    yield source_audio_path, None
    yield None, source_audio_path


def smoke_v2(
    source_audio_path,
    target_audio_path,
    diffusion_steps=30,
    length_adjust=1.0,
    intelligebility_cfg_rate=0.0,
    similarity_cfg_rate=0.7,
    top_p=0.9,
    temperature=1.0,
    repetition_penalty=1.0,
    convert_style=False,
    anonymization_only=False,
    stream_output=True,
):
    record(
        "v2",
        source_audio_path,
        target_audio_path,
        {
            "diffusion_steps": diffusion_steps,
            "length_adjust": length_adjust,
            "intelligebility_cfg_rate": intelligebility_cfg_rate,
            "similarity_cfg_rate": similarity_cfg_rate,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "convert_style": convert_style,
            "anonymization_only": anonymization_only,
            "stream_output": stream_output,
        },
    )
    yield source_audio_path, None
    yield None, source_audio_path


# Keep the real upstream interface-building code and replace only the expensive
# inference callbacks. Chromium still drives the actual Gradio page and queue.
seed_app.convert_voice_v1_wrapper = smoke_v1
seed_app.convert_voice_v2_wrapper = smoke_v2

v1 = seed_app.create_v1_interface()
v2 = seed_app.create_v2_interface()

with gr.Blocks(title="Seed Voice Conversion smoke") as demo:
    gr.Markdown("# Seed Voice Conversion")
    with gr.Tabs():
        with gr.TabItem("V1 - Voice & Singing Voice Conversion"):
            v1.render()
        with gr.TabItem("V2 - Voice & Style Conversion"):
            v2.render()

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        share=False,
        show_error=True,
        quiet=False,
    )
