from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import gradio as gr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=("v1", "v2", "both"), default="v2")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    home = Path(os.environ.get("SEED_VC_HOME", "/content/seed-vc")).resolve()
    source = home / "src"
    if not source.is_dir():
        raise SystemExit(
            f"Seed-VC is not installed at {source}. Run install_seed_vc.sh first."
        )

    os.chdir(source)
    sys.path.insert(0, str(source))
    import app as seed_app

    interfaces: list[tuple[str, gr.Interface]] = []
    if args.version in {"v2", "both"}:
        model_args = argparse.Namespace(compile=args.compile)
        print("Loading the actual Seed-VC V2 checkpoints...")
        seed_app.vc_wrapper_v2 = seed_app.load_v2_models(model_args)
        interfaces.append(
            ("V2 - Voice & Style Conversion", seed_app.create_v2_interface())
        )
    if args.version in {"v1", "both"}:
        # V1 keeps upstream's lazy loading behavior: its weights are downloaded
        # on the first conversion request.
        interfaces.append(
            ("V1 - Voice & Singing Voice Conversion", seed_app.create_v1_interface())
        )

    with gr.Blocks(title="Seed Voice Conversion") as demo:
        gr.Markdown("# Seed Voice Conversion")
        if len(interfaces) == 1:
            interfaces[0][1].render()
        else:
            with gr.Tabs():
                for tab_name, interface in interfaces:
                    with gr.TabItem(tab_name):
                        interface.render()

    demo.queue().launch(
        share=not args.no_share,
        show_error=True,
        debug=True,
    )


if __name__ == "__main__":
    main()
