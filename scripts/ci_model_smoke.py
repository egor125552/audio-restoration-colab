from __future__ import annotations

import argparse
import os
from pathlib import Path

from audio_restoration_colab.catalog import MODEL_SPECS
from audio_restoration_colab.jobs import AudioJobService
from audio_restoration_colab.runtime import RuntimeLayout, SubprocessWorker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", choices=MODEL_SPECS)
    parser.add_argument("audio", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cache = Path(os.environ["AUDIO_RESTORATION_CACHE"])
    settings = {item.key: item.default for item in MODEL_SPECS[args.model_id].settings}
    service = AudioJobService(
        jobs_root=cache / "jobs",
        worker=SubprocessWorker(
            layout=RuntimeLayout(project_root=root, cache_root=cache)
        ),
    )
    result = service.process(
        source=args.audio,
        model_id=args.model_id,
        format_choice="wav",
        raw_settings=settings,
        progress=lambda value, message: print(f"{value:.0%}: {message}", flush=True),
    )
    assert result.files and all(path.stat().st_size > 44 for path in result.files)
    print(f"{args.model_id}: настоящий инференс готов", flush=True)


if __name__ == "__main__":
    main()
