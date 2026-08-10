from __future__ import annotations

import json

from audio_restoration_colab.runtime import (
    PROGRESS_PREFIX,
    _WorkerProgressParser,
)
from audio_restoration_colab.studio_instrumental import (
    STUDIO_INSTRUMENTAL_MODELS,
)


def test_studio_ensemble_progress_counts_all_three_models() -> None:
    events: list[tuple[float, str]] = []
    parser = _WorkerProgressParser(
        lambda fraction, message: events.append((fraction, message))
    )
    parser.feed(
        PROGRESS_PREFIX
        + json.dumps(
            {
                "fraction": 0.24,
                "message": "Разделитель: анализирую песню —",
                "tqdm_span": 0.68,
            }
        )
    )

    for index, filename in enumerate(STUDIO_INSTRUMENTAL_MODELS, start=1):
        parser.feed(f"INFO Processing with model: {filename}")
        parser.feed("50%|#####     |")
        parser.feed("100%|##########|")
        assert any(f"модель {index} из 3" in message for _, message in events)

    fractions = [fraction for fraction, _ in events]
    assert fractions == sorted(fractions)
    assert any("модель 1 из 3, 50%" in message for _, message in events)
    assert any("модель 2 из 3, 50%" in message for _, message in events)
    assert any("модель 3 из 3, 100%" in message for _, message in events)
