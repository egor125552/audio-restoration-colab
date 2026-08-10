from __future__ import annotations

from .studio_instrumental import STUDIO_INSTRUMENTAL_MODELS

_PATCH_MARKER = "_audio_restoration_studio_ensemble_progress"


def apply_studio_progress_patch() -> None:
    """Make the three-model studio ensemble progress monotonic and audible."""

    from .runtime import PROGRESS_PREFIX, TQDM_PERCENT, _WorkerProgressParser

    original_feed = _WorkerProgressParser.feed
    if getattr(original_feed, _PATCH_MARKER, False):
        return

    model_indexes = {
        filename: index
        for index, filename in enumerate(STUDIO_INSTRUMENTAL_MODELS, start=1)
    }
    total = len(STUDIO_INSTRUMENTAL_MODELS)

    def patched_feed(self, text: str) -> None:
        untouched: list[str] = []
        for fragment in text.replace("\r", "\n").splitlines():
            stripped = fragment.strip()
            if not stripped:
                continue

            if "Processing with model:" in stripped:
                matched_index = next(
                    (
                        index
                        for filename, index in model_indexes.items()
                        if filename in stripped
                    ),
                    None,
                )
                if matched_index is not None:
                    self._studio_pass_index = matched_index
                    self.progress(
                        self.last_overall,
                        (
                            "Студийная минусовка: "
                            f"модель {matched_index} из {total}…"
                        ),
                    )
                    continue

            pass_index = getattr(self, "_studio_pass_index", 0)
            match = TQDM_PERCENT.search(stripped)
            if (
                pass_index
                and match is not None
                and self.tqdm_span > 0
                and not stripped.startswith(PROGRESS_PREFIX)
            ):
                percent = int(match.group(1))
                ensemble_fraction = (
                    (pass_index - 1) + percent / 100.0
                ) / total
                local = self.local_base + self.tqdm_span * ensemble_fraction
                self._emit(
                    local,
                    (
                        "Студийная минусовка: "
                        f"модель {pass_index} из {total}, {percent}%"
                    ),
                )
                continue

            untouched.append(fragment)

        if untouched:
            original_feed(self, "\n".join(untouched))

    setattr(patched_feed, _PATCH_MARKER, True)
    _WorkerProgressParser.feed = patched_feed
