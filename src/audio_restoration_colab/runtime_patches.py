from __future__ import annotations

from typing import Any

_PATCH_MARKER = "_audio_restoration_native_roformer_segments"


def apply_audio_separator_quality_patches() -> None:
    """Use every RoFormer checkpoint's native temporal configuration.

    The generic UI exposes a segment control for older backends, but RoFormer
    checkpoints are trained with model-specific ``dim_t`` and STFT hop lengths.
    Overriding those values can shorten model output and damage both alignment
    and quality. The patch is optional because the lightweight Gradio process
    deliberately does not install the heavy separator environment.
    """

    try:
        from audio_separator.separator import Separator
    except ImportError:
        return

    original_init = Separator.__init__
    if getattr(original_init, _PATCH_MARKER, False):
        return

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        mdxc_params = dict(kwargs.get("mdxc_params") or {})
        mdxc_params["override_model_segment_size"] = False
        kwargs["mdxc_params"] = mdxc_params
        original_init(self, *args, **kwargs)

    setattr(patched_init, _PATCH_MARKER, True)
    Separator.__init__ = patched_init
