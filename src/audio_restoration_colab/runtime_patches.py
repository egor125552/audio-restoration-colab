from __future__ import annotations

import gc
from typing import Any

from .studio_instrumental import (
    STUDIO_INSTRUMENTAL_ALGORITHM,
    STUDIO_INSTRUMENTAL_MODELS,
    STUDIO_INSTRUMENTAL_PRESET,
)

_INIT_PATCH_MARKER = "_audio_restoration_native_roformer_segments"
_LOAD_MODEL_PATCH_MARKER = "_audio_restoration_studio_release_between_models"
_DEMIX_PATCH_MARKER = "_audio_restoration_safe_short_roformer"


def apply_audio_separator_quality_patches() -> None:
    """Preserve native RoFormer quality and add the studio ensemble.

    RoFormer checkpoints are trained with model-specific ``dim_t`` and STFT hop
    lengths, so their native temporal configuration must not be replaced by the
    generic UI segment value. Some BS-RoFormer checkpoints use a window longer
    than the 12-second safety padding used by the app. audio-separator 0.44.5
    then computes a negative overlap-add start for very short clips. We pad only
    that internal model call past one complete native window and crop its stems
    back before audio-separator continues. Normal songs are untouched.

    ``instrumental_studio`` is a project-local preset. audio-separator already
    knows how to run arbitrary model lists sequentially and ensemble the output,
    so the patch translates that preset into three RoFormer checkpoints using
    ``median_fft`` without modifying the installed package data. Before each
    next studio model is loaded, the previous model object is explicitly
    released so two checkpoints do not need to coexist in VRAM during a switch.
    """

    try:
        from audio_separator.separator import Separator
        from audio_separator.separator.architectures.mdxc_separator import (
            MDXCSeparator,
        )
    except ImportError:
        return

    original_init = Separator.__init__
    if not getattr(original_init, _INIT_PATCH_MARKER, False):

        def patched_init(self, *args: Any, **kwargs: Any) -> None:
            studio_requested = (
                kwargs.get("ensemble_preset") == STUDIO_INSTRUMENTAL_PRESET
            )
            if studio_requested:
                kwargs["ensemble_preset"] = None
                kwargs["ensemble_algorithm"] = STUDIO_INSTRUMENTAL_ALGORITHM
                kwargs["ensemble_weights"] = None

            mdxc_params = dict(kwargs.get("mdxc_params") or {})
            mdxc_params["override_model_segment_size"] = False
            kwargs["mdxc_params"] = mdxc_params
            original_init(self, *args, **kwargs)

            if studio_requested:
                self.ensemble_preset = STUDIO_INSTRUMENTAL_PRESET
                self.ensemble_algorithm = STUDIO_INSTRUMENTAL_ALGORITHM
                self.ensemble_weights = None
                self._ensemble_preset_models = list(STUDIO_INSTRUMENTAL_MODELS)

        setattr(patched_init, _INIT_PATCH_MARKER, True)
        Separator.__init__ = patched_init

    original_load_model = Separator.load_model
    if not getattr(original_load_model, _LOAD_MODEL_PATCH_MARKER, False):

        def patched_load_model(self, model_filename=None) -> None:
            is_studio_switch = (
                self.ensemble_preset == STUDIO_INSTRUMENTAL_PRESET
                and model_filename is not None
                and self.model_instance is not None
            )
            if is_studio_switch:
                previous_model = self.model_instance
                self.model_instance = None
                clear_cache = getattr(previous_model, "clear_gpu_cache", None)
                if callable(clear_cache):
                    clear_cache()
                del previous_model
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
            return original_load_model(self, model_filename)

        setattr(patched_load_model, _LOAD_MODEL_PATCH_MARKER, True)
        Separator.load_model = patched_load_model

    original_demix = MDXCSeparator.demix
    if not getattr(original_demix, _DEMIX_PATCH_MARKER, False):

        def patched_demix(self, mix):
            if not getattr(self, "is_roformer", False):
                return original_demix(self, mix)

            config = self.model_data_cfgdict
            model_config = config.model
            hop_length = getattr(model_config, "stft_hop_length", None)
            if hop_length is None:
                hop_length = config.audio.hop_length
            chunk_size = int(hop_length) * (int(config.inference.dim_t) - 1)
            original_length = int(mix.shape[-1])
            if original_length > chunk_size:
                return original_demix(self, mix)

            import numpy as np

            sample_rate = max(1, int(config.audio.sample_rate))
            target_length = chunk_size + sample_rate
            padded_mix = np.pad(
                mix,
                ((0, 0), (0, target_length - original_length)),
                mode="constant",
            )
            separated = original_demix(self, padded_mix)
            if isinstance(separated, dict):
                return {
                    stem: values[..., :original_length]
                    for stem, values in separated.items()
                }
            return separated[..., :original_length]

        setattr(patched_demix, _DEMIX_PATCH_MARKER, True)
        MDXCSeparator.demix = patched_demix
