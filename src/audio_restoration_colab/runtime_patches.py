from __future__ import annotations

from typing import Any

_INIT_PATCH_MARKER = "_audio_restoration_native_roformer_segments"
_DEMIX_PATCH_MARKER = "_audio_restoration_safe_short_roformer"


def apply_audio_separator_quality_patches() -> None:
    """Preserve native RoFormer quality and handle clips shorter than a window.

    RoFormer checkpoints are trained with model-specific ``dim_t`` and STFT hop
    lengths, so their native temporal configuration must not be replaced by the
    generic UI segment value. Some BS-RoFormer checkpoints use a window longer
    than the 12-second safety padding used by the app. audio-separator 0.44.5
    then computes a negative overlap-add start for very short clips. We pad only
    that internal model call past one complete native window and crop its stems
    back before audio-separator continues. Normal songs are untouched.
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
            mdxc_params = dict(kwargs.get("mdxc_params") or {})
            mdxc_params["override_model_segment_size"] = False
            kwargs["mdxc_params"] = mdxc_params
            original_init(self, *args, **kwargs)

        setattr(patched_init, _INIT_PATCH_MARKER, True)
        Separator.__init__ = patched_init

    original_demix = MDXCSeparator.demix
    if getattr(original_demix, _DEMIX_PATCH_MARKER, False):
        return

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
