"""Русский Google Colab для очистки и дорисовки аудио."""

__version__ = "0.1.0"

from .progress_patches import (
    apply_studio_progress_patch as _apply_studio_progress_patch,
)
from .runtime_patches import (
    apply_audio_separator_quality_patches as _apply_audio_separator_quality_patches,
)
from .top_models import apply_top_model_catalog as _apply_top_model_catalog

_apply_top_model_catalog()
_apply_audio_separator_quality_patches()
_apply_studio_progress_patch()
del _apply_top_model_catalog
del _apply_audio_separator_quality_patches
del _apply_studio_progress_patch
