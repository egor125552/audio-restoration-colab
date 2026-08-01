"""Русский Google Colab для очистки и дорисовки аудио."""

__version__ = "0.1.0"

from .top_models import apply_top_model_catalog as _apply_top_model_catalog

_apply_top_model_catalog()
del _apply_top_model_catalog
