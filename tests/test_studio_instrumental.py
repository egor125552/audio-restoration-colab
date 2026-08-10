from __future__ import annotations

import sys
import types

from audio_restoration_colab.catalog import MODEL_SPECS
from audio_restoration_colab.runtime_patches import (
    apply_audio_separator_quality_patches,
)
from audio_restoration_colab.studio_instrumental import (
    STUDIO_INSTRUMENTAL_ALGORITHM,
    STUDIO_INSTRUMENTAL_MODELS,
    STUDIO_INSTRUMENTAL_PRESET,
)


def test_studio_instrumental_is_next_to_clean_mode() -> None:
    model_ids = list(MODEL_SPECS)
    clean_index = model_ids.index("stems_instrumental_clean")
    studio_index = model_ids.index("stems_instrumental_studio")

    assert studio_index == clean_index + 1
    model = MODEL_SPECS["stems_instrumental_studio"]
    assert model.ensemble_preset == STUDIO_INSTRUMENTAL_PRESET
    assert model.output_roles == ("instrumental", "vocals")
    assert "трёх" in model.title
    assert "медиан" in model.description.lower()


def test_studio_preset_expands_to_three_models(monkeypatch) -> None:
    audio_separator_module = types.ModuleType("audio_separator")
    audio_separator_module.__path__ = []
    separator_module = types.ModuleType("audio_separator.separator")
    separator_module.__path__ = []
    architectures_module = types.ModuleType(
        "audio_separator.separator.architectures"
    )
    architectures_module.__path__ = []
    mdxc_module = types.ModuleType(
        "audio_separator.separator.architectures.mdxc_separator"
    )

    class FakeSeparator:
        def __init__(self, *args, **kwargs) -> None:
            self.received_args = args
            self.received_kwargs = dict(kwargs)
            self.ensemble_preset = kwargs.get("ensemble_preset")
            self.ensemble_algorithm = kwargs.get("ensemble_algorithm")
            self.ensemble_weights = kwargs.get("ensemble_weights")
            self._ensemble_preset_models = None

    class FakeMDXCSeparator:
        def demix(self, mix):
            return mix

    separator_module.Separator = FakeSeparator
    mdxc_module.MDXCSeparator = FakeMDXCSeparator
    audio_separator_module.separator = separator_module
    separator_module.architectures = architectures_module
    architectures_module.mdxc_separator = mdxc_module

    monkeypatch.setitem(sys.modules, "audio_separator", audio_separator_module)
    monkeypatch.setitem(
        sys.modules,
        "audio_separator.separator",
        separator_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "audio_separator.separator.architectures",
        architectures_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "audio_separator.separator.architectures.mdxc_separator",
        mdxc_module,
    )

    apply_audio_separator_quality_patches()

    studio = FakeSeparator(
        ensemble_preset=STUDIO_INSTRUMENTAL_PRESET,
        mdxc_params={"override_model_segment_size": True},
    )
    assert studio.received_kwargs["ensemble_preset"] is None
    assert studio.received_kwargs["ensemble_algorithm"] == "median_fft"
    assert studio.received_kwargs["ensemble_weights"] is None
    assert studio.received_kwargs["mdxc_params"]["override_model_segment_size"] is False
    assert studio.ensemble_preset == STUDIO_INSTRUMENTAL_PRESET
    assert studio.ensemble_algorithm == STUDIO_INSTRUMENTAL_ALGORITHM
    assert studio.ensemble_weights is None
    assert studio._ensemble_preset_models == list(STUDIO_INSTRUMENTAL_MODELS)
    assert len(studio._ensemble_preset_models) == 3

    regular = FakeSeparator(ensemble_preset="instrumental_clean")
    assert regular.received_kwargs["ensemble_preset"] == "instrumental_clean"
    assert regular._ensemble_preset_models is None
