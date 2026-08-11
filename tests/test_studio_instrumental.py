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


def test_studio_preset_expands_to_three_models_and_releases_previous(
    monkeypatch,
) -> None:
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
    default_model = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

    class FakeLoadedModel:
        def __init__(self) -> None:
            self.cache_cleared = False

        def clear_gpu_cache(self) -> None:
            self.cache_cleared = True

    class FakeSeparator:
        def __init__(self, *args, **kwargs) -> None:
            self.received_args = args
            self.received_kwargs = dict(kwargs)
            self.ensemble_preset = kwargs.get("ensemble_preset")
            self.ensemble_algorithm = kwargs.get("ensemble_algorithm")
            self.ensemble_weights = kwargs.get("ensemble_weights")
            self._ensemble_preset_models = None
            self.model_instance = None
            self.loaded_model = None

        def load_model(self, model_filename=default_model) -> None:
            if (
                self._ensemble_preset_models is not None
                and model_filename == default_model
            ):
                model_filename = list(self._ensemble_preset_models)
            self.loaded_model = model_filename
            if isinstance(model_filename, list):
                return
            if model_filename is not None:
                self.model_instance = FakeLoadedModel()

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
    mdxc_params = studio.received_kwargs["mdxc_params"]
    assert mdxc_params["override_model_segment_size"] is False
    assert studio.ensemble_preset == STUDIO_INSTRUMENTAL_PRESET
    assert studio.ensemble_algorithm == STUDIO_INSTRUMENTAL_ALGORITHM
    assert studio.ensemble_weights is None
    assert studio._ensemble_preset_models == list(STUDIO_INSTRUMENTAL_MODELS)
    assert len(studio._ensemble_preset_models) == 3

    # The real worker calls load_model() without arguments for an ensemble.
    # Keep that default call intact so audio-separator expands the preset list.
    studio.load_model()
    assert studio.loaded_model == list(STUDIO_INSTRUMENTAL_MODELS)
    assert studio.model_instance is None

    previous = FakeLoadedModel()
    studio.model_instance = previous
    studio.load_model(STUDIO_INSTRUMENTAL_MODELS[1])
    assert previous.cache_cleared is True
    assert studio.loaded_model == STUDIO_INSTRUMENTAL_MODELS[1]
    assert studio.model_instance is not previous

    regular = FakeSeparator(ensemble_preset="instrumental_clean")
    assert regular.received_kwargs["ensemble_preset"] == "instrumental_clean"
    assert regular._ensemble_preset_models is None
    regular_previous = FakeLoadedModel()
    regular.model_instance = regular_previous
    regular.load_model("some-other-model.ckpt")
    assert regular_previous.cache_cleared is False
