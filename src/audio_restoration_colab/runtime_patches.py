from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

_INIT_PATCH_MARKER = "_audio_restoration_native_roformer_segments"
_DEMIX_PATCH_MARKER = "_audio_restoration_safe_short_roformer"
_SEPARATE_PATCH_MARKER = "_audio_restoration_canonical_output_names"

_ROLE_ALIASES = {
    "vocals": {"vocals", "vocal", "voice"},
    "instrumental": {
        "instrumental",
        "karaoke",
        "no vocals",
        "no vocal",
        "novocals",
    },
    "drums": {"drums", "drum"},
    "bass": {"bass"},
    "guitar": {"guitar"},
    "piano": {"piano", "keys"},
    "other": {"other", "non vocals"},
    "dry": {"dry", "dereverb", "no reverb", "no echo"},
    "reverb": {"reverb", "echo"},
    "clean": {
        "clean",
        "dry",
        "no bleed",
        "no aspiration",
        "no noise",
    },
    "noise": {"noise"},
    "bleed": {"bleed"},
    "breaths": {"aspiration", "breath", "breaths"},
    "kick": {"kick", "bombo"},
    "snare": {"snare", "redoblante"},
    "toms": {"toms", "tom"},
    "cymbals": {"cymbals", "platillos"},
    "hihat": {"hihat", "hi hat", "hh"},
    "ride": {"ride"},
    "crash": {"crash"},
    "speech": {"speech", "dialog", "dialogue"},
    "music": {"music"},
    "sfx": {"sfx", "effect", "effects"},
}


def apply_audio_separator_quality_patches() -> None:
    """Patch audio-separator for quality, short clips, and stable stem names.

    RoFormer checkpoints are trained with model-specific ``dim_t`` and STFT hop
    lengths, so their native temporal configuration must not be replaced by the
    generic UI segment value. Some checkpoints use a window longer than the
    safety padding used by the app, so very short clips are padded internally.

    audio-separator also appends the model or ensemble name to output files.
    Names such as ``(Vocals)_preset_instrumental_full.wav`` previously made the
    app mistake the vocal track for an instrumental. The separator wrapper now
    copies returned files to short canonical names before role detection.
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

    original_separate = Separator.separate
    if not getattr(original_separate, _SEPARATE_PATCH_MARKER, False):

        def patched_separate(self, *args: Any, **kwargs: Any):
            raw_paths = original_separate(self, *args, **kwargs)
            output_dir = Path(str(getattr(self, "output_dir", "."))).resolve()
            return canonicalize_separator_outputs(raw_paths, output_dir)

        setattr(patched_separate, _SEPARATE_PATCH_MARKER, True)
        Separator.separate = patched_separate

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


def canonicalize_separator_outputs(
    raw_paths: Any,
    output_dir: Path,
) -> Any:
    """Return short role-bearing paths without model or preset words."""

    if not raw_paths:
        return raw_paths
    was_single = isinstance(raw_paths, (str, Path))
    paths = [raw_paths] if was_single else list(raw_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized: list[str] = []

    for index, raw_path in enumerate(paths):
        path = Path(str(raw_path))
        candidate = path if path.is_absolute() else output_dir / path
        if not candidate.is_file():
            fallback = output_dir / path.name
            if fallback.is_file():
                candidate = fallback
        if not candidate.is_file():
            normalized.append(str(raw_path))
            continue

        role = output_role_from_name(candidate)
        if role is None:
            normalized.append(str(candidate.resolve()))
            continue

        target = output_dir / f"mapped-{index:02d}-{role}.wav"
        if candidate.resolve() != target.resolve():
            target.unlink(missing_ok=True)
            shutil.copy2(candidate, target)
        normalized.append(str(target.resolve()))

    if was_single:
        return normalized[0]
    return normalized


def output_role_from_name(path: Path) -> str | None:
    """Read the actual stem marker, ignoring model names after it."""

    parenthesized = re.findall(r"\(([^()]*)\)", path.stem)
    if parenthesized:
        label = _normalize_role_label(parenthesized[-1])
        return _role_for_label(label)

    label = _normalize_role_label(path.stem)
    for role, aliases in _ROLE_ALIASES.items():
        for alias in aliases:
            if label == alias or label.endswith(f" {alias}"):
                return role
    return None


def _role_for_label(label: str) -> str | None:
    for role, aliases in _ROLE_ALIASES.items():
        if label in aliases:
            return role
    return None


def _normalize_role_label(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())
