from __future__ import annotations

from dataclasses import dataclass

from .jobs import ROLE_TITLES
from .mixer import role_group

VOCAL_ROLES = frozenset({"vocals", "clean", "dry", "breaths", "speech"})
DRUM_ROLES = frozenset(
    {
        "drums",
        "kick",
        "snare",
        "toms",
        "cymbals",
        "hihat",
        "ride",
        "crash",
    }
)


@dataclass(frozen=True)
class ResultLayout:
    choices: tuple[tuple[str, str], ...]
    editor_visible: bool
    gain_visibility: dict[str, bool]
    other_gain_label: str
    show_all_preset: bool
    show_no_vocals_preset: bool
    show_only_vocals_preset: bool
    show_no_drums_preset: bool


def build_result_layout(roles: list[str] | tuple[str, ...]) -> ResultLayout:
    unique_roles = tuple(dict.fromkeys(str(role) for role in roles if role))
    choices = tuple(
        (ROLE_TITLES.get(role, role), role) for role in unique_roles
    )
    groups = {role_group(role) for role in unique_roles}
    has_vocals = any(role in VOCAL_ROLES for role in unique_roles)
    has_drums = any(role in DRUM_ROLES for role in unique_roles)
    other_roles = [
        role for role in unique_roles if role_group(role) == "other"
    ]

    if other_roles == ["instrumental"]:
        other_label = "Минусовка, %"
    elif other_roles == ["music"]:
        other_label = "Музыка, %"
    elif other_roles == ["sfx"]:
        other_label = "Звуковые эффекты, %"
    else:
        other_label = "Остальное, %"

    return ResultLayout(
        choices=choices,
        editor_visible=len(unique_roles) > 1,
        gain_visibility={
            "vocals": "vocals" in groups,
            "drums": "drums" in groups,
            "bass": "bass" in groups,
            "guitar": "guitar" in groups,
            "piano": "piano" in groups,
            "other": "other" in groups,
        },
        other_gain_label=other_label,
        show_all_preset=len(unique_roles) > 1,
        show_no_vocals_preset=has_vocals and len(unique_roles) > 1,
        show_only_vocals_preset=has_vocals,
        show_no_drums_preset=has_drums and len(unique_roles) > 1,
    )
