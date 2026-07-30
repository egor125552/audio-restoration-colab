from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .catalog import (
    MODEL_SPECS,
    default_browser_settings,
    get_model,
    normalize_settings,
)

DEFAULT_MODEL_ID = "flashsr_medium"


@dataclass(frozen=True)
class SelectionView:
    information: str
    visible_panels: dict[str, bool]
    values: dict[str, Any]


def build_model_information(model_id: str) -> str:
    model = get_model(model_id)
    return (
        f"## {model.title}\n\n"
        f"**Назначение:** {model.purpose}\n\n"
        f"**Размер:** {model.size_text}\n\n"
        f"{model.description}\n\n"
        f"**Важно:** {model.warning}"
    )


def selection_view(
    model_id: str,
    browser_state: Mapping[str, Any] | None,
) -> SelectionView:
    if model_id not in MODEL_SPECS:
        model_id = DEFAULT_MODEL_ID
    state = _normalized_browser_state(browser_state)
    denoise_id = (
        model_id
        if model_id in {"denoise_normal", "denoise_aggressive"}
        else "denoise_normal"
    )
    return SelectionView(
        information=build_model_information(model_id),
        visible_panels={
            "denoise": model_id.startswith("denoise_"),
            "lavasr": model_id == "lavasr_small",
            "flashsr": model_id == "flashsr_medium",
            "audiosr": model_id == "audiosr_large",
        },
        values={
            "denoise_quality": state[denoise_id]["quality"],
            "denoise_segment": state[denoise_id]["segment"],
            "lavasr_input_rate": state["lavasr_small"]["input_rate"],
            "lavasr_denoise": state["lavasr_small"]["denoise"],
            "lavasr_batch": state["lavasr_small"]["batch"],
            "flashsr_lowpass": state["flashsr_medium"]["lowpass"],
            "audiosr_mode": state["audiosr_large"]["mode"],
            "audiosr_steps": state["audiosr_large"]["steps"],
            "audiosr_guidance": state["audiosr_large"]["guidance"],
            "audiosr_seed": state["audiosr_large"]["seed"],
            "audiosr_lowpass": state["audiosr_large"]["lowpass"],
        },
    )


def merge_active_settings(
    browser_state: Mapping[str, Any] | None,
    model_id: str,
    visible_values: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    state = _normalized_browser_state(browser_state)
    if model_id not in MODEL_SPECS:
        return state
    current = dict(state[model_id])
    current.update(visible_values)
    state[model_id] = normalize_settings(model_id, current)
    return state


def _normalized_browser_state(
    browser_state: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    defaults = default_browser_settings()
    if not isinstance(browser_state, Mapping):
        return defaults
    result = deepcopy(defaults)
    for model_id in MODEL_SPECS:
        raw = browser_state.get(model_id)
        if isinstance(raw, Mapping):
            result[model_id] = normalize_settings(model_id, raw)
    return result
