# ruff: noqa: I001
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


BASE_URL = os.environ.get("SEED_VC_SMOKE_URL", "http://127.0.0.1:7860")
MARKER = Path(
    os.environ.get(
        "SEED_VC_SMOKE_MARKER",
        "/tmp/seed-vc-gradio-callbacks.jsonl",
    )
)
SOURCE_WAV = Path(
    os.environ.get("SEED_VC_SMOKE_SOURCE", "/tmp/seed-vc-source.wav")
)
REFERENCE_WAV = Path(
    os.environ.get("SEED_VC_SMOKE_REFERENCE", "/tmp/seed-vc-reference.wav")
)

V1_CONTROLS = {
    "Diffusion Steps / 扩散步数": ("diffusion_steps", 11.0),
    "Length Adjust / 长度调整": ("length_adjust", 1.1),
    "Inference CFG Rate": ("inference_cfg_rate", 0.8),
    "Pitch shift / 音调变换": ("pitch_shift", 1.0),
}
V1_CHECKBOXES = {
    "Use F0 conditioned model / 启用F0输入": ("f0_condition", True),
    "Auto F0 adjust / 自动F0调整": ("auto_f0_adjust", False),
}
V2_CONTROLS = {
    "Diffusion Steps / 扩散步数": ("diffusion_steps", 31.0),
    "Length Adjust / 长度调整": ("length_adjust", 1.1),
    "Intelligibility CFG Rate": ("intelligebility_cfg_rate", 0.1),
    "Similarity CFG Rate": ("similarity_cfg_rate", 0.8),
    "Top-p": ("top_p", 0.8),
    "Temperature": ("temperature", 1.1),
    "Repetition Penalty": ("repetition_penalty", 1.1),
}
V2_CHECKBOXES = {
    "convert style/emotion/accent": ("convert_style", True),
    "anonymization only": ("anonymization_only", True),
}


def marker_records() -> list[dict[str, object]]:
    if not MARKER.exists():
        return []
    records: list[dict[str, object]] = []
    for line in MARKER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def marker_versions() -> list[str]:
    return [str(record["version"]) for record in marker_records()]


def wait_for_callback(version: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if version in marker_versions():
            return
        time.sleep(0.2)
    seen = marker_versions()
    raise AssertionError(
        f"Gradio backend callback {version!r} was not reached. Seen: {seen}"
    )


def active_panel(page: Page):
    panels = page.locator('[role="tabpanel"]:visible')
    if panels.count():
        return panels.last
    return page.locator("body")


def input_for_visible_label(page: Page, label: str, input_type: str):
    labels = page.get_by_text(label, exact=True)
    for index in range(labels.count()):
        label_node = labels.nth(index)
        if not label_node.is_visible():
            continue
        component = label_node.locator(
            f"xpath=ancestor::*[.//input[@type='{input_type}']][1]"
        )
        control = component.locator(f'input[type="{input_type}"]')
        if control.count():
            return control.first
    raise AssertionError(
        f"Visible {input_type} input not found for Gradio label {label!r}"
    )


def audio_file_input(page: Page, label: str):
    # Audio components replace their inner DOM after upload, so resolve each
    # field afresh by its visible label rather than by a numeric position.
    components = page.locator(".gradio-audio").filter(has_text=label)
    for index in range(components.count()):
        component = components.nth(index)
        if not component.is_visible():
            continue
        file_input = component.locator('input[type="file"]')
        if file_input.count():
            return file_input.first
    return input_for_visible_label(page, label, "file")


def set_slider(page: Page, label: str, value: float) -> None:
    slider = input_for_visible_label(page, label, "range")
    slider.evaluate(
        """(el, value) => {
            el.value = String(value);
            el.dispatchEvent(new Event('input', {bubbles: true}));
            el.dispatchEvent(new Event('change', {bubbles: true}));
        }""",
        value,
    )


def set_checkbox(page: Page, label: str, value: bool) -> None:
    checkbox = input_for_visible_label(page, label, "checkbox")
    if value:
        checkbox.check(force=True)
    else:
        checkbox.uncheck(force=True)


def exercise_controls(
    page: Page,
    sliders: dict[str, tuple[str, float]],
    checkboxes: dict[str, tuple[str, bool]],
) -> dict[str, object]:
    expected: dict[str, object] = {}
    for label, (backend_name, value) in sliders.items():
        set_slider(page, label, value)
        expected[backend_name] = value
    for label, (backend_name, value) in checkboxes.items():
        set_checkbox(page, label, value)
        expected[backend_name] = value
    return expected


def assert_backend_controls(version: str, expected: dict[str, object]) -> None:
    matches = [
        record for record in marker_records() if record.get("version") == version
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one {version} backend record, got {matches}")
    controls = matches[0].get("controls")
    if not isinstance(controls, dict):
        raise AssertionError(f"{version} backend did not record controls: {matches[0]}")

    for name, expected_value in expected.items():
        actual = controls.get(name)
        if isinstance(expected_value, float):
            if not isinstance(actual, (int, float)):
                raise AssertionError(
                    f"{version} control {name} is not numeric in backend: {actual!r}"
                )
            if abs(float(actual) - expected_value) > 1e-6:
                raise AssertionError(
                    f"{version} control {name} expected {expected_value}, got {actual}"
                )
        elif actual != expected_value:
            raise AssertionError(
                f"{version} control {name} expected {expected_value!r}, got {actual!r}"
            )


def submit_current_interface(
    page: Page,
    version: str,
    sliders: dict[str, tuple[str, float]],
    checkboxes: dict[str, tuple[str, bool]],
) -> None:
    panel = active_panel(page)
    expected = exercise_controls(page, sliders, checkboxes)

    audio_file_input(page, "Source Audio / 源音频").set_input_files(
        str(SOURCE_WAV)
    )
    audio_file_input(page, "Reference Audio / 参考音频").set_input_files(
        str(REFERENCE_WAV)
    )

    submit = panel.get_by_role("button", name=re.compile(r"^Submit$", re.I))
    if not submit.count():
        submit = page.get_by_role("button", name=re.compile(r"^Submit$", re.I))
    if not submit.count():
        raise AssertionError("Gradio Submit button not found")
    submit.first.click(force=True)
    wait_for_callback(version)
    assert_backend_controls(version, expected)

    clear = panel.get_by_role("button", name=re.compile(r"^Clear$", re.I))
    if not clear.count():
        clear = page.get_by_role("button", name=re.compile(r"^Clear$", re.I))
    if not clear.count():
        raise AssertionError("Gradio Clear button not found")
    clear.first.click(force=True)


def open_v2_tab(page: Page) -> None:
    tab = page.get_by_role(
        "tab",
        name=re.compile(r"^V2 - Voice & Style Conversion$"),
    )
    if tab.count():
        tab.first.click()
        return
    fallback = page.get_by_text("V2 - Voice & Style Conversion", exact=True)
    if not fallback.count():
        raise AssertionError("V2 Gradio tab not found")
    fallback.last.click()


def main() -> None:
    for audio in (SOURCE_WAV, REFERENCE_WAV):
        if not audio.is_file():
            raise AssertionError(f"Missing browser smoke audio: {audio}")
    MARKER.unlink(missing_ok=True)

    page_errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        page.goto(BASE_URL, wait_until="networkidle", timeout=60_000)

        if "Seed Voice Conversion" not in page.locator("body").inner_text():
            raise AssertionError(
                "Seed-VC heading is missing from rendered Gradio page"
            )

        submit_current_interface(page, "v1", V1_CONTROLS, V1_CHECKBOXES)
        open_v2_tab(page)
        submit_current_interface(page, "v2", V2_CONTROLS, V2_CHECKBOXES)

        browser.close()

    if page_errors:
        raise AssertionError(f"Browser page errors: {page_errors}")
    versions = marker_versions()
    if versions.count("v1") != 1 or versions.count("v2") != 1:
        raise AssertionError(f"Unexpected backend callback record: {versions}")
    print(
        "GRADIO E2E OK: V1/V2 WAV uploads, every visible setting, Submit and "
        "Clear reached the Python backend with changed values"
    )


if __name__ == "__main__":
    main()
