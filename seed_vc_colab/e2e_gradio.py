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


def marker_versions() -> list[str]:
    if not MARKER.exists():
        return []
    versions: list[str] = []
    for line in MARKER.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        versions.append(payload["version"])
    return versions


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


def set_first_slider(panel) -> None:
    sliders = panel.locator('input[type="range"]')
    if not sliders.count():
        raise AssertionError("No visible Gradio slider found")
    sliders.first.evaluate(
        """el => {
            const min = Number(el.min || 0);
            const max = Number(el.max || 100);
            el.value = String(Math.min(max, min + 1));
            el.dispatchEvent(new Event('input', {bubbles: true}));
            el.dispatchEvent(new Event('change', {bubbles: true}));
        }"""
    )


def set_first_checkbox(panel) -> None:
    boxes = panel.locator('input[type="checkbox"]')
    if boxes.count():
        boxes.first.check(force=True)


def submit_current_interface(page: Page, version: str) -> None:
    panel = active_panel(page)
    file_inputs = panel.locator('input[type="file"]')
    if file_inputs.count() < 2:
        # Some Gradio versions keep inactive tab panels in a way that makes
        # role-based scoping unreliable. Fall back to the current visible half.
        file_inputs = page.locator('input[type="file"]')
        if file_inputs.count() < 2:
            raise AssertionError(
                "Seed-VC Gradio page has fewer than two file inputs"
            )
        if version == "v2" and file_inputs.count() >= 4:
            source_input = file_inputs.nth(2)
            reference_input = file_inputs.nth(3)
        else:
            source_input = file_inputs.nth(0)
            reference_input = file_inputs.nth(1)
    else:
        source_input = file_inputs.nth(0)
        reference_input = file_inputs.nth(1)

    source_input.set_input_files(str(SOURCE_WAV))
    reference_input.set_input_files(str(REFERENCE_WAV))
    set_first_slider(panel)
    set_first_checkbox(panel)

    submit = panel.get_by_role("button", name=re.compile(r"^Submit$", re.I))
    if not submit.count():
        submit = page.get_by_role("button", name=re.compile(r"^Submit$", re.I))
    if not submit.count():
        raise AssertionError("Gradio Submit button not found")
    submit.first.click(force=True)
    wait_for_callback(version)

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

        submit_current_interface(page, "v1")
        open_v2_tab(page)
        submit_current_interface(page, "v2")

        browser.close()

    if page_errors:
        raise AssertionError(f"Browser page errors: {page_errors}")
    versions = marker_versions()
    if versions.count("v1") != 1 or versions.count("v2") != 1:
        raise AssertionError(f"Unexpected backend callback record: {versions}")
    print(
        "GRADIO E2E OK: V1 and V2 upload/settings/Submit/Clear "
        "reached Python backend"
    )


if __name__ == "__main__":
    main()
