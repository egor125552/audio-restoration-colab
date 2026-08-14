from __future__ import annotations

import json
import os
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_URL = os.environ.get("XVC_SMOKE_URL", "http://127.0.0.1:7860")
MARKER = Path(os.environ.get("XVC_SMOKE_MARKER", "/tmp/xvc-gradio-callbacks.jsonl"))
SOURCE_WAV = Path(os.environ.get("XVC_SMOKE_SOURCE", "/tmp/xvc-source.wav"))
REFERENCE_WAV = Path(os.environ.get("XVC_SMOKE_REFERENCE", "/tmp/xvc-reference.wav"))


def records() -> list[dict]:
    if not MARKER.exists():
        return []
    return [
        json.loads(line)
        for line in MARKER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def wait_for(event: str, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for item in records():
            if item.get("event") == event:
                return item
        time.sleep(0.2)
    raise AssertionError(f"Backend event {event!r} did not arrive. Seen: {records()}")


def set_audio(page, label: str, path: Path) -> None:
    component = page.locator(".gradio-audio").filter(has_text=label).first
    file_input = component.locator('input[type="file"]')
    if not file_input.count():
        raise AssertionError(f"File input not found for {label!r}")
    file_input.set_input_files(str(path))


def main() -> None:
    MARKER.unlink(missing_ok=True)
    for audio in (SOURCE_WAV, REFERENCE_WAV):
        if not audio.is_file():
            raise AssertionError(f"Missing smoke WAV: {audio}")

    page_errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        page.goto(BASE_URL, wait_until="networkidle", timeout=60_000)

        body = page.locator("body").inner_text()
        if "X-VC" not in body or "Исходная речь" not in body or "Голос-референс" not in body:
            raise AssertionError("Important X-VC controls are missing from the rendered page.")

        page.get_by_role("button", name="Загрузить модель", exact=True).click()
        wait_for("load_model")

        set_audio(page, "Исходная речь", SOURCE_WAV)
        set_audio(page, "Голос-референс", REFERENCE_WAV)

        page.get_by_text("Потоковый", exact=True).click()
        slider = page.get_by_label("Текущий фрагмент, мс")
        slider.fill("400")

        page.get_by_role("button", name="Преобразовать голос", exact=True).click()
        record = wait_for("convert")
        controls = record.get("controls", {})
        if controls.get("mode") != "Потоковый" or int(controls.get("current_ms", 0)) != 400:
            raise AssertionError(f"Changed controls did not reach Python: {record}")

        page.get_by_role("button", name="Очистить", exact=True).click()
        wait_for("clear")

        browser.close()

    if page_errors:
        raise AssertionError(f"Browser page errors: {page_errors}")

    events = [item.get("event") for item in records()]
    if events != ["load_model", "convert", "clear"]:
        raise AssertionError(f"Unexpected backend event order: {events}")

    print("GRADIO E2E OK: uploads and all three buttons reached Python.")
