from __future__ import annotations

import json
import os
import time
import wave
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_URL = os.environ.get("QWEN_TRAIN_SMOKE_URL", "http://127.0.0.1:7860")
MARKER = Path(os.environ.get("QWEN_TRAIN_SMOKE_MARKER", "/tmp/qwen3-trainer-smoke.jsonl"))
SOURCE_DIR = Path("/tmp/qwen3-trainer-source")


def records() -> list[dict]:
    if not MARKER.exists():
        return []
    return [json.loads(x) for x in MARKER.read_text(encoding="utf-8").splitlines() if x.strip()]


def wait_for(event: str, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for item in records():
            if item.get("event") == event:
                return item
        time.sleep(0.2)
    raise AssertionError(f"Не дождался backend-события {event!r}. Есть: {records()}")


def make_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(b"\x00\x00" * 24000)


def main() -> None:
    MARKER.unlink(missing_ok=True)
    make_wav(SOURCE_DIR / "sample.wav")
    page_errors: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        page.goto(BASE_URL, wait_until="networkidle", timeout=60_000)

        body = page.locator("body").inner_text()
        if "Qwen3-TTS LoRA" not in body or "Обучение" not in body:
            raise AssertionError("Русский интерфейс не отрисовался.")

        page.get_by_label("Имя проекта").fill("Smoke Voice")
        page.get_by_label("Папка с исходными аудиофайлами").fill(str(SOURCE_DIR))

        page.get_by_role("button", name="Создать или открыть проект", exact=True).click()
        wait_for("create_project")
        page.get_by_role("button", name="Проверить папку", exact=True).click()
        wait_for("inspect_source")
        page.get_by_role("button", name="Скачать модели", exact=True).click()
        wait_for("download_models")

        page.get_by_role("tab", name="Датасет", exact=True).click()
        page.get_by_role("button", name="Подготовить датасет", exact=True).click()
        wait_for("prepare_dataset")

        page.get_by_role("tab", name="Обучение", exact=True).click()
        page.get_by_label("Всего эпох").fill("3")
        page.get_by_role("button", name="Начать обучение", exact=True).click()
        start = wait_for("start_training")
        if int(start.get("epochs", 0)) != 3:
            raise AssertionError(f"Изменённое число эпох не дошло до Python: {start}")
        page.get_by_role("button", name="Обновить состояние", exact=True).click()
        wait_for("refresh_status")
        page.get_by_role("button", name="Остановить обучение", exact=True).click()
        wait_for("stop_training")

        browser.close()

    if page_errors:
        raise AssertionError(f"Ошибки страницы: {page_errors}")

    expected = ["create_project", "inspect_source", "download_models", "prepare_dataset", "start_training", "refresh_status", "stop_training"]
    seen = [x.get("event") for x in records()]
    if seen != expected:
        raise AssertionError(f"Неожиданный порядок callback: {seen}")
    print("GRADIO E2E OK: все основные кнопки дошли до Python.")


if __name__ == "__main__":
    main()
