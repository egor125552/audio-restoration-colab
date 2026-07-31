from __future__ import annotations

import wave
from pathlib import Path

from playwright.sync_api import Page, expect


def test_russian_interface_and_model_switching(page: Page) -> None:
    console_errors: list[str] = []
    page.on(
        "console",
        lambda message: (
            console_errors.append(message.text)
            if message.type == "error"
            else None
        ),
    )
    page.on("pageerror", lambda error: console_errors.append(str(error)))

    page.goto("http://127.0.0.1:7860", wait_until="networkidle")

    expect(
        page.get_by_role(
            "heading",
            name="Восстановление и очистка аудио",
            exact=True,
        )
    ).to_be_visible()
    expect(
        page.get_by_role("button", name="4. Начать обработку", exact=True)
    ).to_be_visible()
    expect(page.get_by_role("status")).to_contain_text(
        "Ожидаю аудиофайл"
    )
    expect(
        page.get_by_text("Скачать лог последнего запуска", exact=True)
    ).to_be_visible()

    test_audio = _create_test_wave(Path("test-artifacts/input.wav"))
    page.locator("input[type='file']").set_input_files(str(test_audio))
    expect(page.get_by_text("input.wav", exact=False)).to_be_visible()

    model_box = page.get_by_label("2. Модель", exact=False)
    expect(model_box).to_be_visible()
    model_box.click()
    model_box.fill("Дорисовка, большая")
    page.get_by_role(
        "option",
        name="Дорисовка, большая",
        exact=True,
    ).click()
    expect(
        page.get_by_role(
            "heading",
            name="AudioSR — большая и медленная",
            exact=True,
        )
    ).to_be_visible()
    expect(page.get_by_text("Количество шагов", exact=True)).to_be_visible()

    artifact_dir = Path("test-artifacts")
    artifact_dir.mkdir(exist_ok=True)
    page.screenshot(
        path=str(artifact_dir / "gradio-interface.png"),
        full_page=True,
    )
    assert console_errors == []


def _create_test_wave(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(48_000)
        audio.writeframes(b"\x00\x00" * 4_800)
    return path.resolve()
