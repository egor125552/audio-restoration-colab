from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from threading import Thread

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "x_vc_colab"))

from runtime import ASSETS_DIR, CHECKPOINT_DIR, RUNTIME_CONFIG, prepare_assets

state = {
    "percent": 50,
    "message": "подготавливаю загрузку файлов модели",
}
result = {"value": None, "error": None}


def emit(kind: str, percent: int, message: str) -> None:
    print(f"{kind} {percent} {message}", flush=True)


class TextProgress:
    def __call__(self, value: float, desc: str | None = None, **_kwargs) -> None:
        # prepare_assets сообщает начало каждого реального этапа. Эти проценты —
        # прогресс по этапам, а не выдуманная оценка скорости сети.
        mapping = [
            (0.05, 55),
            (0.30, 70),
            (0.58, 84),
            (0.75, 94),
            (0.90, 97),
        ]
        percent = 50
        for threshold, mapped in mapping:
            if value >= threshold:
                percent = mapped
        message = desc or "подготавливаю файлы модели"
        state["percent"] = percent
        state["message"] = message
        emit("XVC_PROGRESS", percent, message)


def worker() -> None:
    try:
        result["value"] = prepare_assets(progress=TextProgress())
    except BaseException as exc:  # передаём ошибку обратно в основной поток
        result["error"] = exc


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def human_bytes(value: int) -> str:
    units = ["Б", "КБ", "МБ", "ГБ", "ТБ"]
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.1f} {unit}"
        number /= 1024
    return f"{number:.1f} ТБ"


def main() -> None:
    thread = Thread(target=worker, daemon=True)
    thread.start()
    started = time.monotonic()
    next_notice = started + 15

    while thread.is_alive():
        thread.join(timeout=1)
        now = time.monotonic()
        if thread.is_alive() and now >= next_notice:
            elapsed = int(now - started)
            emit(
                "XVC_HEARTBEAT",
                int(state["percent"]),
                f"всё ещё: {state['message']}; прошло {elapsed} с",
            )
            next_notice = now + 15

    if result["error"] is not None:
        raise result["error"]

    total = directory_size(CHECKPOINT_DIR) + directory_size(ASSETS_DIR)
    if not RUNTIME_CONFIG.is_file():
        raise RuntimeError("Конфигурация X-VC не была создана.")
    emit("XVC_PROGRESS", 99, f"файлы модели занимают примерно {human_bytes(total)}")


if __name__ == "__main__":
    main()
