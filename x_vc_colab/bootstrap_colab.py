from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path
from threading import Thread

ROOT = Path(__file__).resolve().parents[1]
XVC_HOME = Path(os.environ.get("XVC_HOME", "/content/x-vc")).resolve()
INSTALLER = ROOT / "x_vc_colab" / "install_xvc.sh"
INSTALL_LOG = Path("/content/xvc-install.log") if Path("/content").exists() else Path("/tmp/xvc-install.log")
ASSET_LOG = Path("/content/xvc-assets.log") if Path("/content").exists() else Path("/tmp/xvc-assets.log")
CONSOLE = sys.stdout


def say(percent: int, text: str) -> None:
    print(f"{percent}% — {text}", file=CONSOLE, flush=True)


def tail(path: Path, limit: int = 8000) -> str:
    if not path.exists():
        return "Лог не создан."
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


def run_installer() -> None:
    say(12, "устанавливаю Python и зависимости X-VC")
    INSTALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with INSTALL_LOG.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            ["bash", str(INSTALLER), str(XVC_HOME)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        started = time.monotonic()
        next_notice = started + 15
        while proc.poll() is None:
            time.sleep(1)
            now = time.monotonic()
            if now >= next_notice:
                elapsed = int(now - started)
                say(12, f"зависимости всё ещё устанавливаются, прошло {elapsed} с")
                next_notice = now + 15

    if proc.returncode != 0:
        raise RuntimeError(
            "Не удалось установить зависимости X-VC. Последние строки лога:\n" + tail(INSTALL_LOG)
        )
    say(45, "зависимости X-VC установлены")


def prepare_assets_with_progress() -> None:
    venv_python = XVC_HOME / ".venv" / "bin" / "python"
    if not venv_python.is_file():
        raise RuntimeError(f"Python X-VC не найден: {venv_python}")

    asset_helper = ROOT / "x_vc_colab" / "prepare_assets_cli.py"
    say(50, "начинаю подготовку файлов модели")
    proc = subprocess.Popen(
        [str(venv_python), "-u", str(asset_helper)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "XVC_HOME": str(XVC_HOME)},
    )
    assert proc.stdout is not None
    lines: list[str] = []
    for raw in proc.stdout:
        line = raw.rstrip()
        lines.append(line)
        if line.startswith("XVC_PROGRESS "):
            _, percent, message = line.split(" ", 2)
            say(int(percent), message)
        elif line.startswith("XVC_HEARTBEAT "):
            _, percent, message = line.split(" ", 2)
            say(int(percent), message)
    returncode = proc.wait()
    ASSET_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if returncode != 0:
        raise RuntimeError(
            "Не удалось подготовить файлы модели X-VC. Последние строки лога:\n"
            + "\n".join(lines[-80:])
        )
    say(98, "все файлы модели скачаны и конфигурация готова")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="Для CI: проверить установку без многогигабайтной загрузки весов.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    say(10, "код X-VC Colab получен")
    run_installer()
    if args.skip_assets:
        say(100, "CI-подготовка завершена без скачивания тяжёлых весов")
        return
    prepare_assets_with_progress()
    say(100, "X-VC полностью подготовлена; можно запускать интерфейс")


if __name__ == "__main__":
    main()
