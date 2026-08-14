from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


PUBLIC_URL_RE = re.compile(r"https://[A-Za-z0-9.-]+\.gradio\.live")


def stop_pid_file(pid_file: Path) -> None:
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        pid_file.unlink(missing_ok=True)
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        pass
    else:
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
    pid_file.unlink(missing_ok=True)


def read_log(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    xvc_home = Path(os.environ.get("XVC_HOME", "/content/x-vc"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=str(xvc_home / ".venv" / "bin" / "python"))
    parser.add_argument("--app", default=str(here / "app.py"))
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--log", default="/content/xvc-gradio.log")
    parser.add_argument("--pid-file", default="/content/xvc-gradio.pid")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python = Path(args.python)
    app = Path(args.app)
    log_path = Path(args.log)
    pid_file = Path(args.pid_file)

    if not python.is_file():
        print(f"X-VC Python не найден: {python}", file=sys.stderr)
        return 2
    if not app.is_file():
        print(f"Gradio-приложение X-VC не найдено: {app}", file=sys.stderr)
        return 2

    stop_pid_file(pid_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    with log_path.open("a", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            [
                str(python),
                "-u",
                str(app),
                "--share",
                "--port",
                str(args.port),
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )

    pid_file.write_text(str(proc.pid), encoding="utf-8")
    print("Запускаю интерфейс X-VC. Жду публичную ссылку...", flush=True)

    deadline = time.time() + args.timeout
    while time.time() < deadline:
        text = read_log(log_path)
        match = PUBLIC_URL_RE.search(text)
        if match:
            url = match.group(0)
            print("X-VC интерфейс готов.", flush=True)
            print(f"XVC_PUBLIC_URL={url}", flush=True)
            return 0

        return_code = proc.poll()
        if return_code is not None:
            print(
                f"Gradio завершился до появления публичной ссылки, код {return_code}.",
                file=sys.stderr,
            )
            tail = text[-6000:] if text else "Лог пуст."
            print(tail, file=sys.stderr)
            pid_file.unlink(missing_ok=True)
            return 1

        time.sleep(1)

    text = read_log(log_path)
    tail = text[-6000:] if text else "Лог пуст."
    print(
        f"Gradio не выдал публичную ссылку за {args.timeout} секунд.",
        file=sys.stderr,
    )
    print(tail, file=sys.stderr)
    stop_pid_file(pid_file)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
