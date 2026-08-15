from __future__ import annotations

import codecs
import os
import subprocess
import sys
from collections.abc import Callable, Sequence


def run_streamed(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    write: Callable[[str], object] | None = None,
) -> int:
    """Run a child process and relay stdout/stderr through the current Python kernel.

    Colab captures the notebook kernel's Python stdout reliably, but output inherited
    directly by a grandchild process can be hidden. We therefore pipe the child's
    bytes back into this process and immediately write them to sys.stdout.

    Chunks are streamed as they arrive, so carriage returns used by tqdm progress
    bars are preserved instead of being converted into hundreds of separate lines.
    """
    if write is None:
        write = sys.stdout.write

    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    child_env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=None,
        bufsize=0,
        env=child_env,
    )
    assert process.stdout is not None

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            text = decoder.decode(chunk)
            if text:
                write(text)
                sys.stdout.flush()
        tail = decoder.decode(b"", final=True)
        if tail:
            write(tail)
            sys.stdout.flush()
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise


def _self_test() -> None:
    captured: list[str] = []
    code = (
        "import sys,time; "
        "sys.stdout.write('Эпоха 1/2 10%\\r'); sys.stdout.flush(); "
        "sys.stdout.write('Эпоха 1/2 100%\\n'); sys.stdout.flush()"
    )
    rc = run_streamed([sys.executable, "-u", "-c", code], write=captured.append)
    text = "".join(captured)
    if rc != 0 or "10%\r" not in text or "100%\n" not in text:
        raise AssertionError(f"Потоковый вывод повреждён: rc={rc}, text={text!r}")
    print("COLAB STREAM RELAY OK")


if __name__ == "__main__":
    _self_test()
