from __future__ import annotations

import codecs
import os
import selectors
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence


def _port_is_ready(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.05):
            return True
    except OSError:
        return False


def run_streamed(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    write: Callable[[str], object] | None = None,
    ready_port: int | None = None,
    on_ready: Callable[[], object] | None = None,
) -> int:
    """Run a child process and relay stdout/stderr through the current Python kernel.

    Colab captures the notebook kernel's Python stdout reliably, but output inherited
    directly by a grandchild process can be hidden. We therefore pipe the child's
    bytes back into this process and immediately write them to sys.stdout.

    If ready_port and on_ready are provided, the same main thread watches the local
    server port and invokes on_ready once it becomes reachable. This lets Colab show
    its own port-proxy iframe without losing the live stdout stream.

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
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    ready_called = False
    eof = False

    try:
        while True:
            for key, _ in selector.select(timeout=0.1):
                chunk = os.read(key.fileobj.fileno(), 4096)
                if not chunk:
                    eof = True
                    selector.unregister(key.fileobj)
                    break
                text = decoder.decode(chunk)
                if text:
                    write(text)
                    sys.stdout.flush()

            if not ready_called and ready_port is not None and _port_is_ready(ready_port):
                ready_called = True
                if on_ready is not None:
                    on_ready()

            if eof and process.poll() is not None:
                break

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
    finally:
        selector.close()


def _self_test() -> None:
    captured: list[str] = []
    ready: list[bool] = []

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    code = (
        "import socket,sys; "
        f"s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(('127.0.0.1',{port})); s.listen(1); "
        "sys.stdout.write('Эпоха 1/2 10%\\r'); sys.stdout.flush(); "
        "c,_=s.accept(); c.close(); s.close(); "
        "sys.stdout.write('Эпоха 1/2 100%\\n'); sys.stdout.flush()"
    )
    rc = run_streamed(
        [sys.executable, "-u", "-c", code],
        write=captured.append,
        ready_port=port,
        on_ready=lambda: ready.append(True),
    )
    text = "".join(captured)
    if rc != 0 or "10%\r" not in text or "100%\n" not in text or ready != [True]:
        raise AssertionError(f"Потоковый вывод повреждён: rc={rc}, ready={ready}, text={text!r}")
    print("COLAB STREAM RELAY + PORT READY OK")


if __name__ == "__main__":
    _self_test()
