from __future__ import annotations

import os
import sys
from pathlib import Path

app = Path(__file__).with_name("app.py")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
print("Запускаю интерфейс без скрытия stdout/stderr.", flush=True)
print("Ячейка останется занятой, но весь вывод и прогресс обучения будут видны ниже.", flush=True)
os.execv(sys.executable, [sys.executable, "-u", str(app)])
