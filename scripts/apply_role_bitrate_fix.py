from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(path: str, fragments: tuple[str, ...]) -> None:
    text = (ROOT / path).read_text(encoding="utf-8")
    missing = [fragment for fragment in fragments if fragment not in text]
    if missing:
        raise RuntimeError(f"В {path} отсутствуют ожидаемые изменения: {missing}")


def main() -> None:
    require(
        "workers/separator_server.py",
        (
            "len(expected_roles) == 2",
            '"reverb": {"reverb", "reverberation", "echo", "wet"}',
            '"breathing"',
        ),
    )
    require(
        "src/audio_restoration_colab/outputs.py",
        (
            "def probe_audio_bitrate",
            "source_bitrate: int | None = None",
            '"stream=bit_rate:format=bit_rate"',
        ),
    )
    require(
        "src/audio_restoration_colab/jobs.py",
        (
            "probe_audio_bitrate(source)",
            "source_bitrate=source_bitrate",
        ),
    )


if __name__ == "__main__":
    main()
