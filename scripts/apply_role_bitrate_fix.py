from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Не найден ожидаемый фрагмент в {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_region(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    if replacement in text:
        return
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"Не найден заменяемый блок в {path}")
    path.write_text(updated, encoding="utf-8")


def patch_separator_server() -> None:
    path = ROOT / "workers" / "separator_server.py"
    replace_once(
        path,
        '''    if len(expected_roles) == 1 and not assignments and unused:\n        assignments[expected_roles[0]] = unused.pop(0)\n\n    missing = [role for role in expected_roles if role not in assignments]\n''',
        '''    if len(expected_roles) == 1 and not assignments and unused:\n        assignments[expected_roles[0]] = unused.pop(0)\n\n    # Many two-stem RoFormer checkpoints name the residual track simply\n    # ``other``. Once one side is identified unambiguously, the sole remaining\n    # file must be the configured counterpart (for example dry + reverb,\n    # clean + bleed, or clean + breaths). We never guess when neither side is\n    # recognized, and we never apply this fallback to models with 3+ stems.\n    if (\n        len(expected_roles) == 2\n        and len(assignments) == 1\n        and len(unused) == 1\n    ):\n        missing_role = next(\n            role for role in expected_roles if role not in assignments\n        )\n        assignments[missing_role] = unused.pop(0)\n\n    missing = [role for role in expected_roles if role not in assignments]\n''',
    )
    replace_once(
        path,
        '        "vocals": {"vocals", "vocal", "voice"},\n',
        '        "vocals": {\n            "vocals",\n            "vocal",\n            "voice",\n            "lead vocals",\n            "lead vocal",\n        },\n',
    )
    replace_once(
        path,
        '''        "instrumental": {\n            "instrumental",\n            "karaoke",\n            "no vocals",\n            "no vocal",\n            "novocals",\n        },\n''',
        '''        "instrumental": {\n            "instrumental",\n            "instruments",\n            "accompaniment",\n            "karaoke",\n            "no vocals",\n            "no vocal",\n            "novocals",\n        },\n''',
    )
    replace_once(
        path,
        '        "piano": {"piano", "keys"},\n',
        '        "piano": {"piano", "keys", "keyboard"},\n',
    )
    replace_once(
        path,
        '        "dry": {"dry", "dereverb", "no reverb", "no echo"},\n',
        '        "dry": {\n            "dry",\n            "dereverb",\n            "dereverbed",\n            "no reverb",\n            "no echo",\n        },\n',
    )
    replace_once(
        path,
        '        "reverb": {"reverb", "echo"},\n',
        '        "reverb": {"reverb", "reverberation", "echo", "wet"},\n',
    )
    replace_once(
        path,
        '''        "clean": {\n            "clean",\n            "dry",\n            "no bleed",\n            "no aspiration",\n            "no noise",\n        },\n''',
        '''        "clean": {\n            "clean",\n            "dry",\n            "no bleed",\n            "no leakage",\n            "no aspiration",\n            "no breaths",\n            "no noise",\n        },\n''',
    )
    replace_once(
        path,
        '        "bleed": {"bleed"},\n',
        '        "bleed": {"bleed", "leakage", "residual bleed"},\n',
    )
    replace_once(
        path,
        '        "breaths": {"aspiration", "breath", "breaths"},\n',
        '        "breaths": {\n            "aspiration",\n            "aspirations",\n            "breath",\n            "breaths",\n            "breathing",\n        },\n',
    )
    replace_once(
        path,
        '        "sfx": {"sfx", "effect", "effects"},\n',
        '        "sfx": {\n            "sfx",\n            "effect",\n            "effects",\n            "sound effect",\n            "sound effects",\n        },\n',
    )


def patch_outputs() -> None:
    path = ROOT / "src" / "audio_restoration_colab" / "outputs.py"
    replace_once(
        path,
        "import re\nimport zipfile\n",
        "import json\nimport re\nimport subprocess\nimport zipfile\n",
    )
    replacement = '''def probe_audio_bitrate(source: Path) -> int | None:\n    command = [\n        "ffprobe",\n        "-v",\n        "error",\n        "-select_streams",\n        "a:0",\n        "-show_entries",\n        "stream=bit_rate:format=bit_rate",\n        "-of",\n        "json",\n        str(source),\n    ]\n    try:\n        completed = subprocess.run(\n            command,\n            check=False,\n            capture_output=True,\n            text=True,\n        )\n    except FileNotFoundError:\n        return None\n    if completed.returncode != 0:\n        return None\n    try:\n        payload = json.loads(completed.stdout or "{}")\n    except json.JSONDecodeError:\n        return None\n\n    candidates: list[object] = []\n    streams = payload.get("streams")\n    if isinstance(streams, list) and streams and isinstance(streams[0], dict):\n        candidates.append(streams[0].get("bit_rate"))\n    format_info = payload.get("format")\n    if isinstance(format_info, dict):\n        candidates.append(format_info.get("bit_rate"))\n    for raw_value in candidates:\n        try:\n            bitrate = int(raw_value)\n        except (TypeError, ValueError):\n            continue\n        if bitrate > 0:\n            return bitrate\n    return None\n\n\ndef _codec_bitrate(\n    source_bitrate: int | None,\n    *,\n    default: int,\n    minimum: int,\n    maximum: int,\n) -> str:\n    bitrate = default if source_bitrate is None else int(source_bitrate)\n    return str(max(minimum, min(maximum, bitrate)))\n\n\ndef build_ffmpeg_command(\n    source: Path,\n    target: Path,\n    *,\n    source_bitrate: int | None = None,\n) -> list[str]:\n    command = [\n        "ffmpeg",\n        "-y",\n        "-hide_banner",\n        "-loglevel",\n        "error",\n        "-i",\n        str(source),\n        "-vn",\n    ]\n    suffix = target.suffix.lower()\n    codec_options = {\n        ".mp3": [\n            "-c:a",\n            "libmp3lame",\n            "-b:a",\n            _codec_bitrate(\n                source_bitrate,\n                default=320_000,\n                minimum=32_000,\n                maximum=320_000,\n            ),\n        ],\n        ".wav": ["-c:a", "pcm_s24le"],\n        ".flac": ["-c:a", "flac"],\n        ".m4a": [\n            "-c:a",\n            "aac",\n            "-b:a",\n            _codec_bitrate(\n                source_bitrate,\n                default=320_000,\n                minimum=32_000,\n                maximum=512_000,\n            ),\n        ],\n        ".aac": [\n            "-c:a",\n            "aac",\n            "-b:a",\n            _codec_bitrate(\n                source_bitrate,\n                default=320_000,\n                minimum=32_000,\n                maximum=512_000,\n            ),\n        ],\n        ".opus": [\n            "-c:a",\n            "libopus",\n            "-b:a",\n            _codec_bitrate(\n                source_bitrate,\n                default=256_000,\n                minimum=16_000,\n                maximum=512_000,\n            ),\n        ],\n        ".aiff": ["-c:a", "pcm_s24be"],\n    }\n    if suffix == ".ogg":\n        if source_bitrate is None:\n            codec_options[suffix] = ["-c:a", "libvorbis", "-q:a", "8"]\n        else:\n            codec_options[suffix] = [\n                "-c:a",\n                "libvorbis",\n                "-b:a",\n                _codec_bitrate(\n                    source_bitrate,\n                    default=256_000,\n                    minimum=32_000,\n                    maximum=500_000,\n                ),\n            ]\n    try:\n        command.extend(codec_options[suffix])\n    except KeyError as error:\n        raise ValueError(f"Неподдерживаемый формат результата: {suffix}") from error\n    command.append(str(target))\n    return command\n\n\n'''
    replace_region(
        path,
        r"def build_ffmpeg_command\(.*?\n\n\ndef create_result_zip",
        replacement + "def create_result_zip",
    )


def patch_jobs() -> None:
    path = ROOT / "src" / "audio_restoration_colab" / "jobs.py"
    replace_once(
        path,
        '''from .outputs import (\n    build_ffmpeg_command,\n    create_result_zip,\n    output_extension,\n    safe_stem,\n)\n''',
        '''from .outputs import (\n    build_ffmpeg_command,\n    create_result_zip,\n    output_extension,\n    probe_audio_bitrate,\n    safe_stem,\n)\n''',
    )
    replace_once(
        path,
        '''            extension = output_extension(format_choice, source)\n            source_name = safe_stem(source.name)\n''',
        '''            extension = output_extension(format_choice, source)\n            source_bitrate = (\n                probe_audio_bitrate(source)\n                if format_choice == "source"\n                else None\n            )\n            source_name = safe_stem(source.name)\n''',
    )
    replace_once(
        path,
        '''                self.ffmpeg_runner(\n                    build_ffmpeg_command(raw_result.path, target)\n                )\n''',
        '''                self.ffmpeg_runner(\n                    build_ffmpeg_command(\n                        raw_result.path,\n                        target,\n                        source_bitrate=source_bitrate,\n                    )\n                )\n''',
    )


def patch_separator_tests() -> None:
    path = ROOT / "tests" / "test_separator_server.py"
    marker = '''    def test_multiple_unknown_outputs_fail_instead_of_guessing(self) -> None:\n'''
    addition = '''    def test_two_stem_other_maps_to_known_counterpart(self) -> None:\n        cases = (\n            (("dry", "reverb"), "dry", "reverb"),\n            (("clean", "bleed"), "bleed", "clean"),\n            (("clean", "breaths"), "aspiration", "clean"),\n            (("vocals", "instrumental"), "vocals", "instrumental"),\n        )\n        for expected_roles, known_label, remaining_role in cases:\n            with self.subTest(expected_roles=expected_roles):\n                with tempfile.TemporaryDirectory() as directory:\n                    root = Path(directory)\n                    known = root / f"song_({known_label})_model.wav"\n                    residual = root / "song_(other)_model.wav"\n                    known.write_bytes(b"known")\n                    residual.write_bytes(b"counterpart")\n\n                    mapped = self.server._canonicalize_outputs(\n                        paths=[residual, known],\n                        output_dir=root,\n                        expected_roles=expected_roles,\n                    )\n\n                    self.assertEqual(\n                        mapped[remaining_role].read_bytes(),\n                        b"counterpart",\n                    )\n\n'''
    text = path.read_text(encoding="utf-8")
    if addition not in text:
        if marker not in text:
            raise RuntimeError(f"Не найден маркер теста в {path}")
        path.write_text(text.replace(marker, addition + marker, 1), encoding="utf-8")


def patch_output_tests() -> None:
    path = ROOT / "tests" / "test_outputs.py"
    replace_once(
        path,
        "import zipfile\nfrom pathlib import Path\n",
        "import zipfile\nfrom pathlib import Path\nfrom unittest.mock import Mock, patch\n",
    )
    replace_once(
        path,
        '''    output_extension,\n    safe_stem,\n)\n''',
        '''    output_extension,\n    probe_audio_bitrate,\n    safe_stem,\n)\n''',
    )
    marker = '''    def test_zip_contains_only_result_names(self) -> None:\n'''
    addition = '''    def test_mp3_command_can_preserve_source_bitrate(self) -> None:\n        command = build_ffmpeg_command(\n            Path("/tmp/input.wav"),\n            Path("/tmp/output.mp3"),\n            source_bitrate=192_000,\n        )\n\n        self.assertEqual(\n            command[command.index("-b:a") + 1],\n            "192000",\n        )\n\n    @patch("audio_restoration_colab.outputs.subprocess.run")\n    def test_probe_audio_bitrate_prefers_audio_stream(self, run: Mock) -> None:\n        run.return_value = Mock(\n            returncode=0,\n            stdout=(\n                '{"streams": [{"bit_rate": "192000"}], '\n                '"format": {"bit_rate": "201000"}}'\n            ),\n        )\n\n        self.assertEqual(probe_audio_bitrate(Path("song.mp3")), 192_000)\n\n'''
    text = path.read_text(encoding="utf-8")
    if addition not in text:
        if marker not in text:
            raise RuntimeError(f"Не найден маркер теста в {path}")
        path.write_text(text.replace(marker, addition + marker, 1), encoding="utf-8")


def patch_job_tests() -> None:
    path = ROOT / "tests" / "test_jobs.py"
    replace_once(
        path,
        "from pathlib import Path\n",
        "from pathlib import Path\nfrom unittest.mock import patch\n",
    )
    marker = '''    def test_preview_failure_falls_back_to_raw_wav(self) -> None:\n'''
    addition = '''    @patch(\n        "audio_restoration_colab.jobs.probe_audio_bitrate",\n        return_value=192_000,\n    )\n    def test_source_mp3_preserves_reported_bitrate(self, _probe) -> None:\n        with tempfile.TemporaryDirectory() as directory:\n            root = Path(directory)\n            source = root / "song.mp3"\n            source.write_bytes(b"input")\n            commands: list[list[str]] = []\n\n            def tracking_ffmpeg(command: list[str]) -> None:\n                commands.append(command)\n                fake_ffmpeg(command)\n\n            service = AudioJobService(\n                jobs_root=root / "jobs",\n                worker=FakeWorker(),\n                ffmpeg_runner=tracking_ffmpeg,\n            )\n            service.process(\n                source=source,\n                model_id="denoise_normal",\n                format_choice="source",\n                raw_settings={},\n                progress=lambda _fraction, _message: None,\n            )\n\n            result_commands = [\n                command\n                for command in commands\n                if Path(command[-1]).parent.name == "results"\n            ]\n            self.assertEqual(len(result_commands), 2)\n            self.assertTrue(\n                all(\n                    command[command.index("-b:a") + 1] == "192000"\n                    for command in result_commands\n                )\n            )\n\n'''
    text = path.read_text(encoding="utf-8")
    if addition not in text:
        if marker not in text:
            raise RuntimeError(f"Не найден маркер теста в {path}")
        path.write_text(text.replace(marker, addition + marker, 1), encoding="utf-8")


def main() -> None:
    patch_separator_server()
    patch_outputs()
    patch_jobs()
    patch_separator_tests()
    patch_output_tests()
    patch_job_tests()


if __name__ == "__main__":
    main()
