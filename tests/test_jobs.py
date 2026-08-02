from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from audio_restoration_colab.jobs import (
    AudioJobService,
    JobProcessingError,
    JobProgress,
    validate_source,
)
from audio_restoration_colab.runtime import ModelResult


class FakeWorker:
    def __init__(self) -> None:
        self.received_settings: dict[str, object] | None = None
        self.received_source: Path | None = None

    def run(
        self,
        *,
        model_id: str,
        source: Path,
        output_dir: Path,
        settings: dict[str, object],
        progress: JobProgress,
    ) -> list[ModelResult]:
        self.received_settings = settings
        self.received_source = source
        clean = output_dir / "worker-clean.wav"
        noise = output_dir / "worker-noise.wav"
        clean.write_bytes(b"clean")
        noise.write_bytes(b"noise")
        return [
            ModelResult(role="clean", path=clean),
            ModelResult(role="noise", path=noise),
        ]


class FailingWorker:
    def run(
        self,
        *,
        model_id: str,
        source: Path,
        output_dir: Path,
        settings: dict[str, object],
        progress: JobProgress,
    ) -> list[ModelResult]:
        del model_id, source, output_dir, settings, progress
        raise ValueError("CUDA out of memory")


def fake_ffmpeg(command: list[str]) -> None:
    source = Path(command[command.index("-i") + 1])
    target = Path(command[-1])
    target.write_bytes(source.read_bytes())


class JobTests(unittest.TestCase):
    def test_source_must_exist_and_have_audio_extension(self) -> None:
        with self.assertRaisesRegex(ValueError, "не найден"):
            validate_source(Path("/missing/song.mp3"))

        with tempfile.TemporaryDirectory() as directory:
            text_file = Path(directory) / "notes.txt"
            text_file.write_text("not audio", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "формат"):
                validate_source(text_file)

    def test_job_normalizes_settings_and_creates_files_and_zip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Моя песня.mp3"
            source.write_bytes(b"input")
            worker = FakeWorker()
            messages: list[str] = []
            commands: list[list[str]] = []

            def tracking_ffmpeg(command: list[str]) -> None:
                commands.append(command)
                fake_ffmpeg(command)

            service = AudioJobService(
                jobs_root=root / "jobs",
                worker=worker,
                ffmpeg_runner=tracking_ffmpeg,
            )

            result = service.process(
                source=source,
                model_id="denoise_normal",
                format_choice="mp3",
                raw_settings={"quality": "maximum", "segment": 999},
                progress=lambda _fraction, message: messages.append(message),
            )

            self.assertEqual(
                worker.received_settings,
                {"quality": "maximum", "segment": 352},
            )
            self.assertIsNotNone(worker.received_source)
            assert worker.received_source is not None
            self.assertEqual(worker.received_source.suffix, ".wav")
            self.assertEqual(worker.received_source.name, "model-input.wav")
            self.assertEqual(len(result.files), 2)
            self.assertTrue(all(path.suffix == ".mp3" for path in result.files))
            self.assertTrue(result.archive.is_file())
            self.assertTrue(result.log_path.is_file())
            self.assertEqual(result.primary_preview.suffix, ".mp3")
            self.assertEqual(result.secondary_preview.suffix, ".mp3")
            self.assertEqual(
                [item.role for item in result.preview_results],
                ["clean", "noise"],
            )
            self.assertTrue(
                all(
                    item.path.parent.name == "previews"
                    for item in result.preview_results
                )
            )
            preview_commands = [
                command
                for command in commands
                if "-b:a" in command
                and command[command.index("-b:a") + 1] == "96k"
            ]
            self.assertEqual(len(preview_commands), 2)
            self.assertTrue(
                all(
                    command[command.index("-b:a") + 1] == "96k"
                    for command in preview_commands
                )
            )
            self.assertIn("Готово", messages[-1])
            self.assertIn(
                "обработка завершена успешно",
                result.log_path.read_text(encoding="utf-8"),
            )

    def test_m4a_is_decoded_to_wav_and_download_can_stay_m4a(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "phone call.m4a"
            source.write_bytes(b"m4a-container")
            worker = FakeWorker()
            commands: list[list[str]] = []

            def tracking_ffmpeg(command: list[str]) -> None:
                commands.append(command)
                fake_ffmpeg(command)

            service = AudioJobService(
                jobs_root=root / "jobs",
                worker=worker,
                ffmpeg_runner=tracking_ffmpeg,
            )
            result = service.process(
                source=source,
                model_id="flashsr_medium",
                format_choice="source",
                raw_settings={"lowpass": True},
                progress=lambda _fraction, _message: None,
            )

            self.assertIsNotNone(worker.received_source)
            assert worker.received_source is not None
            self.assertEqual(worker.received_source.suffix, ".wav")
            self.assertEqual(Path(commands[0][-1]).suffix, ".wav")
            self.assertTrue(all(path.suffix == ".m4a" for path in result.files))
            self.assertEqual(result.primary_preview.suffix, ".mp3")
            self.assertEqual(result.secondary_preview.suffix, ".mp3")

    def test_preview_failure_falls_back_to_raw_wav(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "speech.wav"
            source.write_bytes(b"input")
            worker = FakeWorker()

            def ffmpeg_with_preview_failure(command: list[str]) -> None:
                if "-b:a" in command:
                    raise ValueError("encoder unavailable")
                fake_ffmpeg(command)

            service = AudioJobService(
                jobs_root=root / "jobs",
                worker=worker,
                ffmpeg_runner=ffmpeg_with_preview_failure,
            )
            result = service.process(
                source=source,
                model_id="denoise_normal",
                format_choice="wav",
                raw_settings={},
                progress=lambda _fraction, _message: None,
            )

            self.assertEqual(result.primary_preview.suffix, ".wav")
            self.assertIn(
                "облегчённое превью",
                result.log_path.read_text(encoding="utf-8"),
            )

    def test_failed_job_keeps_downloadable_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "speech.wav"
            source.write_bytes(b"input")
            service = AudioJobService(
                jobs_root=root / "jobs",
                worker=FailingWorker(),
                ffmpeg_runner=fake_ffmpeg,
            )

            with self.assertRaises(JobProcessingError) as context:
                service.process(
                    source=source,
                    model_id="flashsr_medium",
                    format_choice="wav",
                    raw_settings={"lowpass": True},
                    progress=lambda _fraction, _message: None,
                )

            error = context.exception
            self.assertTrue(error.log_path.is_file())
            log_text = error.log_path.read_text(encoding="utf-8")
            self.assertIn("flashsr_medium", log_text)
            self.assertIn("CUDA out of memory", log_text)

    def test_jobs_outside_system_temp_are_redirected_for_gradio(self) -> None:
        service = AudioJobService(
            jobs_root=Path("/content/audio-restoration-work"),
            worker=FakeWorker(),
            ffmpeg_runner=fake_ffmpeg,
        )
        expected = Path(tempfile.gettempdir()).resolve() / "audio-restoration-work"
        self.assertEqual(service.jobs_root, expected)

    def test_cleanup_removes_only_old_job_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = root / "jobs"
            old = jobs / "old"
            current = jobs / "current"
            old.mkdir(parents=True)
            current.mkdir()
            (old / "file.txt").write_text("old", encoding="utf-8")
            (current / "file.txt").write_text("current", encoding="utf-8")
            old_timestamp = 1_000_000_000
            old.touch()
            import os

            os.utime(old, (old_timestamp, old_timestamp))

            service = AudioJobService(
                jobs_root=jobs,
                worker=FakeWorker(),
                ffmpeg_runner=fake_ffmpeg,
            )
            service.cleanup_old_jobs(max_age_seconds=3600, now=1_000_010_000)

            self.assertFalse(old.exists())
            self.assertTrue(current.exists())


if __name__ == "__main__":
    unittest.main()
