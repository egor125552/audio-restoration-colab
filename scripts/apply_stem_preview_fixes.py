from __future__ import annotations

import re
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one exact match, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def regex_replace_once(
    path: Path,
    pattern: str,
    replacement: str,
) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(
        pattern,
        replacement,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise SystemExit(f"{path}: regex expected one match, found {count}")
    path.write_text(updated, encoding="utf-8")


def patch_separator() -> None:
    path = Path("workers/separator_server.py")
    text = path.read_text(encoding="utf-8")
    if "def _output_role_label(" in text:
        return
    replace_once(path, "import os\nimport shutil\n", "import os\nimport re\nimport shutil\n")
    replacement = '''def _canonicalize_outputs(
    *,
    paths: list[Path],
    output_dir: Path,
    expected_roles: tuple[str, ...],
) -> dict[str, Path]:
    assignments: dict[str, Path] = {}
    unused = list(paths)
    for role in expected_roles:
        matched = next(
            (
                path
                for path in unused
                if _matches_role(
                    path=path,
                    role=role,
                    expected_roles=expected_roles,
                )
            ),
            None,
        )
        if matched is not None:
            assignments[role] = matched
            unused.remove(matched)

    if len(expected_roles) == 1 and not assignments and unused:
        assignments[expected_roles[0]] = unused.pop(0)

    missing = [role for role in expected_roles if role not in assignments]
    if missing:
        filenames = ", ".join(path.name for path in paths)
        raise ValueError(
            "Не удалось определить роли дорожек "
            + ", ".join(missing)
            + f". Получены файлы: {filenames}"
        )

    canonical: dict[str, Path] = {}
    for role, source in assignments.items():
        target = output_dir / f"{role}.wav"
        if source.resolve() != target.resolve():
            if target.exists():
                target.unlink()
            shutil.copy2(source, target)
        canonical[role] = target.resolve()
    return canonical


def _matches_role(
    *,
    path: Path,
    role: str,
    expected_roles: tuple[str, ...],
) -> bool:
    label = _output_role_label(path)
    aliases = {
        "vocals": {"vocals", "vocal", "voice"},
        "instrumental": {
            "instrumental",
            "karaoke",
            "no vocals",
            "no vocal",
            "novocals",
        },
        "drums": {"drums", "drum"},
        "bass": {"bass"},
        "guitar": {"guitar"},
        "piano": {"piano", "keys"},
        "other": {"other", "non vocals"},
        "dry": {"dry", "dereverb", "no reverb", "no echo"},
        "reverb": {"reverb", "echo"},
        "clean": {
            "clean",
            "dry",
            "no bleed",
            "no aspiration",
            "no noise",
        },
        "bleed": {"bleed"},
        "breaths": {"aspiration", "breath", "breaths"},
        "kick": {"kick", "bombo"},
        "snare": {"snare", "redoblante"},
        "toms": {"toms", "tom"},
        "cymbals": {"cymbals", "platillos"},
        "hihat": {"hihat", "hi hat", "hh"},
        "ride": {"ride"},
        "crash": {"crash"},
        "speech": {"speech", "dialog", "dialogue"},
        "music": {"music"},
        "sfx": {"sfx", "effect", "effects"},
    }
    if role == "instrumental" and label == "other":
        return "other" not in expected_roles
    return label in aliases.get(role, {role})


def _output_role_label(path: Path) -> str:
    groups = re.findall(r"\\(([^()]*)\\)", path.stem)
    if groups:
        return _normalize_role_label(groups[-1])

    label = _normalize_role_label(path.stem)
    for prefix in ("input ", "output ", "stem "):
        if label.startswith(prefix):
            return label[len(prefix) :]
    return label


def _normalize_role_label(value: str) -> str:
    return " ".join(
        re.sub(r"[^a-z0-9]+", " ", value.lower()).split()
    )


'''
    regex_replace_once(
        path,
        r"def _canonicalize_outputs\(.*?(?=@contextlib\.contextmanager)",
        replacement,
    )


def patch_app() -> None:
    path = Path("src/audio_restoration_colab/app.py")
    text = path.read_text(encoding="utf-8")
    if "from .result_ui import build_result_layout" not in text:
        replace_once(
            path,
            "from .mixer import build_mix\nfrom .runtime import ModelResult, RouterWorker, RuntimeLayout\n",
            "from .mixer import build_mix\n"
            "from .result_ui import build_result_layout\n"
            "from .runtime import ModelResult, RouterWorker, RuntimeLayout\n",
        )
    text = path.read_text(encoding="utf-8")
    if "stem_preview_state = gr.State({})" not in text:
        replace_once(
            path,
            "        stem_state = gr.State({})\n",
            "        stem_state = gr.State({})\n"
            "        stem_preview_state = gr.State({})\n",
        )
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "Инференс повторно не запускается. Здесь используются уже "
        '"\n                "готовые WAV-дорожки. Горячие клавиши вне полей ввода: ',
        "Инференс повторно не запускается. Для микса используются "
        '"\n                "полные дорожки, а для прослушивания — облегчённые "\n'
        '                "MP3-превью. Горячие клавиши вне полей ввода: ',
    )
    path.write_text(text, encoding="utf-8")

    replacement = '''            except JobProcessingError as error:
                status_html = (
                    "<div role='status' aria-live='assertive'><strong>Ошибка: "
                    + html.escape(str(error))
                    + "</strong><br>Полный лог сохранён ниже и одновременно "
                    "выведен в консоль Colab.</div>"
                )
                hidden = gr.update(visible=False)
                return (
                    status_html,
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                    [],
                    None,
                    str(error.log_path),
                    {},
                    {},
                    hidden,
                    gr.update(choices=[], value=None),
                    None,
                    gr.update(choices=[], value=[]),
                    hidden,
                    hidden,
                    hidden,
                    hidden,
                    hidden,
                    gr.update(visible=False, label="Остальное, %"),
                    hidden,
                    hidden,
                    hidden,
                    hidden,
                )
            except ValueError as error:
                raise gr.Error(str(error)) from error

            payload = {
                item.role: str(item.path) for item in result.raw_results
            }
            preview_payload = {
                item.role: str(item.path)
                for item in result.preview_results
            }
            roles = list(payload)
            layout = build_result_layout(roles)
            first_role = roles[0] if roles else None
            primary_role = roles[0] if roles else None
            secondary_role = roles[1] if len(roles) > 1 else None
            status_html = (
                "<div role='status' aria-live='polite'><strong>"
                + html.escape(result.message)
                + "</strong></div>"
            )
            return (
                status_html,
                gr.update(
                    value=preview_payload.get(primary_role),
                    label=(
                        "Быстрое превью: "
                        + ROLE_TITLES.get(primary_role, primary_role)
                        if primary_role
                        else "Основной результат"
                    ),
                    visible=primary_role is not None,
                ),
                gr.update(
                    value=preview_payload.get(secondary_role),
                    label=(
                        "Быстрое превью: "
                        + ROLE_TITLES.get(secondary_role, secondary_role)
                        if secondary_role
                        else "Второй результат"
                    ),
                    visible=secondary_role is not None,
                ),
                [str(path) for path in result.files],
                str(result.archive),
                str(result.log_path),
                payload,
                preview_payload,
                gr.update(visible=layout.editor_visible),
                gr.update(
                    choices=list(layout.choices),
                    value=first_role,
                ),
                preview_payload.get(first_role) if first_role else None,
                gr.update(
                    choices=list(layout.choices),
                    value=roles,
                ),
                gr.update(visible=layout.gain_visibility["vocals"]),
                gr.update(visible=layout.gain_visibility["drums"]),
                gr.update(visible=layout.gain_visibility["bass"]),
                gr.update(visible=layout.gain_visibility["guitar"]),
                gr.update(visible=layout.gain_visibility["piano"]),
                gr.update(
                    visible=layout.gain_visibility["other"],
                    label=layout.other_gain_label,
                ),
                gr.update(visible=layout.show_all_preset),
                gr.update(visible=layout.show_no_vocals_preset),
                gr.update(visible=layout.show_only_vocals_preset),
                gr.update(visible=layout.show_no_drums_preset),
            )

        run_button.click(
            process_audio,
            inputs=[
                input_file,
                model_dropdown,
                output_format,
                settings_state,
            ],
            outputs=[
                status,
                primary_preview,
                secondary_preview,
                result_files,
                result_zip,
                diagnostic_log,
                stem_state,
                stem_preview_state,
                stem_editor,
                stem_selector,
                stem_preview,
                mix_roles,
                gain_vocals,
                gain_drums,
                gain_bass,
                gain_guitar,
                gain_piano,
                gain_other,
                mix_all,
                mix_no_vocals,
                mix_only_vocals,
                mix_no_drums,
            ],
            concurrency_limit=1,
            show_progress="full",
        )

        stem_selector.change(
            lambda role, state: (state or {}).get(role),
            inputs=[stem_selector, stem_preview_state],
            outputs=stem_preview,
        )

'''
    regex_replace_once(
        path,
        r"            except JobProcessingError as error:.*?"
        r"        stem_selector\.change\(.*?"
        r"            outputs=stem_preview,\n"
        r"        \)\n\n",
        replacement,
    )


def patch_preview_test() -> None:
    path = Path("tests/test_jobs.py")
    text = path.read_text(encoding="utf-8")
    old = '''            preview_commands = [
                command for command in commands if "-b:a" in command
            ]
'''
    new = '''            preview_commands = [
                command
                for command in commands
                if "-b:a" in command
                and command[command.index("-b:a") + 1] == "96k"
            ]
'''
    if old in text:
        replace_once(path, old, new)


def main() -> None:
    patch_separator()
    patch_app()
    patch_preview_test()


if __name__ == "__main__":
    main()
