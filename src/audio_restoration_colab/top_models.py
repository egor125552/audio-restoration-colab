from __future__ import annotations

from dataclasses import replace

from .catalog import MODEL_SPECS

_APPLIED = False


def apply_top_model_catalog() -> None:
    """Оставить в пользовательском каталоге только современный топ-набор.

    Старые Demucs/HTDemucs checkpoint не должны появляться в интерфейсе.
    BS-RoFormer модели выбираются по slug из versioned registry, а не по
    хрупкому внешнему имени файла.
    """

    global _APPLIED
    if _APPLIED:
        return

    six_stems = MODEL_SPECS["stems_six"]
    MODEL_SPECS["stems_six"] = replace(
        six_stems,
        title="Топовое разделение на шесть дорожек — BS-RoFormer SW",
        short_title="Разделитель — топовые 6 стемов",
        size_text="около 700 МБ; скачивается один раз",
        purpose="Вокал, барабаны, бас, гитара, пианино и остальное",
        description=(
            "BS-RoFormer SW одновременно создаёт шесть основных дорожек. "
            "Checkpoint скачивается из живого реестра и проверяется по SHA-256."
        ),
        warning=(
            "Специализированная модель одного инструмента иногда даёт чище "
            "конкретный стем, но старые Demucs-модели здесь не используются."
        ),
        model_filename=(
            "bsinfer:roformer-model-bs-roformer-sw-by-jarredou"
        ),
        ensemble_preset=None,
        source_text="OpenMIRLab BS-RoFormer registry / SW Fixed",
    )

    four_stems = MODEL_SPECS["stems_four"]
    MODEL_SPECS["stems_four"] = replace(
        four_stems,
        title="Топовое разделение на четыре дорожки — BS-RoFormer",
        short_title="Разделитель — топовые 4 стема",
        size_text="BS-RoFormer MUSDB18HQ; скачивается один раз",
        purpose="Вокал, барабаны, бас и остальная музыка",
        description=(
            "Четырёхстемовый BS-RoFormer для вокала, барабанов, баса и "
            "остальной музыки. Checkpoint проверяется по SHA-256."
        ),
        warning=(
            "Гитара и фортепиано останутся внутри дорожки «остальное». "
            "Для шести отдельных дорожек используй BS-RoFormer SW."
        ),
        model_filename=(
            "bsinfer:roformer-model-bs-roformer-musdb18hq-by-zfturbo"
        ),
        ensemble_preset=None,
        source_text="OpenMIRLab BS-RoFormer registry / MUSDB18HQ",
    )

    guitar = MODEL_SPECS["stems_guitar"]
    MODEL_SPECS["stems_guitar"] = replace(
        guitar,
        model_filename="melband_roformer_guitar_becruily.ckpt",
        source_text="becruily / Mel-Band RoFormer guitar",
    )

    # Эти пункты были построены на Demucs checkpoint. Они не должны тихо
    # оставаться в меню под современными русскими названиями. Подробные
    # барабаны вернутся только после проверки Mel-Band DrumSep.
    for model_id in ("stems_drums_detailed", "stems_cinematic"):
        MODEL_SPECS.pop(model_id, None)

    for model_id, model in tuple(MODEL_SPECS.items()):
        filename = (model.model_filename or "").lower()
        if (
            filename.startswith("demucs:")
            or "htdemucs" in filename
            or filename.endswith(".th")
        ):
            MODEL_SPECS.pop(model_id, None)

    _APPLIED = True
