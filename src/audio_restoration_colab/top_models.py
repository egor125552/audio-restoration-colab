from __future__ import annotations

from dataclasses import replace

from .catalog import MODEL_SPECS

_APPLIED = False


def apply_top_model_catalog() -> None:
    """Оставить в пользовательском каталоге только современный топ-набор.

    Старые Demucs/HTDemucs checkpoint не должны появляться в интерфейсе.
    Базовые задачи 4 и 6 стемов переводятся на Mel-Band/BS-RoFormer,
    а специализированные RoFormer-ансамбли из исходного каталога остаются.
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
            "Это основной универсальный режим для дальнейшего микширования."
        ),
        warning=(
            "Специализированная модель одного инструмента иногда даёт чище "
            "конкретный стем, но старые Demucs-модели здесь не используются."
        ),
        model_filename="BS-Rofo-SW-Fixed.ckpt",
        ensemble_preset=None,
        source_text="BS-RoFormer SW Fixed / современный 6-stem checkpoint",
    )

    four_stems = MODEL_SPECS["stems_four"]
    MODEL_SPECS["stems_four"] = replace(
        four_stems,
        title="Топовое разделение на четыре дорожки — Mel-Band RoFormer",
        short_title="Разделитель — топовые 4 стема",
        size_text="крупная Mel-Band RoFormer модель",
        purpose="Вокал, барабаны, бас и остальная музыка",
        description=(
            "Большая четырёхстемовая Mel-Band RoFormer. Используется вместо "
            "старого HTDemucs, когда гитара и пианино отдельно не нужны."
        ),
        warning=(
            "Гитара и фортепиано останутся внутри дорожки «остальное». "
            "Для шести отдельных дорожек используй BS-RoFormer SW."
        ),
        model_filename="mel_band_roformer_4stems_large_ver1.ckpt",
        ensemble_preset=None,
        source_text="Aname / Mel-Band RoFormer 4 Stems Large",
    )

    # Эти пункты были построены на Demucs checkpoint. Они не должны тихо
    # оставаться в меню под современными русскими названиями. Подробные
    # барабаны вернутся отдельным пунктом после интеграции Mel-Band DrumSep.
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
