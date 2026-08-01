from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SettingSpec:
    key: str
    label: str
    kind: str
    default: Any
    help_text: str
    choices: tuple[tuple[str, Any], ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    title: str
    short_title: str
    backend: str
    size_text: str
    purpose: str
    description: str
    warning: str
    output_roles: tuple[str, ...]
    settings: tuple[SettingSpec, ...]
    model_filename: str | None = None
    ensemble_preset: str | None = None
    category: str = "restoration"
    source_text: str = ""
    license_text: str = ""


DENOISE_SETTINGS = (
    SettingSpec(
        key="quality",
        label="Качество и скорость",
        kind="choice",
        default="balanced",
        choices=(
            ("Быстро", "fast"),
            ("Сбалансированно", "balanced"),
            ("Максимальное качество", "maximum"),
        ),
        help_text="Чем выше качество, тем дольше обработка.",
    ),
    SettingSpec(
        key="segment",
        label="Размер обрабатываемого фрагмента",
        kind="integer",
        default=256,
        minimum=128,
        maximum=352,
        step=32,
        help_text="256 — безопасное значение для Tesla T4.",
    ),
)

STEM_SETTINGS = (
    SettingSpec(
        key="quality",
        label="Качество и скорость",
        kind="choice",
        default="balanced",
        choices=(
            ("Быстро", "fast"),
            ("Сбалансированно", "balanced"),
            ("Максимальное качество", "maximum"),
        ),
        help_text=(
            "Качество управляет перекрытием фрагментов. "
            "Максимальный режим заметно медленнее."
        ),
    ),
    SettingSpec(
        key="segment",
        label="Размер спектрального фрагмента",
        kind="integer",
        default=256,
        minimum=128,
        maximum=512,
        step=32,
        help_text="256 безопасно для T4; 320–512 могут повысить качество.",
    ),
    SettingSpec(
        key="overlap",
        label="Перекрытие соседних фрагментов",
        kind="integer",
        default=8,
        minimum=2,
        maximum=16,
        step=2,
        help_text="Большее перекрытие уменьшает стыки, но замедляет обработку.",
    ),
    SettingSpec(
        key="chunk_minutes",
        label="Длина большого куска",
        kind="integer",
        default=10,
        minimum=1,
        maximum=30,
        step=1,
        help_text=(
            "Длинные песни режутся на крупные куски, чтобы расход памяти "
            "не рос вместе с длительностью."
        ),
    ),
    SettingSpec(
        key="keep_loaded",
        label="Оставлять модель загруженной",
        kind="boolean",
        default=True,
        help_text=(
            "Повторный запуск той же модели начнётся без повторной загрузки "
            "весов, пока работает Colab."
        ),
    ),
)


MODEL_SPECS: dict[str, ModelSpec] = {
    "denoise_normal": ModelSpec(
        model_id="denoise_normal",
        title="DeNoise — обычная очистка",
        short_title="DeNoise, обычная",
        backend="separator",
        size_text="около 0,9 ГБ",
        purpose="Музыка, речь и обычные записи",
        description=(
            "Отделяет чистый звук от шума. Возвращает два файла: очищенный "
            "звук и выделенный шум."
        ),
        warning=(
            "Это очистка, а не дорисовка частот. Полезно запускать перед "
            "FlashSR или AudioSR, если в записи есть заметный шум."
        ),
        output_roles=("clean", "noise"),
        settings=DENOISE_SETTINGS,
        model_filename="denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
        source_text="UVR / audio-separator",
        license_text="Код MIT; условия весов проверять у автора модели.",
    ),
    "denoise_aggressive": ModelSpec(
        model_id="denoise_aggressive",
        title="DeNoise — агрессивная очистка",
        short_title="DeNoise, агрессивная",
        backend="separator",
        size_text="около 0,9 ГБ",
        purpose="Очень шумные записи",
        description=(
            "Сильнее отделяет шум. Возвращает очищенный звук и выделенный шум."
        ),
        warning=(
            "Может убрать тихие инструменты, хвосты реверберации и фоновые "
            "звуки. Сначала лучше попробовать обычную версию."
        ),
        output_roles=("clean", "noise"),
        settings=DENOISE_SETTINGS,
        model_filename=(
            "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt"
        ),
        source_text="UVR / audio-separator",
        license_text="Код MIT; условия весов проверять у автора модели.",
    ),
    "lavasr_small": ModelSpec(
        model_id="lavasr_small",
        title="LavaSR — маленькая и быстрая",
        short_title="Дорисовка, маленькая",
        backend="lavasr",
        size_text="около 50 МБ",
        purpose="Речь, звонки и телефонные записи",
        description=(
            "Быстро дорисовывает верхние частоты и может одновременно "
            "приглушить шум. Работает с исходной частотой от 8 до 48 кГц."
        ),
        warning=(
            "Модель обучена на речи. На музыке возможны металлический звук "
            "и потеря стерео."
        ),
        output_roles=("restored",),
        settings=(
            SettingSpec(
                key="input_rate",
                label="Ожидаемая полоса исходника",
                kind="choice",
                default="auto",
                choices=(
                    ("Определить автоматически", "auto"),
                    ("8 кГц", 8000),
                    ("16 кГц", 16000),
                    ("24 кГц", 24000),
                    ("32 кГц", 32000),
                    ("44,1 кГц", 44100),
                    ("48 кГц", 48000),
                ),
                help_text="Автоматический режим подходит чаще всего.",
            ),
            SettingSpec(
                key="denoise",
                label="Одновременно убрать шум",
                kind="boolean",
                default=False,
                help_text="Включай только для речи с постоянным шумом.",
            ),
            SettingSpec(
                key="batch",
                label="Режим длинного файла",
                kind="boolean",
                default=True,
                help_text="Уменьшает расход памяти на длинных записях.",
            ),
        ),
    ),
    "flashsr_medium": ModelSpec(
        model_id="flashsr_medium",
        title="FlashSR — средняя и быстрая",
        short_title="Дорисовка, средняя",
        backend="flashsr",
        size_text="около 3,2 ГБ весов",
        purpose="Музыка, речь и обычные звуки",
        description=(
            "Однопроходная дорисовка верхних частот до 48 кГц. "
            "Стереоканалы обрабатываются отдельно и сохраняются."
        ),
        warning=(
            "Дорисованные частоты правдоподобны, но не совпадают с удалённым "
            "оригиналом. Первый запуск долго скачивает три файла весов."
        ),
        output_roles=("restored",),
        settings=(
            SettingSpec(
                key="lowpass",
                label="Подготовить неровный частотный срез",
                kind="boolean",
                default=True,
                help_text=(
                    "Полезно после MP3 и нейросетевого разделения. "
                    "Выключи, если срез уже ровный."
                ),
            ),
        ),
    ),
    "audiosr_large": ModelSpec(
        model_id="audiosr_large",
        title="AudioSR — большая и медленная",
        short_title="Дорисовка, большая",
        backend="audiosr",
        size_text="несколько гигабайт",
        purpose="Музыка, речь и обычные звуки",
        description=(
            "Многопроходная дорисовка до 48 кГц. Даёт разные варианты "
            "при разном зерне и заметно медленнее FlashSR."
        ),
        warning=(
            "Модель плохо понимает дырявый срез после MP3 и сильные "
            "искажения. Для таких файлов оставь фильтр включённым."
        ),
        output_roles=("restored",),
        settings=(
            SettingSpec(
                key="mode",
                label="Тип звука",
                kind="choice",
                default="basic",
                choices=(
                    ("Обычный звук и музыка", "basic"),
                    ("Речь", "speech"),
                ),
                help_text="Для минусовки выбирай обычный звук и музыку.",
            ),
            SettingSpec(
                key="steps",
                label="Количество шагов",
                kind="integer",
                default=50,
                minimum=10,
                maximum=100,
                step=10,
                help_text="Больше шагов — медленнее; 50 обычно достаточно.",
            ),
            SettingSpec(
                key="guidance",
                label="Сила обработки",
                kind="number",
                default=3.5,
                minimum=1.0,
                maximum=10.0,
                step=0.1,
                help_text="Слишком большое значение может добавить артефакты.",
            ),
            SettingSpec(
                key="seed",
                label="Случайное зерно",
                kind="integer",
                default=42,
                minimum=0,
                maximum=2_147_483_647,
                step=1,
                help_text="Поменяй число, чтобы получить другой вариант.",
            ),
            SettingSpec(
                key="lowpass",
                label="Подготовить неровный частотный срез",
                kind="boolean",
                default=True,
                help_text="Рекомендуется для MP3 и разделённых файлов.",
            ),
        ),
    ),
    "stems_vocal_balanced": ModelSpec(
        model_id="stems_vocal_balanced",
        title="Вокал и минусовка — сбалансированный ансамбль",
        short_title="Разделитель — вокал и минусовка",
        backend="stems",
        size_text="несколько моделей; скачиваются лениво",
        purpose="Чистый вокал и полноценная минусовка",
        description=(
            "Ансамбль моделей audio-separator. Даёт хороший баланс между "
            "полнотой вокала и малым просачиванием инструментов."
        ),
        warning=(
            "Ансамбль медленнее одной модели. Все веса кэшируются, "
            "а повторный запуск не скачивает их заново."
        ),
        output_roles=("vocals", "instrumental"),
        settings=STEM_SETTINGS,
        ensemble_preset="vocal_balanced",
        category="stems",
        source_text="Кураторский preset audio-separator",
        license_text="Код MIT; лицензии отдельных весов показываются источником.",
    ),
    "stems_vocal_clean": ModelSpec(
        model_id="stems_vocal_clean",
        title="Максимально чистый вокал",
        short_title="Разделитель — максимально чистый вокал",
        backend="stems",
        size_text="ансамбль из двух моделей",
        purpose="Вокал с минимальным просачиванием музыки",
        description=(
            "Собирает вокал несколькими моделями и подавляет остатки "
            "инструментов. Подходит для ремиксов и последующей обработки."
        ),
        warning="Может убрать тихие бэк-вокалы и воздушные детали.",
        output_roles=("vocals", "instrumental"),
        settings=STEM_SETTINGS,
        ensemble_preset="vocal_clean",
        category="stems",
        source_text="Кураторский preset audio-separator",
        license_text="Код MIT; лицензии весов зависят от авторов.",
    ),
    "stems_instrumental_clean": ModelSpec(
        model_id="stems_instrumental_clean",
        title="Максимально чистая минусовка",
        short_title="Разделитель — чистая минусовка",
        backend="stems",
        size_text="ансамбль из двух моделей",
        purpose="Караоке и удаление вокала",
        description=(
            "Ставит чистоту инструментала выше сохранения каждой тихой "
            "детали вокала."
        ),
        warning="Иногда ослабляет инструменты, похожие по спектру на голос.",
        output_roles=("instrumental", "vocals"),
        settings=STEM_SETTINGS,
        ensemble_preset="instrumental_clean",
        category="stems",
        source_text="Кураторский preset audio-separator",
        license_text="Код MIT; лицензии весов зависят от авторов.",
    ),
    "stems_instrumental_full": ModelSpec(
        model_id="stems_instrumental_full",
        title="Полная минусовка с максимумом инструментов",
        short_title="Разделитель — полная минусовка",
        backend="stems",
        size_text="ансамбль из двух моделей",
        purpose="Сохранение тихих инструментов и атмосферы",
        description=(
            "Сохраняет больше музыкальных деталей, даже если в минусовке "
            "останется немного вокального следа."
        ),
        warning="Чище не всегда значит полнее: здесь приоритет полноте музыки.",
        output_roles=("instrumental", "vocals"),
        settings=STEM_SETTINGS,
        ensemble_preset="instrumental_full",
        category="stems",
        source_text="Кураторский preset audio-separator",
        license_text="Код MIT; лицензии весов зависят от авторов.",
    ),
    "stems_karaoke": ModelSpec(
        model_id="stems_karaoke",
        title="Караоке — удалить главный вокал",
        short_title="Разделитель — караоке",
        backend="stems",
        size_text="ансамбль из трёх моделей",
        purpose="Караоке с возможным сохранением бэк-вокала",
        description=(
            "Специализированный ансамбль для удаления ведущего вокала. "
            "На некоторых песнях сохраняет часть хора и бэк-вокалов."
        ),
        warning="Результат зависит от того, насколько тесно сведены хор и солист.",
        output_roles=("instrumental", "vocals"),
        settings=STEM_SETTINGS,
        ensemble_preset="karaoke",
        category="stems",
        source_text="Кураторский preset audio-separator",
        license_text="Код MIT; лицензии весов зависят от авторов.",
    ),
    "stems_six": ModelSpec(
        model_id="stems_six",
        title="Шесть основных дорожек",
        short_title="Разделитель — 6 стемов",
        backend="stems",
        size_text="около 1,5 ГБ",
        purpose="Вокал, барабаны, бас, гитара, пианино и остальное",
        description=(
            "Универсальное разделение одним запуском. Хорошая отправная "
            "точка для мини-микшера и дальнейших специализированных проходов."
        ),
        warning=(
            "Отдельные специализированные модели могут дать чище один "
            "конкретный инструмент."
        ),
        output_roles=("vocals", "drums", "bass", "guitar", "piano", "other"),
        settings=STEM_SETTINGS,
        model_filename="htdemucs_6s.yaml",
        category="stems",
        source_text="Meta Demucs через audio-separator",
        license_text="Код MIT; веса Demucs распространяются авторами модели.",
    ),
    "stems_four": ModelSpec(
        model_id="stems_four",
        title="Четыре основные дорожки — высокое качество",
        short_title="Разделитель — 4 стема",
        backend="stems",
        size_text="около 1 ГБ",
        purpose="Вокал, барабаны, бас и остальная музыка",
        description=(
            "Fine-tuned HTDemucs. Обычно чище универсальной шестидорожечной "
            "модели, если гитара и пианино отдельно не нужны."
        ),
        warning="Гитара и пианино останутся внутри дорожки «остальное».",
        output_roles=("vocals", "drums", "bass", "other"),
        settings=STEM_SETTINGS,
        model_filename="htdemucs_ft.yaml",
        category="stems",
        source_text="Meta Demucs через audio-separator",
        license_text="Код MIT; веса Demucs распространяются авторами модели.",
    ),

    "stems_drums_detailed": ModelSpec(
        model_id="stems_drums_detailed",
        title="Барабаны по элементам",
        short_title="Разделитель — бочка, рабочий, томы и тарелки",
        backend="stems",
        size_text="около 1 ГБ; скачивается лениво",
        purpose="Разобрать барабанную партию на четыре элемента",
        description=(
            "Проверенный DrumSep через demucs-infer. Возвращает бочку, "
            "рабочий барабан, томы и тарелки. Модель и checkpoint "
            "остаются в кэше между запусками."
        ),
        warning=(
            "Лучший результат получается, если сначала подать уже выделенную "
            "общую дорожку барабанов. На полной песне возможны примеси."
        ),
        output_roles=("kick", "snare", "toms", "cymbals"),
        settings=STEM_SETTINGS,
        model_filename="demucs:drumsep",
        category="stems",
        source_text="DrumSep через demucs-infer 4.2.2",
        license_text="MIT; checkpoint проверяется по SHA-256 реестром.",
    ),
    "stems_cinematic": ModelSpec(
        model_id="stems_cinematic",
        title="Диалоги, музыка и звуковые эффекты",
        short_title="Разделитель — речь, музыка и эффекты",
        backend="stems",
        size_text="ансамбль из трёх Demucs-моделей",
        purpose="Фильмы, ролики, записи телевизора и игр",
        description=(
            "CDX23 DnR разделяет аудио на речь, музыку и звуковые эффекты. "
            "Полезно для очистки диалогов и подготовки аудиодескрипции."
        ),
        warning="Модель тяжелее обычного четырёхстемового разделения.",
        output_roles=("speech", "music", "sfx"),
        settings=STEM_SETTINGS,
        model_filename="demucs:cdx23_dnr",
        category="stems",
        source_text="CDX23 DnR через demucs-infer 4.2.2",
        license_text="Источник весов и SHA-256 закреплены в реестре пакета.",
    ),
    "stems_guitar": ModelSpec(
        model_id="stems_guitar",
        title="Гитара отдельно",
        short_title="Разделитель — гитара",
        backend="stems",
        size_text="около 1 ГБ",
        purpose="Выделение общей гитарной партии",
        description=(
            "Специализированная Mel-Band RoFormer для отделения гитары "
            "от остальной песни."
        ),
        warning=(
            "Похожие по тембру синтезаторы и струнные иногда попадают "
            "в гитарную дорожку."
        ),
        output_roles=("guitar", "other"),
        settings=STEM_SETTINGS,
        model_filename="becruily_guitar.ckpt",
        category="stems",
        source_text="becruily / Mel-Band RoFormer",
        license_text="Условия весов у автора; код запуска MIT.",
    ),
    "dereverb_big": ModelSpec(
        model_id="dereverb_big",
        title="Удаление реверберации — бережное",
        short_title="DeReverb — бережный",
        backend="stems",
        size_text="около 1 ГБ",
        purpose="Музыка и вокал с комнатным хвостом",
        description=(
            "Mel-Band RoFormer отделяет сухой сигнал от реверберации, "
            "стараясь сохранить естественность."
        ),
        warning="На очень длинном эхо может оставить часть хвоста.",
        output_roles=("dry", "reverb"),
        settings=STEM_SETTINGS,
        model_filename="dereverb_big_mbr_ep_362.ckpt",
        category="dereverb",
        source_text="Sucial / Mel-Band RoFormer",
        license_text="Условия весов у автора; код запуска MIT.",
    ),
    "dereverb_super": ModelSpec(
        model_id="dereverb_super",
        title="Удаление реверберации — максимальное",
        short_title="DeReverb — максимальный",
        backend="stems",
        size_text="большая модель, около 1–2 ГБ",
        purpose="Сильная реверберация и концертные записи",
        description=(
            "Более крупная модель с сильным подавлением комнатного хвоста."
        ),
        warning=(
            "Может сделать голос слишком сухим и ослабить пространственные "
            "инструменты."
        ),
        output_roles=("dry", "reverb"),
        settings=STEM_SETTINGS,
        model_filename="dereverb_super_big_mbr_ep_346.ckpt",
        category="dereverb",
        source_text="Sucial / Mel-Band RoFormer",
        license_text="Условия весов у автора; код запуска MIT.",
    ),
    "dereverb_echo": ModelSpec(
        model_id="dereverb_echo",
        title="Удаление реверберации и эха",
        short_title="DeReverb — эхо и хвост",
        backend="stems",
        size_text="около 1 ГБ",
        purpose="Длинное эхо, речь и вокал",
        description=(
            "Специализированная модель одновременно подавляет обычную "
            "реверберацию и заметные повторения эха."
        ),
        warning="На стереомузыке обязательно сравни результат с исходником.",
        output_roles=("dry", "reverb"),
        settings=STEM_SETTINGS,
        model_filename="dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt",
        category="dereverb",
        source_text="Sucial / Mel-Band RoFormer",
        license_text="Условия весов у автора; код запуска MIT.",
    ),
    "dereverb_fused": ModelSpec(
        model_id="dereverb_fused",
        title="Удаление реверберации — универсальное",
        short_title="DeReverb — универсальный",
        backend="stems",
        size_text="около 1 ГБ",
        purpose="Смешанные записи без очевидного типа реверберации",
        description=(
            "Fused-вариант объединяет поведение нескольких DeReverb-настроек."
        ),
        warning="Если результат слишком сухой, выбери бережную модель.",
        output_roles=("dry", "reverb"),
        settings=STEM_SETTINGS,
        model_filename="dereverb_echo_mbr_fused.ckpt",
        category="dereverb",
        source_text="Sucial / Mel-Band RoFormer",
        license_text="Условия весов у автора; код запуска MIT.",
    ),
    "stems_bleed_suppressor": ModelSpec(
        model_id="stems_bleed_suppressor",
        title="Подавление просачивания после разделения",
        short_title="Финальная очистка — Bleed Suppressor",
        backend="stems",
        size_text="около 1 ГБ",
        purpose="Подчистить уже выделенный вокал или инструмент",
        description=(
            "Второй проход после основного разделения. Удаляет тихие остатки "
            "других инструментов из выбранной дорожки."
        ),
        warning="Не предназначена для подачи полной песни как первого этапа.",
        output_roles=("clean", "bleed"),
        settings=STEM_SETTINGS,
        model_filename="mel_band_roformer_bleed_suppressor_v1.ckpt",
        category="stems",
        source_text="unwa-97chris / Mel-Band RoFormer",
        license_text="Условия весов у автора; код запуска MIT.",
    ),
    "stems_aspiration": ModelSpec(
        model_id="stems_aspiration",
        title="Дыхание и придыхания отдельно",
        short_title="Разделитель — дыхание вокалиста",
        backend="stems",
        size_text="около 1 ГБ",
        purpose="Вокал, подкасты и дикторская речь",
        description=(
            "Отделяет вдохи и придыхания, чтобы их можно было ослабить, "
            "а не вырезать вручную."
        ),
        warning="Слишком сильное удаление дыхания может звучать неестественно.",
        output_roles=("clean", "breaths"),
        settings=STEM_SETTINGS,
        model_filename=(
            "aspiration_mel_band_roformer_less_aggr_sdr_18.1201.ckpt"
        ),
        category="stems",
        source_text="Sucial / Mel-Band RoFormer",
        license_text="Условия весов у автора; код запуска MIT.",
    ),
}


def get_model(model_id: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_id]
    except KeyError as error:
        raise ValueError(f"Неизвестная модель: {model_id}") from error


def default_browser_settings() -> dict[str, dict[str, Any]]:
    return {
        model_id: {
            setting.key: setting.default for setting in model.settings
        }
        for model_id, model in MODEL_SPECS.items()
    }


def normalize_settings(
    model_id: str,
    raw_settings: Mapping[str, Any] | None,
) -> dict[str, Any]:
    model = get_model(model_id)
    raw = raw_settings or {}
    return {
        setting.key: _normalize_value(setting, raw.get(setting.key))
        for setting in model.settings
    }


def _normalize_value(setting: SettingSpec, raw: Any) -> Any:
    if raw is None:
        return setting.default
    if setting.kind == "choice":
        allowed = {value for _, value in setting.choices}
        return raw if raw in allowed else setting.default
    if setting.kind == "boolean":
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on", "да"}
        return bool(raw)
    if setting.kind == "integer":
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return setting.default
        return int(_clamp(value, setting.minimum, setting.maximum))
    if setting.kind == "number":
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return setting.default
        return float(_clamp(value, setting.minimum, setting.maximum))
    return setting.default


def _clamp(
    value: float,
    minimum: float | None,
    maximum: float | None,
) -> float:
    if minimum is not None:
        value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value
