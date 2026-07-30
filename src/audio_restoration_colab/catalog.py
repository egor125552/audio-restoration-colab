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
            "Модель обучена на речи. Музыкальная версия пока не выпущена, "
            "поэтому на музыке возможны металлический звук и потеря стерео."
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
                help_text="Автоматический режим подходит в большинстве случаев.",
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
            "Однопроходная дорисовка верхних частот до 48 кГц. Стереоканалы "
            "обрабатываются отдельно и сохраняются."
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
                    "Полезно после MP3 и нейросетевого разделения. Выключи, "
                    "если исходник уже имеет ровный низкочастотный срез."
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
            "Многопроходная дорисовка до 48 кГц. Даёт разные варианты при "
            "разном зерне и заметно медленнее FlashSR."
        ),
        warning=(
            "Авторы предупреждают: модель плохо понимает дырявый срез после "
            "MP3 и сильные искажения. Для таких файлов оставь фильтр включённым."
        ),
        output_roles=("restored",),
        settings=(
            SettingSpec(
                key="mode",
                label="Тип звука",
                kind="choice",
                default="basic",
                choices=(("Обычный звук и музыка", "basic"), ("Речь", "speech")),
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
                help_text="Рекомендуется для MP3 и разделённых нейросетью файлов.",
            ),
        ),
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
