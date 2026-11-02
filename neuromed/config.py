"""Конфігурація: YAML → вкладені dataclass-и з перевіркою ключів.

На відміну від вихідного репозиторію (де вкладені словники не перетворювались на
об'єкти, а одруківки мовчки ігнорувались), тут невідомий ключ — це помилка, а
значення на кшталт ``1e-4`` (які PyYAML читає як рядок) приводяться до float.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Помилка в конфігурації."""


@dataclass
class PreprocessConfig:
    """Попередня обробка знімка (стаття, розд. 3.2)."""

    hsv_s_equalize: bool = True  # вирівнювання гістограми каналу насиченості (HSV)
    clahe: bool = True  # CLAHE на каналі яскравості L (LAB)
    clahe_clip: float = 2.0
    clahe_grid: int = 16
    grayscale: bool = True  # відтінки сірого → модель з 1 вхідним каналом


@dataclass
class AugmentConfig:
    """Модифікації вхідних даних: поворот, скошення, розмиття, артефакти тощо."""

    enabled: bool = True
    hflip: float = 0.5
    vflip: float = 0.5
    rotate_deg: float = 20.0
    shear_deg: float = 8.0
    scale_min: float = 0.9
    scale_max: float = 1.1
    translate: float = 0.05  # частка розміру зображення
    brightness: float = 0.15
    contrast: float = 0.15
    saturation: float = 0.1  # лише для кольорових входів
    gamma: float = 0.15
    blur_prob: float = 0.25
    blur_sigma_max: float = 1.2
    noise_std: float = 0.02
    artifact_prob: float = 0.3  # плями/відблиски
    artifact_max: int = 4


@dataclass
class HardExamplesConfig:
    """Буфер найскладніших зразків (стаття, розд. 4): зважений випадковий повтор."""

    enabled: bool = True
    replay_prob: float = 0.5  # ймовірність додаткового кроку на складних зразках після кожного батча
    max_size: int = 0  # 0 → розмір одного фолда


@dataclass
class DataConfig:
    datasets: list[str] = field(default_factory=list)  # теки, підготовлені командою `prepare`
    input_size: int = 288  # сторона входу мережі (кратна 16)
    val_fraction: float = 0.15  # фіксована контрольна вибірка (критерії зупинки/прийняття)
    test_fraction: float = 0.15  # тестова вибірка (лише для звітів)
    chunks: int = 5  # кількість фолдів/непересічних частин навчальної вибірки
    epoch_strategy: str = "kfold"  # kfold | alternate | full
    batch_size: int = 8
    split_seed: int = 42
    on_device: bool = True  # тримати весь (невеликий) датасет у пам'яті GPU
    cache_dir: str = ""  # "" → <перший датасет>/.cache
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    hard_examples: HardExamplesConfig = field(default_factory=HardExamplesConfig)


@dataclass
class ModelConfig:
    type: str = "segmenter"  # segmenter | classifier
    base: int = 64  # «основа» моделі: 64 → 32 → 16 → 8 у статті
    pretrained: bool = True  # трансплантація ваг VGG-16 у чотири блоки кодувальника
    dropout_kind: str = "element"  # element (як у вихідному коді) | channel
    labels: list[str] = field(default_factory=lambda: ["disc"])
    hidden_mult: int = 2  # класифікатор: ширина прихованого FC-шару = hidden_mult * base


@dataclass
class LossConfig:
    pos_weight: float = 3.0  # вага позитивних пікселів/міток у BCE
    neg_weight: float = 1.0
    dice_weight: float = 1.0  # лише для сегментації
    label_pos_weight: dict[str, float] = field(default_factory=dict)  # класифікатор: вага позитиву по мітках


@dataclass
class TrainConfig:
    """Перша ітерація: transfer learning і донавчання до достатніх показників."""

    max_epochs: int = 80
    min_epochs: int = 10
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    amp: bool = True  # змішана точність на CUDA
    channels_last: bool = True  # формат NHWC на CUDA: тензорні ядра без перестановок (~20 % швидше)
    grad_clip: float = 5.0
    freeze_transferred_epochs: int = 2  # спершу вчимо лише нові шари поверх трансплантованих
    dropout: float = 0.0
    patience: int = 10
    tolerance: float = 1e-3
    eval_every: int = 1
    target: dict[str, float] = field(default_factory=dict)  # «достатні показники»: метрика → мінімум
    selection: list[str] = field(default_factory=lambda: ["det_recall", "dice"])  # сума → вибір найкращої епохи


@dataclass
class AcceptanceConfig:
    """Коли стиснена мережа ще «не фейлить»."""

    minimum: dict[str, float] = field(default_factory=dict)  # абсолютні мінімуми метрик
    max_drop: dict[str, float] = field(default_factory=dict)  # допустиме падіння відносно першої ітерації


@dataclass
class CompressConfig:
    """Алгоритм зжимання з інструкції (кроки 4.1–4.6)."""

    prune_ratio: float = 0.5  # частка нейронів, що вирізається за ітерацію (64 → 32 → 16 → 8)
    min_prune_ratio: float = 0.125
    retry_smaller_ratio: bool = True  # після провалу повторити з удвічі меншою часткою
    importance: str = "l2"  # l1 | l2 — норма ваг нейрона
    round_to: int = 1  # округлення кількості каналів, що лишаються
    min_channels: int = 4
    max_iterations: int = 8
    spread_epochs: int = 8  # 4.1: dropout 0.1–0.2 → 0.5
    dropout_start: float = 0.15
    dropout_max: float = 0.5
    squeeze_epochs: int = 16  # 4.2–4.3: dropout 0.5 → 0, частка нормування 0 → 0.95
    norm_max: float = 0.95
    norm_strength: float = 1.0
    norm_method: str = "both"  # penalty (частка у функції втрат) | projection (жорстке обмеження норми) | both
    norm_include_consumers: bool = True  # нормувати також вхідні ваги наступних шарів
    lr_scale_after_cut: float = 0.5  # 4.4: «крок поменше» після вирізання
    min_lr: float = 1e-6
    stabilize_min_epochs: int = 4  # 4.5: дотренування після стресу
    stabilize_max_epochs: int = 40
    stabilize_patience: int = 6
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)


@dataclass
class DistillConfig:
    """Дистиляція знань: вчитель — модель першої ітерації (стаття, розд. 3.3)."""

    enabled: bool = False
    alpha: float = 0.5  # частка втрат дистиляції
    temperature: float = 2.0


@dataclass
class EvalConfig:
    threshold: float = 0.5
    min_component_frac: float = 0.0005  # мінімальна площа компоненти маски (частка кадру)
    detection_iou: float = 0.2  # IoU, з якого компонента вважається знайденим диском
    calibrate_recall: dict[str, float] = field(default_factory=dict)  # мітка → цільова повнота для порогу
    edge_cases: int = 10  # кількість «крайніх випадків» для перевірки
    batch_size: int = 16
    benchmark_threads: int = 4
    benchmark_runs: int = 20


@dataclass
class Config:
    name: str = "experiment"
    output_dir: str = "runs"
    seed: int = 42
    device: str = "auto"  # auto | cpu | cuda
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    compress: CompressConfig = field(default_factory=CompressConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name


_CHOICES = {
    ("model", "type"): {"segmenter", "classifier"},
    ("model", "dropout_kind"): {"element", "channel"},
    ("data", "epoch_strategy"): {"kfold", "alternate", "full"},
    ("compress", "importance"): {"l1", "l2"},
    ("compress", "norm_method"): {"penalty", "projection", "both"},
}


def _coerce(tp: Any, value: Any, path: str) -> Any:
    """Приводить прості типи (зокрема рядки '1e-4' → float)."""
    if tp is float and isinstance(value, (int, str)) and not isinstance(value, bool):
        try:
            return float(value)
        except ValueError as exc:
            raise ConfigError(f"{path}: очікувалось число, отримано {value!r}") from exc
    if tp is int and isinstance(value, float) and value.is_integer():
        return int(value)
    origin = typing.get_origin(tp)
    if origin is dict and isinstance(value, dict):
        _, vt = typing.get_args(tp)
        return {str(k): _coerce(vt, v, f"{path}.{k}") for k, v in value.items()}
    if origin is list and isinstance(value, (list, tuple)):
        (it,) = typing.get_args(tp)
        return [_coerce(it, v, f"{path}[{i}]") for i, v in enumerate(value)]
    return value


def _build(cls: type, data: Any, path: str) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path or 'config'}: очікувався словник, отримано {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ConfigError(f"Невідомі ключі в '{path or 'корені'}': {', '.join(unknown)}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        sub = f"{path}.{f.name}" if path else f.name
        tp = hints[f.name]
        if dataclasses.is_dataclass(tp):
            kwargs[f.name] = _build(tp, data[f.name], sub)
        else:
            kwargs[f.name] = _coerce(tp, data[f.name], sub)
    return cls(**kwargs)


def validate(cfg: Config) -> None:
    for (section, key), allowed in _CHOICES.items():
        value = getattr(getattr(cfg, section), key)
        if value not in allowed:
            raise ConfigError(f"{section}.{key}={value!r}; допустимо: {', '.join(sorted(allowed))}")
    if cfg.data.input_size % 16:
        raise ConfigError("data.input_size має бути кратним 16 (чотири пулінги)")
    if not 0.0 < cfg.compress.prune_ratio < 1.0:
        raise ConfigError("compress.prune_ratio має бути в (0, 1)")
    if not 0.0 <= cfg.compress.norm_max < 1.0:
        raise ConfigError("compress.norm_max має бути в [0, 1)")
    if cfg.data.chunks < 2 and cfg.data.epoch_strategy != "full":
        raise ConfigError("data.chunks має бути ≥ 2 для стратегій kfold/alternate")
    if cfg.model.type == "segmenter" and len(cfg.model.labels) != 1:
        raise ConfigError("сегментатор має рівно одну мітку (маску диска)")


def apply_overrides(data: dict, overrides: list[str]) -> dict:
    """Застосовує перевизначення виду ``train.lr=3e-4`` до сирого словника."""
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Перевизначення має вигляд ключ=значення: {item!r}")
        key, raw = item.split("=", 1)
        node = data
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"{key}: '{part}' не є секцією")
        node[parts[-1]] = yaml.safe_load(raw)
    return data


def config_from_dict(data: dict) -> Config:
    cfg = _build(Config, data, "")
    validate(cfg)
    return cfg


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> Config:
    data: dict = {}
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    apply_overrides(data, overrides or [])
    return config_from_dict(data)


def config_to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


def save_config(cfg: Config, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False, allow_unicode=True)
