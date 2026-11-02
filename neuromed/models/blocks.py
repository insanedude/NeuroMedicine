"""Будівельні блоки: згорткова одиниця з керованим dropout і VGG-подібний кодувальник."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# План кодувальника VGG-16 до четвертого пулінгу: (блок, кількість згорток, множник ширини).
# Саме ці чотири логічні блоки (64-64, 128-128, 256-256-256, 512-512-512) трансплантуються з VGG-16.
VGG_ENCODER_PLAN: tuple[tuple[str, int, int], ...] = (
    ("enc1", 2, 1),
    ("enc2", 2, 2),
    ("enc3", 3, 4),
    ("enc4", 3, 8),
)


def encoder_widths(base: int) -> dict[str, int]:
    return {f"{name}.{i}": base * mult for name, n, mult in VGG_ENCODER_PLAN for i in range(n)}


def encoder_last(block: str) -> str:
    """Назва групи останньої згортки блоку (її вихід іде в пулінг і skip-з'єднання)."""
    n = {name: n for name, n, _ in VGG_ENCODER_PLAN}[block]
    return f"{block}.{n - 1}"


class ScheduledDropout(nn.Module):
    """Dropout, імовірність якого виставляє розклад стиснення (p = 0 → тотожне відображення).

    Модуль присутній у моделі завжди, тож state_dict не змінюється при ввімкненні
    dropout (у вихідному коді для цього модель перебудовувалась через add_dropout).
    ``schedulable=False`` — для останнього блоку, де dropout не застосовується.
    """

    def __init__(self, kind: str = "element", schedulable: bool = True) -> None:
        super().__init__()
        self.kind = kind
        self.schedulable = schedulable
        self.p = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        if self.kind == "channel" and x.dim() == 4:
            return F.dropout2d(x, self.p, training=True)
        return F.dropout(x, self.p, training=True)

    def extra_repr(self) -> str:
        return f"kind={self.kind}, p={self.p:.3f}, schedulable={self.schedulable}"


class ConvUnit(nn.Module):
    """Conv3×3 → активація → dropout."""

    def __init__(self, cin: int, cout: int, act: str = "leaky", dropout_kind: str = "element",
                 schedulable: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.01, inplace=True) if act == "leaky" else nn.ReLU(inplace=True)
        self.drop = ScheduledDropout(dropout_kind, schedulable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.conv(x)))


def build_vgg_blocks(model: nn.Module, widths: dict[str, int], in_channels: int, act: str,
                     dropout_kind: str) -> None:
    """Додає до моделі блоки enc1..enc4 (пулінг — у forward моделі, щоб брати skip-з'єднання)."""
    cin = in_channels
    for name, n, _ in VGG_ENCODER_PLAN:
        units = []
        for i in range(n):
            cout = widths[f"{name}.{i}"]
            units.append(ConvUnit(cin, cout, act, dropout_kind))
            cin = cout
        model.add_module(name, nn.Sequential(*units))


def vgg_encoder_spec(in_channels: int) -> tuple[dict[str, list[str]], dict[str, list[str | int]]]:
    groups: dict[str, list[str]] = {}
    inputs: dict[str, list[str | int]] = {}
    prev: str | int = in_channels
    for name, n, _ in VGG_ENCODER_PLAN:
        for i in range(n):
            group, module = f"{name}.{i}", f"{name}.{i}.conv"
            groups[group] = [module]
            inputs[module] = [prev]
            prev = group
    return groups, inputs


def transfer_convs(model: nn.Module) -> list[nn.Conv2d]:
    """Згортки кодувальника в порядку VGG-16 (цілі для трансплантації)."""
    return [getattr(model, name)[i].conv for name, n, _ in VGG_ENCODER_PLAN for i in range(n)]


def set_dropout(model: nn.Module, p: float) -> None:
    for module in model.modules():
        if isinstance(module, ScheduledDropout) and module.schedulable:
            module.p = float(p)


def check_widths(defaults: dict[str, int], widths: dict[str, int] | None) -> dict[str, int]:
    result = dict(defaults)
    if widths:
        unknown = set(widths) - set(defaults)
        if unknown:
            raise ValueError(f"Невідомі групи каналів: {sorted(unknown)}")
        result.update({k: int(v) for k, v in widths.items()})
    bad = [k for k, v in result.items() if v < 1]
    if bad:
        raise ValueError(f"Нульова ширина груп: {bad}")
    return result
