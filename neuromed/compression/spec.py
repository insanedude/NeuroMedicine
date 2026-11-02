"""Опис структури моделі для структурного прорідження.

Модель повідомляє, які шари *продукують* кожну групу каналів (нейронів) і з яких
груп складається вхід кожного шару-*споживача*. Цього достатньо, щоб коректно
вирізати нейрони навіть там, де виходи конкатенуються (skip-з'єднання U-Net):
зсуви каналів у конкатенації обчислюються з поточних ширин груп.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

PRUNABLE_TYPES = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)


@dataclass
class PruneSpec:
    # група каналів → шари, вихідні канали яких утворюють цю групу
    groups: dict[str, list[str]]
    # шар-споживач → розкладка його вхідних каналів: назви груп або фіксована кількість (вхід зображення)
    inputs: dict[str, list[str | int]]


def weight_dims(module: nn.Module) -> tuple[int, int]:
    """(вимір вихідних каналів, вимір вхідних каналів) у тензорі ваг шару."""
    if isinstance(module, nn.ConvTranspose2d):
        return 1, 0  # ваги ConvTranspose2d: [in, out, kH, kW]
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        return 0, 1
    raise TypeError(f"Непідтримуваний тип шару для прорідження: {type(module).__name__}")


def segment_offsets(layout: list[str | int], widths: dict[str, int]) -> list[tuple[str | int, int, int]]:
    """[(сегмент, зсув, розмір)] для розкладки входу споживача."""
    result, offset = [], 0
    for seg in layout:
        size = seg if isinstance(seg, int) else widths[seg]
        result.append((seg, offset, size))
        offset += size
    return result


def validate_spec(model: nn.Module, spec: PruneSpec, widths: dict[str, int]) -> None:
    """Перевіряє, що опис узгоджений з реальними формами ваг (ловить помилки в моделях)."""
    modules = dict(model.named_modules())
    for group, producers in spec.groups.items():
        for name in producers:
            module = modules[name]
            out_dim, _ = weight_dims(module)
            if module.weight.shape[out_dim] != widths[group]:
                raise ValueError(f"{name}: вихідних каналів {module.weight.shape[out_dim]}, "
                                 f"група {group}={widths[group]}")
    for name, layout in spec.inputs.items():
        module = modules[name]
        _, in_dim = weight_dims(module)
        total = sum(size for _, _, size in segment_offsets(layout, widths))
        if module.weight.shape[in_dim] != total:
            raise ValueError(f"{name}: вхідних каналів {module.weight.shape[in_dim]}, за описом {total}")
