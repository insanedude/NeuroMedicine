"""Transfer learning: «трансплантація» перших чотирьох блоків VGG-16 у кодувальник."""

from __future__ import annotations

import torch
from torch import nn

from neuromed.models.blocks import transfer_convs
from neuromed.utils import get_logger


def load_vgg16_convs() -> list[nn.Conv2d]:
    """Згортки VGG-16, натренованої на ImageNet (ваги завантажує torchvision, ~530 МБ одноразово)."""
    from torchvision.models import VGG16_Weights, vgg16

    features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
    return [m for m in features if isinstance(m, nn.Conv2d)]


@torch.no_grad()
def transplant_vgg16(model: nn.Module, source_convs: list[nn.Conv2d] | None = None) -> int:
    """Копіює ваги VGG-16 у блоки enc1..enc4 моделі; повертає кількість перенесених шарів.

    * Сірий вхід (1 канал): фільтри першого шару сумуються по RGB — для зображення
      з трьома однаковими каналами це математично те саме, але втричі дешевше.
    * Якщо модель вужча за VGG-16 (основа < 64), беруться фільтри з найбільшою
      нормою, а входи наступного шару — відповідно до вибраних каналів.
    """
    targets = transfer_convs(model)
    sources = source_convs if source_convs is not None else load_vgg16_convs()
    if len(sources) < len(targets):
        raise ValueError("У джерелі замало згорткових шарів для трансплантації")
    prev_idx: torch.Tensor | None = None
    for dst, src in zip(targets, sources):
        weight, bias = src.weight.detach().cpu(), src.bias.detach().cpu()
        if prev_idx is None:
            if dst.in_channels == 1 and weight.shape[1] == 3:
                weight = weight.sum(dim=1, keepdim=True)
            elif dst.in_channels != weight.shape[1]:
                raise ValueError(f"Вхідних каналів {dst.in_channels}, у VGG-16 — {weight.shape[1]}")
        else:
            weight = weight.index_select(1, prev_idx)
        if dst.out_channels > weight.shape[0]:
            raise ValueError(f"Шар ширший за VGG-16 ({dst.out_channels} > {weight.shape[0]})")
        if dst.out_channels < weight.shape[0]:
            idx = weight.flatten(1).norm(dim=1).topk(dst.out_channels).indices.sort().values
        else:
            idx = torch.arange(weight.shape[0])
        dst.weight.copy_(weight.index_select(0, idx).to(dst.weight.device))
        dst.bias.copy_(bias.index_select(0, idx).to(dst.bias.device))
        prev_idx = idx
    get_logger().info("Трансплантовано %d згорткових шарів VGG-16 (блоки 1–4)", len(targets))
    return len(targets)
