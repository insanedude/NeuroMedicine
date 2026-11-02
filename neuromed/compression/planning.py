"""Вибір нейронів (каналів), що будуть вирізані на ітерації стиснення."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from neuromed.compression.spec import weight_dims


@dataclass
class PrunePlan:
    """Для кожної групи: індекси каналів, що лишаються, і тих, що вирізаються (відсортовані)."""

    ratio: float
    keep: dict[str, torch.Tensor]
    remove: dict[str, torch.Tensor]

    @property
    def is_empty(self) -> bool:
        return not self.remove

    def new_widths(self, widths: dict[str, int]) -> dict[str, int]:
        result = dict(widths)
        for group, idx in self.keep.items():
            result[group] = int(idx.numel())
        return result

    def removed_channels(self) -> int:
        return int(sum(idx.numel() for idx in self.remove.values()))


def channel_scores(model: nn.Module, group: str, method: str = "l2") -> torch.Tensor:
    """Важливість каналів групи — норма їхніх вихідних ваг (з bias)."""
    spec = model.prune_spec()
    modules = dict(model.named_modules())
    scores: torch.Tensor | None = None
    for name in spec.groups[group]:
        module = modules[name]
        out_dim, _ = weight_dims(module)
        w = module.weight.detach().transpose(0, out_dim).flatten(1).float()
        if module.bias is not None:
            w = torch.cat([w, module.bias.detach().float()[:, None]], dim=1)
        s = w.abs().sum(dim=1) if method == "l1" else w.pow(2).sum(dim=1).sqrt()
        scores = s if scores is None else scores + s
    assert scores is not None
    return scores.cpu()


def keep_count(width: int, ratio: float, round_to: int = 1, min_channels: int = 1) -> int:
    n = int(round(width * (1.0 - ratio)))
    if round_to > 1:
        n = max(round_to, int(round(n / round_to)) * round_to)
    n = max(n, min(min_channels, width))
    return min(n, width)


def plan_pruning(model: nn.Module, ratio: float, method: str = "l2", round_to: int = 1,
                 min_channels: int = 4) -> PrunePlan:
    """Рівномірно зменшує кожну групу на ``ratio`` (основа 64 → 32 при ratio = 0.5),
    вирізаючи канали з найменшою нормою ваг."""
    spec = model.prune_spec()
    keep: dict[str, torch.Tensor] = {}
    remove: dict[str, torch.Tensor] = {}
    for group, width in model.widths.items():
        if group not in spec.groups:
            continue
        n_keep = keep_count(width, ratio, round_to, min_channels)
        if n_keep >= width:
            continue
        order = torch.argsort(channel_scores(model, group, method), descending=True)
        keep[group] = order[:n_keep].sort().values
        remove[group] = order[n_keep:].sort().values
    return PrunePlan(ratio=ratio, keep=keep, remove=remove)
