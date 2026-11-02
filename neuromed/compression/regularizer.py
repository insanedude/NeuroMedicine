"""Нормування по вагах нейронів, які будуть вирізані (кроки 4.2–4.3 інструкції).

Для кожної групи береться спільна (group-lasso) L2-норма всіх ваг вирізуваних
нейронів: їхніх фільтрів і bias та, за бажанням, вхідних ваг наступних шарів, що
читають ці канали. Норма ділиться на початкове значення, тож штраф безрозмірний
(≈1 на старті, → 0, коли нейрони «спорожніли»).

Два режими «частки нормування» α (0 → 0.95):
* ``penalty``    — L = (1 − α)·L_задачі + α·strength·R: штраф поступово витісняє
                   задачу саме з вирізуваних ваг, решта мережі переймає їхній досвід;
* ``projection`` — після кожного кроку оптимізатора норма групи обмежується
                   (1 − α)·початкової, тобто наприкінці лишається ≤ 5 % від неї;
* ``both``       — штраф плюс обмеження (типово): штраф м'яко переносить знання,
                   обмеження гарантує, що вирізання наприкінці майже не змінює вихід.
"""

from __future__ import annotations

import torch
from torch import nn

from neuromed.compression.planning import PrunePlan
from neuromed.compression.spec import segment_offsets, weight_dims


class ChannelNormRegularizer:
    def __init__(self, model: nn.Module, plan: PrunePlan, include_consumers: bool = True) -> None:
        spec = model.prune_spec()
        modules = dict(model.named_modules())
        widths = model.widths
        self.terms: dict[str, list[tuple[nn.Parameter, int, torch.Tensor]]] = {}
        for group, idx in plan.remove.items():
            terms = []
            for name in spec.groups[group]:
                module = modules[name]
                out_dim, _ = weight_dims(module)
                device = module.weight.device
                terms.append((module.weight, out_dim, idx.to(device)))
                if module.bias is not None:
                    terms.append((module.bias, 0, idx.to(device)))
            if include_consumers:
                for name, layout in spec.inputs.items():
                    for seg, offset, _ in segment_offsets(layout, widths):
                        if seg == group:
                            module = modules[name]
                            _, in_dim = weight_dims(module)
                            terms.append((module.weight, in_dim, (idx + offset).to(module.weight.device)))
            self.terms[group] = terms
        with torch.no_grad():
            self.initial = {g: self._group_norm(g).clamp_min(1e-12) for g in self.terms}

    def _group_norm(self, group: str) -> torch.Tensor:
        total = sum(p.index_select(d, i).float().pow(2).sum() for p, d, i in self.terms[group])
        return torch.sqrt(total)

    def penalty(self) -> torch.Tensor:
        """Середня відносна норма вирізуваних груп (диференційовна)."""
        values = [self._group_norm(g) / self.initial[g] for g in self.terms]
        return torch.stack(values).mean()

    @torch.no_grad()
    def project(self, alpha: float) -> None:
        """Обмежує норму кожної групи значенням (1 − α)·початкової."""
        for group, terms in self.terms.items():
            cap = (1.0 - alpha) * self.initial[group]
            norm = self._group_norm(group)
            if norm > cap:
                scale = (cap / norm).item()
                for p, d, i in terms:
                    p.index_copy_(d, i, p.index_select(d, i) * scale)

    @torch.no_grad()
    def mean_ratio(self) -> float:
        if not self.terms:
            return 0.0
        return float(torch.stack([self._group_norm(g) / self.initial[g] for g in self.terms]).mean())
