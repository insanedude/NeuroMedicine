"""Фізичне вирізання нейронів: «вирізаєм непотрібні нейрони нафіг» (крок 4.4 інструкції)."""

from __future__ import annotations

import torch
from torch import nn

from neuromed.compression.planning import PrunePlan
from neuromed.compression.spec import PRUNABLE_TYPES, segment_offsets, weight_dims


def _input_indices(layout: list[str | int], plan: PrunePlan, widths: dict[str, int]) -> torch.Tensor | None:
    """Індекси вхідних каналів споживача, що лишаються (з урахуванням конкатенацій)."""
    parts, changed = [], False
    for seg, offset, size in segment_offsets(layout, widths):
        idx = None if isinstance(seg, int) else plan.keep.get(seg)
        if idx is None:
            idx = torch.arange(size)
        else:
            changed = True
        parts.append(idx + offset)
    return torch.cat(parts) if changed else None


def apply_plan(model: nn.Module, plan: PrunePlan) -> nn.Module:
    """Будує нову, меншу модель того ж типу і переносить у неї зрізані ваги.

    Результат еквівалентний вихідній моделі, у якій ваги вирізаних нейронів
    обнулені (це перевіряє тест), тому після якісного «зганяння» досвіду нормуванням
    вирізання майже не змінює передбачень.
    """
    from neuromed.models import build_model

    spec = model.prune_spec()
    widths = model.widths
    arch = model.arch()
    arch["widths"] = plan.new_widths(widths)
    new_model = build_model(arch)

    out_index = {name: plan.keep[g] for g, names in spec.groups.items() if g in plan.keep for name in names}
    in_index = {}
    for name, layout in spec.inputs.items():
        idx = _input_indices(layout, plan, widths)
        if idx is not None:
            in_index[name] = idx

    state = {}
    for name, module in model.named_modules():
        if not isinstance(module, PRUNABLE_TYPES):
            continue
        out_dim, in_dim = weight_dims(module)
        weight = module.weight.detach()
        bias = module.bias.detach() if module.bias is not None else None
        if name in out_index:
            idx = out_index[name].to(weight.device)
            weight = weight.index_select(out_dim, idx)
            bias = bias.index_select(0, idx) if bias is not None else None
        if name in in_index:
            weight = weight.index_select(in_dim, in_index[name].to(weight.device))
        state[f"{name}.weight"] = weight.clone()
        if bias is not None:
            state[f"{name}.bias"] = bias.clone()
    for key, value in model.state_dict().items():
        state.setdefault(key, value.clone())

    new_model.load_state_dict(state, strict=True)
    device = next(model.parameters()).device
    new_model.to(device)
    new_model.train(model.training)
    return new_model


@torch.no_grad()
def zero_removed(model: nn.Module, plan: PrunePlan) -> None:
    """Обнуляє вихідні ваги вирізуваних каналів (для перевірки еквівалентності в тестах)."""
    spec = model.prune_spec()
    modules = dict(model.named_modules())
    for group, idx in plan.remove.items():
        for name in spec.groups[group]:
            module = modules[name]
            out_dim, _ = weight_dims(module)
            index = idx.to(module.weight.device)
            module.weight.index_fill_(out_dim, index, 0.0)
            if module.bias is not None:
                module.bias.index_fill_(0, index, 0.0)
