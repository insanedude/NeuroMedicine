"""Нормування вирізуваних нейронів: штраф і обмеження норми."""

from __future__ import annotations

import pytest
import torch

from neuromed.compression.planning import plan_pruning
from neuromed.compression.regularizer import ChannelNormRegularizer
from neuromed.compression.spec import weight_dims
from neuromed.compression.surgery import apply_plan
from neuromed.models import DiscSegmenter


def _setup(include_consumers: bool = True):
    model = DiscSegmenter(base=8)
    plan = plan_pruning(model, 0.5, min_channels=1)
    return model, plan, ChannelNormRegularizer(model, plan, include_consumers)


def test_penalty_starts_at_one():
    _, _, reg = _setup()
    assert float(reg.penalty()) == pytest.approx(1.0, abs=1e-5)
    assert reg.mean_ratio() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("alpha", [0.5, 0.95])
def test_projection_caps_norm(alpha):
    _, _, reg = _setup()
    reg.project(alpha)
    for group in reg.terms:
        assert float(reg._group_norm(group) / reg.initial[group]) <= 1 - alpha + 1e-4


def test_projection_does_not_change_pruned_network():
    """Обмеження торкається лише ваг, які потім вирізаються, тож зменшена мережа та сама."""
    model, plan, reg = _setup()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        before = apply_plan(model, plan).eval()(x)
        reg.project(0.95)
        after = apply_plan(model, plan).eval()(x)
    torch.testing.assert_close(before, after)


def test_penalty_gradient_only_on_removed_channels():
    model, plan, reg = _setup(include_consumers=False)
    reg.penalty().backward()
    modules = dict(model.named_modules())
    spec = model.prune_spec()
    for group, keep in plan.keep.items():
        for name in spec.groups[group]:
            module = modules[name]
            out_dim, _ = weight_dims(module)
            grad = module.weight.grad
            assert grad.index_select(out_dim, keep).abs().max() == 0
            assert grad.index_select(out_dim, plan.remove[group]).abs().sum() > 0
