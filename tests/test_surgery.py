"""Вирізання нейронів: зменшена модель має рахувати те саме, що вихідна з обнуленими нейронами."""

from __future__ import annotations

import copy

import pytest
import torch

from neuromed.compression.planning import keep_count, plan_pruning
from neuromed.compression.spec import validate_spec
from neuromed.compression.surgery import apply_plan, zero_removed
from neuromed.models import DiscSegmenter, GlaucomaClassifier
from neuromed.utils import count_parameters

MODELS = [
    pytest.param(lambda: DiscSegmenter(base=8, in_channels=1), (2, 1, 64, 64), id="segmenter"),
    pytest.param(lambda: DiscSegmenter(base=8, in_channels=3, dropout_kind="channel"), (2, 3, 32, 32),
                 id="segmenter-rgb"),
    pytest.param(lambda: GlaucomaClassifier(base=8, in_channels=3, num_outputs=2), (2, 3, 32, 32), id="classifier"),
]


@pytest.mark.parametrize("factory, shape", MODELS)
def test_pruned_model_matches_zeroed_original(factory, shape):
    model = factory().eval()
    plan = plan_pruning(model, 0.5, min_channels=1)
    pruned = apply_plan(model, plan).eval()
    reference = copy.deepcopy(model)
    zero_removed(reference, plan)
    x = torch.randn(shape)
    with torch.no_grad():
        torch.testing.assert_close(pruned(x), reference(x), atol=1e-5, rtol=1e-4)
    assert pruned.widths == plan.new_widths(model.widths)
    assert count_parameters(pruned) < count_parameters(model)


@pytest.mark.parametrize("factory, shape", MODELS)
def test_repeated_pruning_keeps_spec_consistent(factory, shape):
    model = factory()
    for ratio in (0.5, 0.25):
        model = apply_plan(model, plan_pruning(model, ratio, min_channels=1))
        validate_spec(model, model.prune_spec(), model.widths)
    with torch.no_grad():
        assert torch.isfinite(model.eval()(torch.randn(shape))).all()


def test_plan_removes_lowest_norm_channels():
    model = DiscSegmenter(base=8)
    group = "enc1.0"
    conv = dict(model.named_modules())[model.prune_spec().groups[group][0]]
    with torch.no_grad():
        conv.weight[[1, 5]] *= 100.0
        conv.weight[[0, 2]] *= 0.001
    plan = plan_pruning(model, 0.5, min_channels=1)
    kept = set(plan.keep[group].tolist())
    assert {1, 5} <= kept
    assert not ({0, 2} & kept)
    assert sorted(kept | set(plan.remove[group].tolist())) == list(range(model.widths[group]))


def test_keep_count_respects_rounding_and_minimum():
    assert keep_count(64, 0.5) == 32
    assert keep_count(8, 0.5, min_channels=6) == 6
    assert keep_count(30, 0.5, round_to=8) == 16
    assert keep_count(3, 0.5, min_channels=4) == 3


def test_groups_at_minimum_width_are_not_cut():
    plan = plan_pruning(DiscSegmenter(base=4), 0.5, min_channels=4)
    assert "enc1.0" not in plan.remove and "enc4.0" in plan.remove  # глибші шари ще ширші за мінімум
    narrow = DiscSegmenter(base=4, widths={group: 4 for group in DiscSegmenter(base=4).widths})
    assert plan_pruning(narrow, 0.5, min_channels=4).is_empty
