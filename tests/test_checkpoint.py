"""Чекпойнт зменшеної моделі відтворює її точно (ширини шарів беруться з arch)."""

from __future__ import annotations

import pytest
import torch

from neuromed.compression.planning import plan_pruning
from neuromed.compression.surgery import apply_plan
from neuromed.models import DiscSegmenter, GlaucomaClassifier
from neuromed.training.checkpoint import load_checkpoint, save_checkpoint


@pytest.mark.parametrize("model, shape", [
    (DiscSegmenter(base=8), (1, 1, 64, 64)),
    (GlaucomaClassifier(base=8), (1, 3, 32, 32)),
])
@pytest.mark.parametrize("channels_last", [False, True])
def test_pruned_checkpoint_roundtrip(tmp_path, model, shape, channels_last):
    pruned = apply_plan(model, plan_pruning(model, 0.5, min_channels=1)).eval()
    if channels_last:
        pruned = pruned.to(memory_format=torch.channels_last)
    path = tmp_path / "model.pt"
    save_checkpoint(path, pruned, {"iteration": 3}, {"name": "test"})
    loaded, payload = load_checkpoint(path)
    assert payload["arch"]["widths"] == pruned.widths
    assert payload["meta"]["iteration"] == 3
    assert payload["config"]["name"] == "test"
    x = torch.randn(shape)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), pruned(x))


def test_rejects_foreign_file(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)
