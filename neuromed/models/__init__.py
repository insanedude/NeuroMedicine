"""Моделі та їх фабрика."""

from __future__ import annotations

from torch import nn

from neuromed.models.blocks import set_dropout
from neuromed.models.classifier import GlaucomaClassifier
from neuromed.models.segmentation import DiscSegmenter

MODEL_TYPES: dict[str, type[nn.Module]] = {
    DiscSegmenter.model_type: DiscSegmenter,
    GlaucomaClassifier.model_type: GlaucomaClassifier,
}

__all__ = ["DiscSegmenter", "GlaucomaClassifier", "MODEL_TYPES", "build_model", "create_model", "set_dropout"]


def build_model(arch: dict) -> nn.Module:
    """Відтворює модель (зокрема прорідженої ширини) з опису ``model.arch()``."""
    kwargs = dict(arch)
    model_type = kwargs.pop("type")
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Невідомий тип моделі: {model_type}")
    return MODEL_TYPES[model_type](**kwargs)


def create_model(model_cfg, in_channels: int) -> nn.Module:
    """Нова модель з конфігурації; за потреби — з трансплантованими вагами VGG-16."""
    arch = {"type": model_cfg.type, "base": model_cfg.base, "in_channels": in_channels,
            "num_outputs": len(model_cfg.labels), "dropout_kind": model_cfg.dropout_kind}
    if model_cfg.type == GlaucomaClassifier.model_type:
        arch["hidden_mult"] = model_cfg.hidden_mult
    model = build_model(arch)
    if model_cfg.pretrained:
        from neuromed.models.transfer import transplant_vgg16

        transplant_vgg16(model)
    return model
