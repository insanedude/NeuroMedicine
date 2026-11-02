"""Класифікатор кропів диска зорового нерва (глаукома / валідність кропу)."""

from __future__ import annotations

import torch
from torch import nn

from neuromed.compression.spec import PruneSpec
from neuromed.models.blocks import (
    ScheduledDropout,
    build_vgg_blocks,
    check_widths,
    encoder_last,
    encoder_widths,
    vgg_encoder_spec,
)


def default_widths(base: int, hidden_mult: int = 2) -> dict[str, int]:
    widths = encoder_widths(base)
    widths["fc"] = base * hidden_mult
    return widths


class GlaucomaClassifier(nn.Module):
    """Кодувальник VGG-16 (4 блоки) → глобальний середній пулінг → FC → логіти міток.

    Відповідає HandmadeGlaucomaClassifier з гілки 16_01_2026 (мітки glaucoma /
    atrophy / valid_image); глобальний пулінг робить модель незалежною від розміру входу.
    """

    model_type = "classifier"

    def __init__(self, base: int = 64, in_channels: int = 3, num_outputs: int = 2,
                 widths: dict[str, int] | None = None, dropout_kind: str = "element", hidden_mult: int = 2) -> None:
        super().__init__()
        w = check_widths(default_widths(base, hidden_mult), widths)
        self.base, self.in_channels, self.num_outputs = base, in_channels, num_outputs
        self.dropout_kind, self.hidden_mult, self.widths = dropout_kind, hidden_mult, w

        build_vgg_blocks(self, w, in_channels, "relu", dropout_kind)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(w["enc4.2"], w["fc"])
        self.fc_act = nn.ReLU(inplace=True)
        self.fc_drop = ScheduledDropout("element")
        self.fc2 = nn.Linear(w["fc"], num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.enc1(x))
        x = self.pool(self.enc2(x))
        x = self.pool(self.enc3(x))
        x = self.pool(self.enc4(x))
        x = torch.flatten(self.gap(x), 1)
        return self.fc2(self.fc_drop(self.fc_act(self.fc1(x))))

    def arch(self) -> dict:
        return {"type": self.model_type, "base": self.base, "in_channels": self.in_channels,
                "num_outputs": self.num_outputs, "dropout_kind": self.dropout_kind,
                "hidden_mult": self.hidden_mult, "widths": dict(self.widths)}

    def prune_spec(self) -> PruneSpec:
        groups, inputs = vgg_encoder_spec(self.in_channels)
        groups["fc"] = ["fc1"]
        inputs["fc1"] = [encoder_last("enc4")]
        inputs["fc2"] = ["fc"]
        return PruneSpec(groups, inputs)
