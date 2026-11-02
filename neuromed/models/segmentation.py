"""Сегментатор зорового нерва (диска) — «definitor» з вихідного репозиторію."""

from __future__ import annotations

import torch
from torch import nn

from neuromed.compression.spec import PruneSpec
from neuromed.models.blocks import (
    ConvUnit,
    build_vgg_blocks,
    check_widths,
    encoder_last,
    encoder_widths,
    vgg_encoder_spec,
)


def default_widths(base: int) -> dict[str, int]:
    widths = encoder_widths(base)
    widths.update({
        "mid": base * 8,
        "dec4.conv": base * 4, "dec4.up": base * 4,
        "dec3.conv": base * 2, "dec3.up": base * 2,
        "dec2.conv": base, "dec2.up": base,
    })
    return widths


class UpBlock(nn.Module):
    """Згортка 3×3 + транспонована згортка ×2 (як _upconv_block у FcnskipNerveDefinitor2)."""

    def __init__(self, cin: int, cmid: int, cout: int, dropout_kind: str, schedulable: bool = True) -> None:
        super().__init__()
        self.unit = ConvUnit(cin, cmid, "leaky", dropout_kind, schedulable)
        self.up = nn.ConvTranspose2d(cmid, cout, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.unit(x))


class DiscSegmenter(nn.Module):
    """Кодувальник VGG-16 (4 блоки) + декодувальник зі skip-з'єднаннями, вихід — логіти маски.

    Схема з інструкції/статті: 64-64 → 128-128 → 256×3 → 512×3 → проміжний 512 →
    три рівні декодування (skip з пулінгів 3, 2, 1) → 1×1 → маска диска.
    Ширина кожного шару задана словником ``widths``: після вирізання нейронів
    модель відтворюється з чекпойнта точно (без «латання» шарів, як у вихідному коді).
    Dropout стоїть після кожної активації, крім останнього блоку (стаття, розд. 4.1).
    """

    model_type = "segmenter"

    def __init__(self, base: int = 64, in_channels: int = 1, num_outputs: int = 1,
                 widths: dict[str, int] | None = None, dropout_kind: str = "element") -> None:
        super().__init__()
        w = check_widths(default_widths(base), widths)
        self.base, self.in_channels, self.num_outputs = base, in_channels, num_outputs
        self.dropout_kind, self.widths = dropout_kind, w

        build_vgg_blocks(self, w, in_channels, "leaky", dropout_kind)
        self.pool = nn.MaxPool2d(2, 2)
        self.mid = ConvUnit(w["enc4.2"], w["mid"], "leaky", dropout_kind)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec4 = UpBlock(w["mid"] + w["enc3.2"], w["dec4.conv"], w["dec4.up"], dropout_kind)
        self.dec3 = UpBlock(w["dec4.up"] + w["enc2.1"], w["dec3.conv"], w["dec3.up"], dropout_kind)
        self.dec2 = UpBlock(w["dec3.up"] + w["enc1.1"], w["dec2.conv"], w["dec2.up"], dropout_kind, schedulable=False)
        self.head = nn.Conv2d(w["dec2.up"], num_outputs, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p1 = self.pool(self.enc1(x))
        p2 = self.pool(self.enc2(p1))
        p3 = self.pool(self.enc3(p2))
        p4 = self.pool(self.enc4(p3))
        m = self.upsample(self.mid(p4))
        d4 = self.dec4(torch.cat([m, p3], dim=1))
        d3 = self.dec3(torch.cat([d4, p2], dim=1))
        d2 = self.dec2(torch.cat([d3, p1], dim=1))
        return self.head(d2)

    def arch(self) -> dict:
        return {"type": self.model_type, "base": self.base, "in_channels": self.in_channels,
                "num_outputs": self.num_outputs, "dropout_kind": self.dropout_kind, "widths": dict(self.widths)}

    def prune_spec(self) -> PruneSpec:
        groups, inputs = vgg_encoder_spec(self.in_channels)
        groups.update({
            "mid": ["mid.conv"],
            "dec4.conv": ["dec4.unit.conv"], "dec4.up": ["dec4.up"],
            "dec3.conv": ["dec3.unit.conv"], "dec3.up": ["dec3.up"],
            "dec2.conv": ["dec2.unit.conv"], "dec2.up": ["dec2.up"],
        })
        inputs.update({
            "mid.conv": [encoder_last("enc4")],
            "dec4.unit.conv": ["mid", encoder_last("enc3")],
            "dec4.up": ["dec4.conv"],
            "dec3.unit.conv": ["dec4.up", encoder_last("enc2")],
            "dec3.up": ["dec3.conv"],
            "dec2.unit.conv": ["dec3.up", encoder_last("enc1")],
            "dec2.up": ["dec2.conv"],
            "head": ["dec2.up"],
        })
        return PruneSpec(groups, inputs)
