"""Функції втрат. Усі повертають втрату для кожного зразка [N] — це потрібно буферу складних зразків."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from neuromed.config import Config


class SegmentationLoss(nn.Module):
    """Зважена BCE (WeightedBCELoss з вихідного коду, але з логітів — стабільніше) + soft Dice."""

    def __init__(self, pos_weight: float, neg_weight: float, dice_weight: float) -> None:
        super().__init__()
        self.pos_weight, self.neg_weight, self.dice_weight = pos_weight, neg_weight, dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        weight = target * self.pos_weight + (1 - target) * self.neg_weight
        loss = (bce * weight).mean(dim=(1, 2, 3))
        if self.dice_weight > 0:
            probs = torch.sigmoid(logits)
            inter = (probs * target).sum(dim=(1, 2, 3))
            dice = 1 - (2 * inter + 1) / (probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1)
            loss = loss + self.dice_weight * dice
        return loss


class MultiLabelLoss(nn.Module):
    """Зважена BCE по мітках; невідомі мітки (−1) не враховуються."""

    def __init__(self, labels: list[str], pos_weight: float, neg_weight: float,
                 label_pos_weight: dict[str, float]) -> None:
        super().__init__()
        weights = [label_pos_weight.get(name, pos_weight) for name in labels]
        self.register_buffer("pos", torch.tensor(weights, dtype=torch.float32))
        self.neg_weight = neg_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        known = (target >= 0).float()
        t = target.clamp(min=0)
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        weight = t * self.pos.to(logits.device) + (1 - t) * self.neg_weight
        return (bce * weight * known).sum(dim=1) / known.sum(dim=1).clamp(min=1)


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Учень імітує «м'які» відповіді вчителя (сигмоїди з температурою), ×T² для масштабу градієнтів."""
    t = temperature
    soft = torch.sigmoid(teacher_logits / t)
    loss = F.binary_cross_entropy_with_logits(student_logits / t, soft, reduction="none") * (t * t)
    return loss.flatten(1).mean(dim=1)


def build_loss(cfg: Config) -> nn.Module:
    if cfg.model.type == "segmenter":
        return SegmentationLoss(cfg.loss.pos_weight, cfg.loss.neg_weight, cfg.loss.dice_weight)
    return MultiLabelLoss(cfg.model.labels, cfg.loss.pos_weight, cfg.loss.neg_weight, cfg.loss.label_pos_weight)
