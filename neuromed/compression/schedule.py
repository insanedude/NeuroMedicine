"""Розклади dropout і частки нормування в межах фази (лінійно за прогресом 0 → 1, покроково)."""

from __future__ import annotations

from dataclasses import dataclass


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * min(max(t, 0.0), 1.0)


@dataclass(frozen=True)
class PhaseSchedule:
    dropout_from: float
    dropout_to: float
    norm_from: float = 0.0
    norm_to: float = 0.0

    def __call__(self, progress: float) -> tuple[float, float]:
        """(dropout p, частка нормування α) для прогресу фази ∈ [0, 1]."""
        return lerp(self.dropout_from, self.dropout_to, progress), lerp(self.norm_from, self.norm_to, progress)


def constant(dropout: float) -> PhaseSchedule:
    return PhaseSchedule(dropout, dropout)


def spread_schedule(cfg) -> PhaseSchedule:
    """4.1: dropout 0.1–0.2 → 0.5 — модель розносить досвід на декілька нейронів."""
    return PhaseSchedule(cfg.dropout_start, cfg.dropout_max)


def squeeze_schedule(cfg) -> PhaseSchedule:
    """4.3: одночасно dropout 0.5 → 0 і частка нормування 0 → 0.95."""
    return PhaseSchedule(cfg.dropout_max, 0.0, 0.0, cfg.norm_max)
