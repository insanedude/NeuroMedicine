"""Буфер найскладніших зразків («мультиплікація важких випадків»).

Стаття, розд. 4: «була сформована вибірка найбільш проблемних зразків (максимальний
розмір — 1 фолд), яка використовувалася на кожному кроці для випадкового відбору
пакетів навчання за допомогою зваженого випадкового вибору, де випадки з вищими
втратами обиралися частіше». Зберігаються лише індекси: при повторі зразок
аугментується заново, тож мережа не завчає конкретні пікселі.
"""

from __future__ import annotations

import numpy as np


class HardExampleBuffer:
    def __init__(self, max_size: int) -> None:
        self.max_size = max(1, int(max_size))
        self.losses: dict[int, float] = {}

    def __len__(self) -> int:
        return len(self.losses)

    def update(self, indices, losses) -> None:
        for i, loss in zip(indices, losses):
            self.losses[int(i)] = float(loss)
        if len(self.losses) > int(self.max_size * 1.5):
            keep = sorted(self.losses.items(), key=lambda kv: kv[1], reverse=True)[:self.max_size]
            self.losses = dict(keep)

    def sample(self, k: int, rng: np.random.Generator) -> np.ndarray:
        keys = np.fromiter(self.losses.keys(), dtype=np.int64)
        weights = np.fromiter(self.losses.values(), dtype=np.float64).clip(min=1e-8)
        return rng.choice(keys, size=k, replace=len(keys) < k, p=weights / weights.sum())
