"""Монітор фази навчання: «треним, поки не стане достатніх показників» і стабілізація після вирізання."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn


def meets(metrics: dict, minimum: dict[str, float]) -> bool:
    return all(metrics.get(k, float("-inf")) >= v for k, v in minimum.items())


def acceptance_goal(minimum: dict[str, float], max_drop: dict[str, float],
                    reference: dict | None) -> Callable[[dict], bool]:
    """Мережа «не фейлить»: абсолютні мінімуми виконано і падіння відносно першої ітерації в межах допуску."""

    def goal(metrics: dict) -> bool:
        if not meets(metrics, minimum):
            return False
        if reference:
            return all(metrics.get(k, float("-inf")) >= reference[k] - drop for k, drop in max_drop.items()
                       if k in reference)
        return True

    return goal


class PhaseMonitor:
    """Відстежує найкращу епоху (спершу ті, що досягли цілі, далі — за сумою метрик ``selection``)
    і вирішує, коли зупинитись: ціль досягнута й немає покращень ``patience`` епох, або покращень
    немає вдвічі довше — подальше тренування безнадійне («фейлить незалежно від кількості тренування»)."""

    def __init__(self, selection: list[str], goal: Callable[[dict], bool] | None = None, patience: int = 10,
                 tolerance: float = 1e-3, min_epochs: int = 1, hopeless_factor: float = 2.0) -> None:
        self.selection, self.goal = selection, goal
        self.patience, self.tolerance, self.min_epochs = patience, tolerance, min_epochs
        self.hopeless_factor = hopeless_factor
        self.best_key: tuple[bool, float] | None = None
        self.best_state: dict | None = None
        self.best_metrics: dict | None = None
        self.best_epoch = -1
        self.first_goal_epoch: int | None = None
        self.since_best = 0

    def score(self, metrics: dict) -> float:
        return float(sum(metrics[k] for k in self.selection))

    def update(self, epoch: int, metrics: dict, model: nn.Module) -> bool:
        met = self.goal(metrics) if self.goal else True
        if met and self.first_goal_epoch is None:
            self.first_goal_epoch = epoch
        score = self.score(metrics)
        better = (self.best_key is None or (met and not self.best_key[0])
                  or (met == self.best_key[0] and score > self.best_key[1] + self.tolerance))
        if better:
            self.best_key = (met, score)
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.best_metrics = dict(metrics)
            self.best_epoch = epoch
            self.since_best = 0
        else:
            self.since_best += 1
        if epoch + 1 < self.min_epochs:
            return False
        reached = self.best_key is not None and self.best_key[0]
        return (reached and self.since_best >= self.patience) or \
            self.since_best >= self.hopeless_factor * self.patience

    @property
    def goal_reached(self) -> bool:
        return self.best_key is not None and self.best_key[0]

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
