"""Тренер: фази навчання з розкладами dropout/нормування, дистиляцією, буфером складних зразків."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from neuromed.compression.regularizer import ChannelNormRegularizer
from neuromed.compression.schedule import PhaseSchedule, constant
from neuromed.config import Config
from neuromed.data.datamodule import DataModule
from neuromed.data.sampling import HardExampleBuffer
from neuromed.models.blocks import set_dropout, transfer_convs
from neuromed.training.losses import build_loss, distillation_loss
from neuromed.training.metrics import ClassificationEvaluator, SampleResult, SegmentationEvaluator
from neuromed.training.monitor import PhaseMonitor
from neuromed.utils import get_logger


@dataclass
class EvalResult:
    metrics: dict[str, float]
    thresholds: dict[str, float] = field(default_factory=dict)
    samples: dict[str, SampleResult] = field(default_factory=dict)


@dataclass
class PhaseResult:
    epochs: int
    seconds: float
    best_metrics: dict | None
    best_epoch: int
    first_goal_epoch: int | None
    goal_reached: bool
    last_metrics: dict | None


class HistoryLog:
    """history.csv: по рядку на епоху всіх фаз (продовжується при відновленні)."""

    BASE = ["global_epoch", "phase", "epoch", "lr", "dropout", "norm_share", "norm_ratio", "train_loss",
            "fold_loss", "seconds"]

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fields: list[str] | None = None
        if path.is_file():
            with open(path, encoding="utf-8", newline="") as fh:
                self.fields = next(csv.reader(fh), None)

    def write(self, row: dict) -> None:
        if self.fields is None:
            self.fields = self.BASE + [k for k in row if k not in self.BASE]
            with open(self.path, "w", encoding="utf-8", newline="") as fh:
                csv.writer(fh).writerow(self.fields)
        with open(self.path, "a", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self.fields, extrasaction="ignore", restval="").writerow(
                {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})


class Trainer:
    def __init__(self, cfg: Config, data: DataModule, device: torch.device, run_dir: Path) -> None:
        self.cfg, self.data, self.device = cfg, data, device
        self.task = cfg.model.type
        self.loss_fn = build_loss(cfg).to(device)
        self.amp = cfg.train.amp and device.type == "cuda"
        self.memory_format = (torch.channels_last if cfg.train.channels_last and device.type == "cuda"
                              else torch.contiguous_format)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.history = HistoryLog(run_dir / "history.csv")
        self.global_epoch = 0
        self.rng = np.random.default_rng(cfg.seed)
        hard = cfg.data.hard_examples
        fold_size = max(1, len(data.train_idx) // max(1, len(data.chunks)))
        self.hard = HardExampleBuffer(hard.max_size or fold_size) if hard.enabled else None
        self.edge_ids: list[str] = []
        self.log = get_logger()

    # ------------------------------------------------------------------ оптимізація
    def prepare(self, model: nn.Module) -> nn.Module:
        """Переносить модель на пристрій у робочому форматі пам'яті (після створення, завантаження, вирізання)."""
        return model.to(self.device, memory_format=self.memory_format)

    def make_optimizer(self, model: nn.Module, lr: float) -> torch.optim.Optimizer:
        t = self.cfg.train
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(t.beta1, t.beta2), weight_decay=t.weight_decay)

    def _logits(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
            return model(x.contiguous(memory_format=self.memory_format)).float()

    def _step(self, model: nn.Module, optimizer: torch.optim.Optimizer, idx: np.ndarray,
              regularizer: ChannelNormRegularizer | None, alpha: float, teacher: nn.Module | None) -> tuple:
        x, y = self.data.batch(idx, augment=True)
        logits = self._logits(model, x)
        per_sample = self.loss_fn(logits, y)
        if teacher is not None:
            with torch.no_grad():
                teacher_logits = self._logits(teacher, x)
            a = self.cfg.distill.alpha
            per_sample = (1 - a) * per_sample + a * distillation_loss(logits, teacher_logits,
                                                                      self.cfg.distill.temperature)
        loss = per_sample.mean()
        c = self.cfg.compress
        if regularizer is not None and alpha > 0 and c.norm_method in ("penalty", "both"):
            # «частка нормування»: (1 − α)·задача + α·норма вирізуваних нейронів
            loss = (1 - alpha) * loss + alpha * c.norm_strength * regularizer.penalty()
        optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if self.cfg.train.grad_clip > 0:
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), self.cfg.train.grad_clip)
        self.scaler.step(optimizer)
        self.scaler.update()
        if regularizer is not None and alpha > 0 and c.norm_method in ("projection", "both"):
            regularizer.project(alpha)
        return per_sample.detach().float().cpu().numpy(), float(loss.detach())

    # ------------------------------------------------------------------ фаза
    def train_phase(self, model: nn.Module, optimizer: torch.optim.Optimizer, *, phase: str, epochs: int,
                    schedule: PhaseSchedule | None = None, regularizer: ChannelNormRegularizer | None = None,
                    teacher: nn.Module | None = None, monitor: PhaseMonitor | None = None,
                    freeze_epochs: int = 0, show: list[str] | None = None) -> PhaseResult:
        schedule = schedule or constant(0.0)
        bs = self.cfg.data.batch_size
        hard_cfg = self.cfg.data.hard_examples
        frozen = [p for conv in transfer_convs(model) for p in conv.parameters()] if freeze_epochs > 0 else []
        if teacher is not None:
            teacher.eval()
        show = show or []
        start = time.perf_counter()
        last_metrics = None
        epochs_run = 0
        for epoch in range(epochs):
            t0 = time.perf_counter()
            for p in frozen:
                p.requires_grad_(epoch >= freeze_epochs)
            model.train()
            train_idx, fold_idx = self.data.epoch_indices(self.global_epoch)
            order = self.rng.permutation(train_idx)
            n_batches = max(1, len(order) // bs)
            losses, dropout, alpha = [], 0.0, 0.0
            for b in range(n_batches):
                dropout, alpha = schedule((epoch * n_batches + b + 1) / (epochs * n_batches))
                set_dropout(model, dropout)
                idx = order[b * bs:(b + 1) * bs]
                per_sample, loss = self._step(model, optimizer, idx, regularizer, alpha, teacher)
                losses.append(loss)
                if self.hard is not None:
                    self.hard.update(idx, per_sample)
                    if len(self.hard) >= bs and self.rng.random() < hard_cfg.replay_prob:
                        hidx = self.hard.sample(bs, self.rng)
                        per_sample, _ = self._step(model, optimizer, hidx, regularizer, alpha, teacher)
                        self.hard.update(hidx, per_sample)
            epochs_run = epoch + 1
            last = epoch == epochs - 1
            row = {"global_epoch": self.global_epoch, "phase": phase, "epoch": epoch + 1,
                   "lr": optimizer.param_groups[0]["lr"], "dropout": dropout, "norm_share": alpha,
                   "norm_ratio": regularizer.mean_ratio() if regularizer is not None else "",
                   "train_loss": float(np.mean(losses)),
                   "fold_loss": self.mean_loss(model, fold_idx) if len(fold_idx) else ""}
            stop = False
            if monitor is not None or last or epoch % max(1, self.cfg.train.eval_every) == 0:
                last_metrics = self.evaluate(model, "val").metrics
                row.update(last_metrics)
                if monitor is not None:
                    stop = monitor.update(epoch, last_metrics, model)
            row["seconds"] = time.perf_counter() - t0
            self.history.write(row)
            self._log_epoch(phase, epoch, epochs, row, show)
            self.global_epoch += 1
            if stop:
                break
        if monitor is not None:
            monitor.restore(model)
        set_dropout(model, schedule(1.0)[0])
        return PhaseResult(
            epochs=epochs_run, seconds=time.perf_counter() - start,
            best_metrics=monitor.best_metrics if monitor else last_metrics,
            best_epoch=monitor.best_epoch if monitor else epochs_run - 1,
            first_goal_epoch=monitor.first_goal_epoch if monitor else None,
            goal_reached=monitor.goal_reached if monitor else True,
            last_metrics=last_metrics)

    def _log_epoch(self, phase: str, epoch: int, epochs: int, row: dict, show: list[str]) -> None:
        parts = [f"{phase} {epoch + 1}/{epochs}", f"loss {row['train_loss']:.4f}"]
        if row["dropout"]:
            parts.append(f"p {row['dropout']:.2f}")
        if row["norm_share"]:
            parts.append(f"α {row['norm_share']:.2f}")
        if row["norm_ratio"] != "":
            parts.append(f"‖w‖ {row['norm_ratio']:.3f}")
        parts += [f"{k} {row[k]:.4f}" for k in show if k in row]
        parts.append(f"{row['seconds']:.1f}s")
        self.log.info(" | ".join(parts))

    # ------------------------------------------------------------------ оцінювання
    def _batches(self, idx: np.ndarray):
        bs = self.cfg.eval.batch_size
        for start in range(0, len(idx), bs):
            yield idx[start:start + bs]

    @torch.no_grad()
    def mean_loss(self, model: nn.Module, idx: np.ndarray) -> float:
        model.eval()
        total = 0.0
        for chunk in self._batches(idx):
            x = self.data.raw_batch(chunk)
            y = self.data.targets.index_select(0, torch.as_tensor(chunk, device=self.data.targets.device))
            total += float(self.loss_fn(self._logits(model, x), y.to(self.device).float()).sum())
        model.train()
        return total / max(1, len(idx))

    @torch.no_grad()
    def evaluate(self, model: nn.Module, split: str = "val", thresholds: dict[str, float] | None = None,
                 indices: np.ndarray | None = None) -> EvalResult:
        """Метрики на вибірці. Для класифікатора без ``thresholds`` пороги калібруються на цій вибірці
        під ``eval.calibrate_recall`` (так робиться на контрольній вибірці; тест використовує її пороги)."""
        was_training = model.training
        model.eval()
        e = self.cfg.eval
        idx = self.data.split_indices(split) if indices is None else indices
        if self.task == "segmenter":
            evaluator = SegmentationEvaluator(e.threshold, e.min_component_frac, e.detection_iou)
        else:
            evaluator = ClassificationEvaluator(self.cfg.model.labels, e.threshold)
        for chunk in self._batches(idx):
            x = self.data.raw_batch(chunk)
            y = self.data.targets.index_select(0, torch.as_tensor(chunk, device=self.data.targets.device))
            y = y.to(self.device).float()
            logits = self._logits(model, x)
            evaluator.update(torch.sigmoid(logits), y, self.data.ids(chunk), self.loss_fn(logits, y))
        if self.task == "segmenter":
            metrics = evaluator.compute()
        else:
            metrics = evaluator.compute(thresholds, None if thresholds else e.calibrate_recall)
        if self.edge_ids:
            known = [evaluator.samples[i].correct for i in self.edge_ids if i in evaluator.samples]
            if known:
                metrics["edge_detected"] = sum(bool(c) for c in known) / len(known)
        model.train(was_training)
        thresholds_used = getattr(evaluator, "thresholds", {})
        return EvalResult(metrics, dict(thresholds_used), evaluator.samples)

    def pick_edge_cases(self, result: EvalResult, k: int) -> list[str]:
        """«Крайні випадки»: найскладніші (за втратою) зразки, які перша ітерація ще розпізнає правильно."""
        candidates = [(s.loss, sid) for sid, s in result.samples.items() if s.correct]
        candidates.sort(reverse=True)
        return [sid for _, sid in candidates[:k]]
