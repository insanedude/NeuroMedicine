"""Метрики (стаття, розд. 3.5): кількість передбачень, TP/TN/FP/FN, додаткові хибні виявлення,
precision, recall, F1, accuracy; для сегментації — ще Dice/IoU, для класифікатора — ROC AUC.

Для сегментатора головне — *виявлення* диска (компонента маски з IoU ≥ порогу з
еталоном): саме його повнота має бути ~100 % при скринінгу. Піксельні метрики
описують якість самої маски.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        if self.tp + self.fp == 0:
            return 1.0 if self.fn == 0 else 0.0
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def specificity(self) -> float:
        return self.tn / (self.tn + self.fp) if self.tn + self.fp else 1.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total else 1.0

    @property
    def iou(self) -> float:
        denom = self.tp + self.fp + self.fn
        return self.tp / denom if denom else 1.0

    def fbeta(self, beta: float = 1.0) -> float:
        p, r = self.precision, self.recall
        b2 = beta * beta
        return (1 + b2) * p * r / (b2 * p + r) if (p + r) > 0 else 0.0


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC через ранги (статистика Манна–Вітні) з урахуванням однакових значень."""
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def threshold_for_recall(scores: np.ndarray, labels: np.ndarray, target: float) -> float | None:
    """Найбільший поріг, за якого повнота ≥ target (для скринінгу: ловимо всіх хворих)."""
    positives = np.sort(scores[labels == 1])[::-1]
    if positives.size == 0:
        return None
    k = max(1, math.ceil(target * positives.size))
    return float(np.clip(positives[k - 1], 1e-6, 1 - 1e-6))


@dataclass
class SampleResult:
    loss: float
    correct: bool | None  # None — не оцінюється (невідома мітка)


@dataclass
class SegmentationEvaluator:
    threshold: float = 0.5
    min_component_frac: float = 0.0005
    detection_iou: float = 0.2
    pixels: Counts = field(default_factory=Counts)
    detection: Counts = field(default_factory=Counts)
    n_predictions: int = 0
    dice_sum: float = 0.0
    loss_sum: float = 0.0
    images: int = 0
    samples: dict[str, SampleResult] = field(default_factory=dict)

    def _components(self, binary: np.ndarray, min_area: int) -> np.ndarray:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        lut = np.zeros(n, dtype=np.int32)
        keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
        lut[keep] = np.arange(1, len(keep) + 1)
        return lut[labels]

    def update(self, probs: torch.Tensor, target: torch.Tensor, ids: list[str], losses: torch.Tensor) -> None:
        pred = probs >= self.threshold
        truth = target > 0.5
        self.pixels.tp += int((pred & truth).sum())
        self.pixels.fp += int((pred & ~truth).sum())
        self.pixels.fn += int((~pred & truth).sum())
        self.pixels.tn += int((~pred & ~truth).sum())
        inter = (pred & truth).flatten(1).sum(1).float()
        sizes = pred.flatten(1).sum(1).float() + truth.flatten(1).sum(1).float()
        dice = torch.where(sizes > 0, 2 * inter / sizes.clamp(min=1), torch.ones_like(sizes))
        self.dice_sum += float(dice.sum())
        self.loss_sum += float(losses.sum())
        self.images += len(ids)

        pred_np = pred[:, 0].to(torch.uint8).cpu().numpy()
        truth_np = truth[:, 0].to(torch.uint8).cpu().numpy()
        min_area = max(1, int(self.min_component_frac * pred_np.shape[1] * pred_np.shape[2]))
        for k, sample_id in enumerate(ids):
            gl = self._components(truth_np[k], min_area)
            pl = self._components(pred_np[k], min_area)
            n_gt, n_pred = int(gl.max()), int(pl.max())
            self.n_predictions += n_pred
            if n_gt == 0:
                correct = n_pred == 0
                self.detection.tn += int(correct)
                self.detection.fp += n_pred
            else:
                joint = np.bincount((gl * (n_pred + 1) + pl).ravel(), minlength=(n_gt + 1) * (n_pred + 1))
                joint = joint.reshape(n_gt + 1, n_pred + 1)
                inter_m = joint[1:, 1:].astype(np.float64)
                area_g, area_p = joint[1:, :].sum(1), joint[:, 1:].sum(0)
                iou = inter_m / np.maximum(area_g[:, None] + area_p[None, :] - inter_m, 1)
                gt_hit = iou.max(1) >= self.detection_iou if n_pred else np.zeros(n_gt, bool)
                pred_hit = iou.max(0) >= self.detection_iou if n_pred else np.zeros(0, bool)
                self.detection.tp += int(gt_hit.sum())
                self.detection.fn += int(n_gt - gt_hit.sum())
                self.detection.fp += int(n_pred - pred_hit.sum())
                correct = bool(gt_hit.all())
            self.samples[sample_id] = SampleResult(float(losses[k]), correct)

    def compute(self) -> dict[str, float]:
        d, p = self.detection, self.pixels
        matched = self.n_predictions - d.fp
        det_precision = matched / self.n_predictions if self.n_predictions else (1.0 if d.fn == 0 else 0.0)
        det_recall = d.recall
        det_f1 = 2 * det_precision * det_recall / (det_precision + det_recall) if det_precision + det_recall else 0.0
        return {
            "loss": self.loss_sum / max(self.images, 1),
            "images": self.images,
            "n_predictions": self.n_predictions,
            "det_tp": d.tp, "det_tn": d.tn, "det_fp": d.fp, "det_fn": d.fn,
            "extra_detections": d.fp,
            "det_recall": det_recall, "det_precision": det_precision, "det_f1": det_f1,
            "pix_precision": p.precision, "pix_recall": p.recall, "pix_f1": p.fbeta(1.0),
            "pix_accuracy": p.accuracy, "iou": p.iou,
            "dice": self.dice_sum / max(self.images, 1),
        }


@dataclass
class ClassificationEvaluator:
    labels: list[str]
    threshold: float = 0.5
    scores: list[np.ndarray] = field(default_factory=list)
    targets: list[np.ndarray] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    samples: dict[str, SampleResult] = field(default_factory=dict)

    def update(self, probs: torch.Tensor, target: torch.Tensor, ids: list[str], losses: torch.Tensor) -> None:
        self.scores.append(probs.float().cpu().numpy())
        self.targets.append(target.float().cpu().numpy())
        self.ids += list(ids)
        self.losses += losses.float().cpu().tolist()

    def compute(self, thresholds: dict[str, float] | None = None,
                calibrate: dict[str, float] | None = None) -> dict[str, float]:
        """``thresholds`` — готові пороги (напр. з контрольної вибірки); ``calibrate`` — підібрати пороги
        під цільову повноту на цих же даних."""
        scores, targets = np.concatenate(self.scores), np.concatenate(self.targets)
        metrics: dict[str, float] = {"loss": float(np.mean(self.losses)) if self.losses else 0.0,
                                     "images": len(self.ids)}
        aucs = []
        for j, name in enumerate(self.labels):
            known = targets[:, j] >= 0
            s, y = scores[known, j], targets[known, j].astype(int)
            thr = (thresholds or {}).get(name)
            if thr is None and calibrate and name in calibrate:
                thr = threshold_for_recall(s, y, calibrate[name])
            thr = self.threshold if thr is None else thr
            self.thresholds[name] = float(thr)
            pred = s >= thr
            c = Counts(int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()),
                       int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum()))
            auc = roc_auc(s, y)
            if not math.isnan(auc):
                aucs.append(auc)
            metrics.update({
                f"{name}_tp": c.tp, f"{name}_fp": c.fp, f"{name}_fn": c.fn, f"{name}_tn": c.tn,
                f"{name}_precision": c.precision, f"{name}_recall": c.recall, f"{name}_f1": c.fbeta(1.0),
                f"{name}_f2": c.fbeta(2.0), f"{name}_accuracy": c.accuracy, f"{name}_specificity": c.specificity,
                f"{name}_auc": auc if not math.isnan(auc) else 0.0, f"{name}_threshold": float(thr),
            })
        metrics["mean_auc"] = float(np.mean(aucs)) if aucs else 0.0
        primary = self.labels[0]
        thr0 = self.thresholds[primary]
        for k, sample_id in enumerate(self.ids):
            y = targets[k, 0]
            correct = None if y < 0 else bool((scores[k, 0] >= thr0) == (y == 1))
            self.samples[sample_id] = SampleResult(self.losses[k], correct)
        metrics["n_predictions"] = int(sum((scores[:, 0] >= thr0) & (targets[:, 0] >= 0)))
        return metrics
