"""Метрики: лічильники, ROC AUC, калібрування порогу, виявлення диска."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from neuromed.training.metrics import (
    ClassificationEvaluator,
    Counts,
    SegmentationEvaluator,
    roc_auc,
    threshold_for_recall,
)


def test_counts():
    c = Counts(tp=8, fp=2, fn=2, tn=88)
    assert c.precision == pytest.approx(0.8)
    assert c.recall == pytest.approx(0.8)
    assert c.fbeta(1.0) == pytest.approx(0.8)
    assert c.specificity == pytest.approx(88 / 90)
    assert c.accuracy == pytest.approx(0.96)
    assert c.iou == pytest.approx(8 / 12)


def _pairwise_auc(scores, labels):
    pos, neg = scores[labels == 1], scores[labels == 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return wins / (len(pos) * len(neg))


def test_roc_auc_matches_pairwise_definition_with_ties():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 200)
    scores = np.round(rng.random(200) * 0.5 + labels * 0.3, 1)  # багато однакових значень
    assert roc_auc(scores, labels) == pytest.approx(_pairwise_auc(scores, labels))


def test_roc_auc_edge_cases():
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0
    assert math.isnan(roc_auc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_threshold_for_recall():
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1])
    labels = np.array([1, 1, 1, 1, 0, 0])
    thr = threshold_for_recall(scores, labels, 0.75)
    assert thr == pytest.approx(0.7)
    assert ((scores >= thr) & (labels == 1)).sum() / 4 >= 0.75
    assert threshold_for_recall(scores, np.zeros(6, int), 0.9) is None


def _square(mask, y0, x0, size):
    mask[y0:y0 + size, x0:x0 + size] = 1.0


def test_segmentation_detection_counts():
    target = torch.zeros(2, 1, 64, 64)
    probs = torch.zeros(2, 1, 64, 64)
    _square(target[0, 0], 20, 20, 16)
    _square(probs[0, 0], 22, 22, 16)  # влучання (IoU > 0.2)
    _square(probs[0, 0], 0, 50, 8)  # додаткове хибне виявлення
    _square(target[1, 0], 10, 10, 16)  # диск пропущено
    ev = SegmentationEvaluator(threshold=0.5, min_component_frac=0.001, detection_iou=0.2)
    ev.update(probs, target, ["a", "b"], torch.tensor([0.1, 0.9]))
    m = ev.compute()
    assert (m["det_tp"], m["det_fn"], m["det_fp"]) == (1, 1, 1)
    assert m["det_recall"] == pytest.approx(0.5)
    assert m["extra_detections"] == 1
    assert m["n_predictions"] == 2
    assert m["det_precision"] == pytest.approx(0.5)
    assert ev.samples["a"].correct and not ev.samples["b"].correct
    assert 0 < m["dice"] < 1


def test_segmentation_empty_image_is_true_negative():
    ev = SegmentationEvaluator()
    zeros = torch.zeros(1, 1, 32, 32)
    ev.update(zeros, zeros, ["empty"], torch.zeros(1))
    m = ev.compute()
    assert m["det_tn"] == 1 and m["det_fp"] == 0 and m["dice"] == 1.0


def test_classification_ignores_unknown_and_calibrates():
    ev = ClassificationEvaluator(["glaucoma", "valid_image"])
    probs = torch.tensor([[0.9, 0.9], [0.4, 0.8], [0.3, 0.1], [0.1, 0.9], [0.95, 0.2]])
    target = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    ev.update(probs, target, list("abcde"), torch.zeros(5))
    m = ev.compute(calibrate={"glaucoma": 1.0})
    assert m["glaucoma_recall"] == 1.0
    assert m["glaucoma_tp"] + m["glaucoma_fn"] == 2  # зразок «e» з невідомою міткою не рахується
    assert m["glaucoma_threshold"] == pytest.approx(0.4)
    assert m["glaucoma_auc"] == 1.0
    assert ev.samples["e"].correct is None
    fixed = ClassificationEvaluator(["glaucoma", "valid_image"])
    fixed.update(probs, target, list("abcde"), torch.zeros(5))
    assert fixed.compute(thresholds={"glaucoma": 0.5})["glaucoma_recall"] == 0.5
