"""Розклади фаз, монітор, функції втрат."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from neuromed.compression.schedule import constant, spread_schedule, squeeze_schedule
from neuromed.config import CompressConfig
from neuromed.training.losses import MultiLabelLoss, SegmentationLoss, distillation_loss
from neuromed.training.monitor import PhaseMonitor, acceptance_goal


def test_schedules_follow_instructions():
    c = CompressConfig()
    spread, squeeze = spread_schedule(c), squeeze_schedule(c)
    assert spread(0.0) == (pytest.approx(0.15), 0.0)
    assert spread(1.0) == (pytest.approx(0.5), 0.0)
    assert squeeze(0.0) == (pytest.approx(0.5), 0.0)
    p, alpha = squeeze(0.5)
    assert p == pytest.approx(0.25) and alpha == pytest.approx(0.475)
    assert squeeze(1.0) == (0.0, pytest.approx(0.95))
    assert squeeze(2.0) == squeeze(1.0)
    assert constant(0.1)(0.7) == (0.1, 0.0)


def test_acceptance_goal_uses_minimum_and_max_drop():
    goal = acceptance_goal({"det_recall": 0.95}, {"dice": 0.04}, {"det_recall": 1.0, "dice": 0.90})
    assert goal({"det_recall": 0.97, "dice": 0.87})
    assert not goal({"det_recall": 0.94, "dice": 0.90})
    assert not goal({"det_recall": 1.0, "dice": 0.85})


def test_monitor_stops_after_patience_and_restores_best():
    model = nn.Linear(1, 1)
    monitor = PhaseMonitor(["score"], goal=lambda m: m["score"] >= 0.5, patience=2, min_epochs=1)
    stops = []
    for epoch, score in enumerate([0.2, 0.6, 0.55, 0.58, 0.59]):
        with torch.no_grad():
            model.weight.fill_(score)
        stops.append(monitor.update(epoch, {"score": score}, model))
    assert stops == [False, False, False, True, True]
    assert monitor.goal_reached and monitor.first_goal_epoch == 1 and monitor.best_epoch == 1
    monitor.restore(model)
    assert float(model.weight) == pytest.approx(0.6)


def test_monitor_gives_up_when_hopeless():
    model = nn.Linear(1, 1)
    monitor = PhaseMonitor(["score"], goal=lambda m: False, patience=2, min_epochs=1)
    stops = [monitor.update(e, {"score": 0.1}, model) for e in range(5)]
    assert stops.index(True) == 4  # без покращень 2 × patience епох
    assert not monitor.goal_reached


def test_segmentation_loss_is_per_sample():
    loss = SegmentationLoss(3.0, 1.0, 1.0)(torch.randn(4, 1, 16, 16), (torch.rand(4, 1, 16, 16) > 0.5).float())
    assert loss.shape == (4,) and (loss > 0).all()


def test_multilabel_loss_ignores_unknown_labels():
    fn = MultiLabelLoss(["glaucoma", "valid_image"], 1.0, 1.0, {"glaucoma": 2.0})
    target = torch.tensor([[-1.0, 1.0]])
    a = fn(torch.tensor([[5.0, 2.0]]), target)
    b = fn(torch.tensor([[-5.0, 2.0]]), target)
    torch.testing.assert_close(a, b)


def test_distillation_gradient_vanishes_when_student_matches_teacher():
    teacher = torch.randn(3, 2)
    student = teacher.clone().requires_grad_(True)
    distillation_loss(student, teacher, 2.0).sum().backward()
    assert student.grad.abs().max() < 1e-6
