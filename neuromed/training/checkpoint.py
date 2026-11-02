"""Чекпойнти: ваги + повний опис архітектури (ширини після вирізання) + метадані для інференсу.

Завантаження з ``weights_only=True`` — файл не може виконати довільний код.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from neuromed.models import build_model
from neuromed.utils import to_builtin

FORMAT = "neuromed/1"


def save_checkpoint(path: str | Path, model: nn.Module, meta: dict, config: dict | None = None) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT,
        "arch": to_builtin(model.arch()),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "meta": to_builtin(meta),
        "config": to_builtin(config or {}),
    }
    torch.save(payload, path)
    return path.stat().st_size


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError(f"{path}: невідомий формат чекпойнта")
    model = build_model(payload["arch"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload
