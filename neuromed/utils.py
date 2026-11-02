"""Дрібні допоміжні функції: логування, зерна генераторів, пристрій, розміри, час."""

from __future__ import annotations

import io
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

LOGGER_NAME = "neuromed"


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Консоль + (необов'язково) файл. Консоль переводимо в UTF-8, щоб кирилиця не ламала вивід у Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname).1s %(message)s", "%H:%M:%S")
    if not any(type(h) is logging.StreamHandler for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(log_file.resolve())
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == resolved for h in logger.handlers):
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна; вкажіть device: cpu")
    return torch.device(name)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def state_dict_bytes(model: nn.Module) -> int:
    """Розмір збережених ваг (як файл на диску)."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}d {hours}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def to_builtin(value: Any) -> Any:
    """numpy/torch-скаляри → звичайні типи Python (для JSON і `torch.load(weights_only=True)`)."""
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_builtin(obj), fh, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class Stopwatch:
    """Секундомір із накопиченням (для «загального часу навчання»)."""

    def __init__(self) -> None:
        self.total = 0.0
        self._start: float | None = None

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is None:
            return 0.0
        elapsed = time.perf_counter() - self._start
        self.total += elapsed
        self._start = None
        return elapsed
