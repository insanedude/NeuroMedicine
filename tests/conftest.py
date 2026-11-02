"""Спільні фікстури тестів."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from neuromed.data.synthetic import generate_synthetic
from neuromed.utils import LOGGER_NAME


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture(autouse=True)
def _close_log_handlers():
    """Запуски додають обробники логера (консоль pytest, log.txt) — закриваємо, щоб не тримати файли."""
    yield
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    """Маленький синтетичний датасет (сегментація + кропи для класифікатора)."""
    out = tmp_path_factory.mktemp("synthetic")
    generate_synthetic(out, count=24, size=128, crop_size=64, seed=0)
    return out
