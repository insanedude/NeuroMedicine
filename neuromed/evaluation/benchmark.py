"""Розмір і час інференсу на CPU — головна мета стиснення («робота на не-графічному процесорі»)."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn

from neuromed.utils import count_parameters, state_dict_bytes


@torch.inference_mode()
def measure_cpu_inference(model: nn.Module, input_shape: tuple[int, int, int], threads: int = 4, runs: int = 20,
                          warmup: int = 3) -> float:
    """Медіанний час (с) обробки одного зображення на CPU з ``threads`` потоками."""
    cpu_model = copy.deepcopy(model).to("cpu", memory_format=torch.contiguous_format).eval()
    previous = torch.get_num_threads()
    torch.set_num_threads(max(1, threads))
    try:
        x = torch.randn(1, *input_shape)
        for _ in range(warmup):
            cpu_model(x)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            cpu_model(x)
            times.append(time.perf_counter() - start)
    finally:
        torch.set_num_threads(previous)
    return float(np.median(times))


def model_stats(model: nn.Module) -> dict:
    widths = getattr(model, "widths", {})
    return {
        "params": count_parameters(model),
        "size_mb": state_dict_bytes(model) / 1e6,
        "base": int(widths.get("enc1.0", 0)),  # ширина першого шару = «основа» при рівномірному прорідженні
    }
