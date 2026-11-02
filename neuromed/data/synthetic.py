"""Синтетичні «знімки очного дна» для тестів і швидкої перевірки конвеєра.

Не замінюють реальних даних: коло очного дна, яскравий диск з екскавацією,
судини від диска, темна макула. Глаукома моделюється великим відношенням
екскавації до диска (cup/disc ≥ 0.6).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from neuromed.data.prepare import DatasetWriter


def synthetic_fundus(size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    """(RGB-зображення, маска диска 0/1, відношення cup/disc)."""
    img = np.zeros((size, size, 3), np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = size / 2 + rng.normal(0, size * 0.01, 2)
    radius = size * rng.uniform(0.44, 0.48)
    dist = np.hypot(xx - c[0], yy - c[1])
    inside = dist < radius
    base = np.array([rng.uniform(150, 210), rng.uniform(60, 100), rng.uniform(25, 50)], np.float32)
    vignette = np.clip(1.05 - 0.45 * (dist / radius) ** 2, 0, 1)
    img[inside] = base * vignette[inside, None]

    macula = c + np.array([rng.normal(0, size * 0.03), rng.normal(0, size * 0.03)])
    img *= 1 - 0.35 * np.exp(-((xx - macula[0]) ** 2 + (yy - macula[1]) ** 2) / (2 * (size * 0.06) ** 2))[..., None]

    side = rng.choice([-1, 1])
    disc = c + np.array([side * size * rng.uniform(0.17, 0.27), rng.normal(0, size * 0.04)])
    r_disc = size * rng.uniform(0.055, 0.085)
    aspect = rng.uniform(0.85, 1.1)
    d_norm = np.hypot((xx - disc[0]) / r_disc, (yy - disc[1]) / (r_disc * aspect))
    disc_color = np.array([245, 215, 150], np.float32)
    weight = np.clip(1.3 - d_norm, 0, 1)[..., None] ** 0.7
    img = img * (1 - weight) + disc_color * weight
    cup_ratio = float(rng.uniform(0.25, 0.8))
    cup = np.clip(1.2 - d_norm / cup_ratio, 0, 1)[..., None]
    img = img * (1 - 0.8 * cup) + np.array([255, 245, 225], np.float32) * 0.8 * cup

    canvas = np.ascontiguousarray(img.clip(0, 255).astype(np.uint8))
    for _ in range(int(rng.integers(6, 11))):
        angle = rng.uniform(0, 2 * np.pi)
        point = disc.copy()
        pts = [point.copy()]
        for _ in range(int(rng.integers(8, 16))):
            angle += rng.normal(0, 0.25)
            point = point + size * 0.035 * np.array([np.cos(angle), np.sin(angle)])
            pts.append(point.copy())
        thickness = int(rng.integers(2, 5)) * max(1, size // 288)
        color = tuple(int(v) for v in (rng.uniform(100, 140), rng.uniform(20, 40), rng.uniform(15, 30)))
        cv2.polylines(canvas, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)
    canvas[~inside] = 0
    noise = rng.normal(0, 4, canvas.shape)
    canvas = cv2.GaussianBlur(np.clip(canvas + noise, 0, 255).astype(np.uint8), (3, 3), 0)
    mask = (d_norm <= 1.0).astype(np.uint8)
    return canvas, mask, cup_ratio


def generate_synthetic(out: str | Path, count: int = 120, size: int = 576, crop_size: int = 256,
                       expand: float = 0.3, seed: int = 0) -> dict:
    writer = DatasetWriter(out, size, crop_size, expand)
    for i in range(count):
        rng = np.random.default_rng(seed + i)
        img, mask, cup_ratio = synthetic_fundus(size, rng)
        glaucoma = 1.0 if cup_ratio >= 0.6 else 0.0
        sample_id = f"synthetic_{i:04d}"
        writer.add_fundus(sample_id, img, mask, "synthetic", sample_id, glaucoma, 1, rng)
    return writer.close({"name": "synthetic", "license": "згенеровано", "url": "", "citation": ""})
