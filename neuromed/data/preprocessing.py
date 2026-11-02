"""Попередня обробка знімків очного дна (стаття, розд. 3.2; вихідні misc_retina/retinaldata).

Порядок: обрізання до кола очного дна → доповнення до квадрата → вирівнювання
гістограми насиченості (HSV) → CLAHE у просторі LAB → (за бажанням) відтінки сірого →
зміна розміру до входу мережі.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from neuromed.config import PreprocessConfig


def read_rgb(path: str | Path) -> np.ndarray:
    """Читання зображення як RGB uint8 (працює з кириличними шляхами у Windows)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Не вдалося прочитати зображення: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_gray(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Не вдалося прочитати зображення: {path}")
    return img


def write_image(path: str | Path, img: np.ndarray) -> None:
    """Запис PNG/JPG; RGB-зображення конвертується в BGR для OpenCV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise IOError(f"Не вдалося записати {path}")
    buf.tofile(str(path))


def fundus_bbox(img: np.ndarray, threshold: int = 10, min_fraction: float = 0.002) -> tuple[int, int, int, int]:
    """Рамка (x0, y0, x1, y1) не-чорної області (кола очного дна)."""
    gray = img.max(axis=2) if img.ndim == 3 else img
    fg = gray > threshold
    rows = np.where(fg.mean(axis=1) > min_fraction)[0]
    cols = np.where(fg.mean(axis=0) > min_fraction)[0]
    h, w = gray.shape
    if rows.size == 0 or cols.size == 0:
        return 0, 0, w, h
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


@dataclass(frozen=True)
class SquareInfo:
    """Як квадратний кадр співвідноситься з вихідним зображенням."""

    x0: int  # початок обрізання у вихідному зображенні
    y0: int
    pad_x: int  # доповнення всередині квадрата
    pad_y: int
    side: int

    def to_source(self, x: float, y: float) -> tuple[float, float]:
        return x - self.pad_x + self.x0, y - self.pad_y + self.y0


def _pad_square(img: np.ndarray, side: int, pad_x: int, pad_y: int) -> np.ndarray:
    out = np.zeros((side, side) + img.shape[2:], dtype=img.dtype)
    out[pad_y:pad_y + img.shape[0], pad_x:pad_x + img.shape[1]] = img
    return out


def fundus_square(img: np.ndarray, mask: np.ndarray | None = None,
                  threshold: int = 10) -> tuple[np.ndarray, np.ndarray | None, SquareInfo]:
    """Обрізає до кола очного дна і доповнює чорним до квадрата (маска — синхронно)."""
    x0, y0, x1, y1 = fundus_bbox(img, threshold)
    crop = img[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    side = max(h, w)
    pad_x, pad_y = (side - w) // 2, (side - h) // 2
    square = _pad_square(crop, side, pad_x, pad_y)
    mask_sq = _pad_square(mask[y0:y1, x0:x1], side, pad_x, pad_y) if mask is not None else None
    return square, mask_sq, SquareInfo(x0, y0, pad_x, pad_y, side)


def pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w:
        return img
    side = max(h, w)
    return _pad_square(img, side, (side - w) // 2, (side - h) // 2)


def equalize_hsv_s(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = cv2.equalizeHist(np.ascontiguousarray(hsv[..., 1]))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def clahe_lab(img: np.ndarray, clip: float = 2.0, grid: int = 16) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    lab[..., 0] = clahe.apply(np.ascontiguousarray(lab[..., 0]))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_image(img: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """RGB uint8 → оброблене зображення HxWxC (C = 1 для сірого, 3 для кольорового)."""
    out = img
    if cfg.hsv_s_equalize:
        out = equalize_hsv_s(out)
    if cfg.clahe:
        out = clahe_lab(out, cfg.clahe_clip, cfg.clahe_grid)
    if cfg.grayscale:
        return cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)[..., None]
    return out


def resize_square(img: np.ndarray, size: int) -> np.ndarray:
    interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_LINEAR
    out = cv2.resize(img, (size, size), interpolation=interp)
    return out[..., None] if out.ndim == 2 else out


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Бінарна маска (0/255 або 0/1) → 0/1 uint8 потрібного розміру."""
    m = (mask > 0).astype(np.uint8) * 255
    out = cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA)
    return (out > 127).astype(np.uint8)


def model_input(img_rgb: np.ndarray, cfg: PreprocessConfig, size: int, prepared_size: int | None = None) -> np.ndarray:
    """Повний шлях підготовки входу: квадрат → (масштаб як при підготовці датасету) → обробка → CxHxW uint8.

    ``prepared_size`` вирівнює роздільність перед CLAHE з тією, на якій готувались
    навчальні дані, щоб результат обробки на інференсі збігався з навчальним.
    """
    img = pad_to_square(img_rgb)
    if prepared_size and img.shape[0] != prepared_size:
        img = resize_square(img, prepared_size)
    proc = preprocess_image(img, cfg)
    return np.ascontiguousarray(resize_square(proc, size).transpose(2, 0, 1))


def normalize(batch_u8: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] → float [-1, 1] (як у вихідному коді: x * 2 − 1)."""
    return batch_u8.float().div_(127.5).sub_(1.0)
