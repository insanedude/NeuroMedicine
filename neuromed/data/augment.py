"""Пакетна аугментація прямо на пристрої (GPU/CPU).

«Модифікації вхідних даних – поворот, скошення, чуть-чуть розмиття, артефакти»:
уся геометрія (поворот, скошення, масштаб, зсув, віддзеркалення) зводиться до
однієї афінної матриці на зразок і застосовується одним ``grid_sample`` до
зображення й маски синхронно; фотометрія — лише до зображення.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from neuromed.config import AugmentConfig


def _uniform(n: int, lo: float, hi: float, device: torch.device) -> torch.Tensor:
    return torch.empty(n, device=device).uniform_(lo, hi)


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(math.ceil(2.5 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-coords.pow(2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    c = x.shape[1]
    kx = kernel.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = kernel.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), kx, groups=c)
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), ky, groups=c)


class BatchAugmenter:
    def __init__(self, cfg: AugmentConfig) -> None:
        self.cfg = cfg

    def geometry(self, x: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        cfg, n, dev = self.cfg, x.shape[0], x.device
        angle = _uniform(n, -cfg.rotate_deg, cfg.rotate_deg, dev) * math.pi / 180
        shear = _uniform(n, -cfg.shear_deg, cfg.shear_deg, dev) * math.pi / 180
        scale = _uniform(n, cfg.scale_min, cfg.scale_max, dev)
        flip_x = torch.where(torch.rand(n, device=dev) < cfg.hflip, -1.0, 1.0)
        flip_y = torch.where(torch.rand(n, device=dev) < cfg.vflip, -1.0, 1.0)
        shift = torch.stack([_uniform(n, -cfg.translate, cfg.translate, dev),
                             _uniform(n, -cfg.translate, cfg.translate, dev)], dim=1) * 2

        cos, sin, tan = angle.cos(), angle.sin(), shear.tan()
        # A = R(кут) · Sh(скошення) · diag(fx, fy) / масштаб (матриця відображає вихід → вхід)
        a00, a01 = cos, cos * tan - sin
        a10, a11 = sin, sin * tan + cos
        theta = torch.stack([
            torch.stack([a00 * flip_x / scale, a01 * flip_y / scale, shift[:, 0]], dim=1),
            torch.stack([a10 * flip_x / scale, a11 * flip_y / scale, shift[:, 1]], dim=1),
        ], dim=1)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        if mask is not None:
            mask = F.grid_sample(mask, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            mask = (mask > 0.5).to(x.dtype)
        return x, mask

    def photometric(self, x: torch.Tensor) -> torch.Tensor:
        cfg, n, dev = self.cfg, x.shape[0], x.device
        view = (n, 1, 1, 1)
        if cfg.brightness > 0:
            x = x * (1 + _uniform(n, -cfg.brightness, cfg.brightness, dev)).view(view)
        if cfg.contrast > 0:
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            x = (x - mean) * (1 + _uniform(n, -cfg.contrast, cfg.contrast, dev)).view(view) + mean
        if cfg.saturation > 0 and x.shape[1] == 3:
            gray = x.mean(dim=1, keepdim=True)
            x = gray + (x - gray) * (1 + _uniform(n, -cfg.saturation, cfg.saturation, dev)).view(view)
        if cfg.gamma > 0:
            x = x.clamp(0, 1).pow(torch.exp(_uniform(n, -cfg.gamma, cfg.gamma, dev)).view(view))
        if cfg.blur_prob > 0:
            selected = torch.rand(n, device=dev) < cfg.blur_prob
            if selected.any():
                sigma = float(torch.empty(1).uniform_(0.3, max(0.31, cfg.blur_sigma_max)))
                x = torch.where(selected.view(view), _gaussian_blur(x, sigma), x)
        if cfg.artifact_prob > 0 and cfg.artifact_max > 0:
            x = x + self._artifacts(x)
        if cfg.noise_std > 0:
            x = x + torch.randn_like(x) * _uniform(n, 0, cfg.noise_std, dev).view(view)
        return x

    def _artifacts(self, x: torch.Tensor) -> torch.Tensor:
        """Випадкові плями (пил на оптиці) та відблиски — гаусові «краплі» різного знаку."""
        n, _, h, w = x.shape
        k, dev = self.cfg.artifact_max, x.device
        active = (torch.rand(n, k, device=dev) < 0.6) & (torch.rand(n, 1, device=dev) < self.cfg.artifact_prob)
        if not active.any():
            return torch.zeros_like(x[:, :1])
        cx, cy = torch.rand(n, k, 1, 1, device=dev), torch.rand(n, k, 1, 1, device=dev)
        sigma = torch.empty(n, k, 1, 1, device=dev).uniform_(0.01, 0.06)
        amp = torch.empty(n, k, 1, 1, device=dev).uniform_(-0.4, 0.4) * active.view(n, k, 1, 1)
        yy = torch.linspace(0, 1, h, device=dev).view(1, 1, h, 1)
        xx = torch.linspace(0, 1, w, device=dev).view(1, 1, 1, w)
        blobs = amp * torch.exp(-((xx - cx).pow(2) + (yy - cy).pow(2)) / (2 * sigma.pow(2)))
        return blobs.sum(dim=1, keepdim=True)

    def __call__(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """x — float у [0, 1], форма [N, C, H, W]; mask — [N, 1, H, W] з 0/1 або None."""
        if not self.cfg.enabled:
            return x, mask
        x, mask = self.geometry(x, mask)
        x = self.photometric(x).clamp_(0, 1)
        return x, mask
