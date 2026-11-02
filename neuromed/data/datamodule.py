"""Завантаження, кешування, розбиття та видача батчів.

Датасети тут невеликі (сотні–тисячі знімків), тому всі зображення один раз
проходять попередню обробку, кешуються на диск (``.npz``) і тримаються в пам'яті
(за можливості — прямо на GPU) як uint8. Батч — це просто індексація тензора й
аугментація на пристрої, без DataLoader-воркерів (у Windows вони повільні).
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from neuromed.config import Config
from neuromed.data.augment import BatchAugmenter
from neuromed.data.preprocessing import model_input, normalize, pad_to_square, read_gray, read_rgb, resize_mask
from neuromed.data.records import Record, load_records, make_chunks, prepared_size, split_records
from neuromed.utils import get_logger


class DataModule:
    def __init__(self, cfg: Config, device: torch.device) -> None:
        self.cfg = cfg
        self.task = cfg.model.type
        self.labels = list(cfg.model.labels)
        self.device = device
        self.log = get_logger()
        self.records: list[Record] = load_records(cfg.data.datasets, self.task, self.labels)
        self.channels = 1 if cfg.data.preprocess.grayscale else 3
        self.prepared_size = prepared_size(cfg.data.datasets, self.task)
        self.input_size = cfg.data.input_size

        images, targets = self._load()
        primary = self.labels[0] if self.task == "classifier" else None
        d = cfg.data
        self.train_idx, self.val_idx, self.test_idx = split_records(
            self.records, d.val_fraction, d.test_fraction, d.split_seed, primary)
        self.chunks = make_chunks(self.records, self.train_idx, max(d.chunks, 1), d.split_seed, primary)
        if len(self.train_idx) == 0 or len(self.val_idx) == 0:
            raise ValueError("Замало даних для розбиття на навчальну та контрольну вибірки")

        storage = device if (d.on_device and device.type == "cuda") else torch.device("cpu")
        self.images = torch.from_numpy(images).to(storage)
        self.targets = torch.from_numpy(targets).to(storage)
        self.augmenter = BatchAugmenter(d.augment)

    # ------------------------------------------------------------------ завантаження
    def _cache_path(self) -> Path:
        d = self.cfg.data
        payload = {
            "task": self.task, "size": self.input_size, "labels": self.labels,
            "pre": vars(d.preprocess), "prepared": self.prepared_size,
            "files": [(str(r.image), r.image.stat().st_mtime_ns, str(r.mask) if r.mask else "") for r in self.records],
        }
        key = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
        root = Path(d.cache_dir) if d.cache_dir else Path(d.datasets[0]) / ".cache"
        return root / f"{self.task}_{self.input_size}_{key}.npz"

    def _load_one(self, rec: Record) -> tuple[np.ndarray, np.ndarray]:
        img = model_input(read_rgb(rec.image), self.cfg.data.preprocess, self.input_size, self.prepared_size)
        if self.task == "segmenter":
            target = resize_mask(pad_to_square(read_gray(rec.mask)), self.input_size)[None]
        else:
            target = np.array([rec.labels[name] for name in self.labels], dtype=np.float32)
        return img, target

    def _load(self) -> tuple[np.ndarray, np.ndarray]:
        cache = self._cache_path()
        if cache.is_file():
            with np.load(cache) as data:
                self.log.info("Дані з кешу: %s (%d зразків)", cache, len(data["images"]))
                return data["images"], data["targets"]
        self.log.info("Попередня обробка %d зображень (результат кешується)...", len(self.records))
        workers = min(8, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            items = list(pool.map(self._load_one, self.records))
        images = np.stack([i for i, _ in items])
        targets = np.stack([t for _, t in items])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, images=images, targets=targets)
        return images, targets

    # ------------------------------------------------------------------ вибірки
    def epoch_indices(self, epoch: int) -> tuple[np.ndarray, np.ndarray]:
        """(навчальні індекси, ротаційний контрольний фолд) для епохи.

        * kfold     — навчання на k−1 частинах, втрати на частині, що лишилась (стаття, розд. 4);
        * alternate — по черзі на одній непересічній частині (інструкція: дані забуваються, досвід лишається);
        * full      — уся навчальна вибірка щоепохи.
        """
        strategy, k = self.cfg.data.epoch_strategy, len(self.chunks)
        if strategy == "full" or k < 2:
            return self.train_idx, np.array([], dtype=np.int64)
        fold = epoch % k
        if strategy == "kfold":
            train = np.concatenate([c for i, c in enumerate(self.chunks) if i != fold])
            return train, self.chunks[fold]
        return self.chunks[fold], self.chunks[(fold + 1) % k]

    def split_indices(self, split: str) -> np.ndarray:
        return {"train": self.train_idx, "val": self.val_idx, "test": self.test_idx,
                "all": np.arange(len(self.records))}[split]

    def batch(self, idx: np.ndarray, augment: bool) -> tuple[torch.Tensor, torch.Tensor]:
        index = torch.as_tensor(idx, dtype=torch.long, device=self.images.device)
        x = self.images.index_select(0, index).to(self.device, non_blocking=True).float().div_(255.0)
        y = self.targets.index_select(0, index).to(self.device, non_blocking=True).float()
        if augment:
            x, mask = self.augmenter(x, y if self.task == "segmenter" else None)
            if mask is not None:
                y = mask
        return x.mul_(2.0).sub_(1.0), y

    def raw_batch(self, idx: np.ndarray) -> torch.Tensor:
        """Неаугментований нормалізований вхід (для оцінювання/інференсу)."""
        index = torch.as_tensor(idx, dtype=torch.long, device=self.images.device)
        return normalize(self.images.index_select(0, index).to(self.device))

    def ids(self, idx: np.ndarray) -> list[str]:
        return [self.records[int(i)].id for i in idx]

    def summary(self) -> str:
        sources: dict[str, int] = {}
        for r in self.records:
            sources[r.source] = sources.get(r.source, 0) + 1
        parts = ", ".join(f"{k}: {v}" for k, v in sorted(sources.items()))
        text = (f"{len(self.records)} зразків ({parts}); train/val/test = "
                f"{len(self.train_idx)}/{len(self.val_idx)}/{len(self.test_idx)}; "
                f"частин: {len(self.chunks)}; вхід {self.channels}×{self.input_size}²")
        if self.task == "classifier":
            for j, name in enumerate(self.labels):
                col = self.targets[:, j].cpu()
                text += f"; {name}: +{int((col == 1).sum())}/−{int((col == 0).sum())}/?{int((col < 0).sum())}"
        return text

    def splits(self) -> dict:
        return {"train": self.ids(self.train_idx), "val": self.ids(self.val_idx), "test": self.ids(self.test_idx),
                "chunks": [self.ids(c) for c in self.chunks]}
