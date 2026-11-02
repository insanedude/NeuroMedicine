"""Маніфести підготовлених датасетів і розбиття на вибірки.

Підготовлений датасет — тека з ``segmentation.csv`` і/або ``classification.csv``
(шляхи відносні до теки) та ``meta.json``. Розбиття групове (обидва ока одного
пацієнта завжди в одній вибірці) і стратифіковане за джерелом і міткою, тож
кожна частина містить суміш усіх датасетів — як просить інструкція: тренувати
поперемінно на непересічних збірних частинах кількох датасетів.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from neuromed.utils import read_json

SEG_MANIFEST = "segmentation.csv"
CLS_MANIFEST = "classification.csv"
UNKNOWN = -1.0  # мітка невідома → не бере участі у втратах і метриках цієї мітки


@dataclass
class Record:
    id: str
    image: Path
    source: str
    group: str
    mask: Path | None = None
    labels: dict[str, float] = field(default_factory=dict)


def manifest_name(task: str) -> str:
    return SEG_MANIFEST if task == "segmenter" else CLS_MANIFEST


def load_records(dataset_dirs: list[str | Path], task: str, labels: list[str]) -> list[Record]:
    records: list[Record] = []
    for directory in dataset_dirs:
        directory = Path(directory)
        path = directory / manifest_name(task)
        if not path.is_file():
            raise FileNotFoundError(f"Немає {path}. Спершу виконайте `python -m neuromed prepare ...`")
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rec = Record(id=row["id"], image=directory / row["image"], source=row["source"], group=row["group"])
                if task == "segmenter":
                    rec.mask = directory / row["mask"]
                else:
                    rec.labels = {name: float(row[name]) if row.get(name, "") != "" else UNKNOWN for name in labels}
                records.append(rec)
    ids = [r.id for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Ідентифікатори зразків у маніфестах повторюються")
    if not records:
        raise ValueError("Маніфести порожні")
    return records


def prepared_size(dataset_dirs: list[str | Path], task: str) -> int | None:
    """Роздільність, до якої датасет масштабувався при підготовці (з meta.json)."""
    for directory in dataset_dirs:
        meta = Path(directory) / "meta.json"
        if meta.is_file():
            info = read_json(meta)
            key = "prepared_size" if task == "segmenter" else "crop_size"
            if info.get(key):
                return int(info[key])
    return None


def _stratum(rec: Record, primary_label: str | None) -> tuple[str, float]:
    label = rec.labels.get(primary_label, UNKNOWN) if primary_label else UNKNOWN
    return rec.source, label


def _group_strata(records: list[Record], indices: np.ndarray, primary_label: str | None) -> dict:
    """Страта → {група → [індекси]}. Група належить страті свого «найважчого» запису."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i in indices:
        groups[records[i].group].append(int(i))
    strata: dict[tuple, dict[str, list[int]]] = defaultdict(dict)
    for group, members in groups.items():
        key = max(_stratum(records[i], primary_label) for i in members)
        strata[key][group] = members
    return strata


def split_records(records: list[Record], val_fraction: float, test_fraction: float, seed: int,
                  primary_label: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(train, val, test) — індекси записів; групи не розриваються між вибірками."""
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    strata = _group_strata(records, np.arange(len(records)), primary_label)
    for key in sorted(strata, key=str):
        groups = strata[key]
        names = sorted(groups)
        rng.shuffle(names)
        total = sum(len(groups[g]) for g in names)
        n_test, n_val = test_fraction * total, val_fraction * total
        taken_test = taken_val = 0
        for g in names:
            members = groups[g]
            if taken_test < n_test - 0.5 * len(members):
                test += members
                taken_test += len(members)
            elif taken_val < n_val - 0.5 * len(members):
                val += members
                taken_val += len(members)
            else:
                train += members
    return np.array(sorted(train)), np.array(sorted(val)), np.array(sorted(test))


def make_chunks(records: list[Record], indices: np.ndarray, k: int, seed: int,
                primary_label: str | None = None) -> list[np.ndarray]:
    """k непересічних частин, збалансованих за джерелом і міткою (групи цілі)."""
    rng = np.random.default_rng(seed + 1)
    chunks: list[list[int]] = [[] for _ in range(k)]
    strata = _group_strata(records, indices, primary_label)
    for key in sorted(strata, key=str):
        groups = strata[key]
        names = sorted(groups)
        rng.shuffle(names)
        counts = [0] * k
        for g in names:
            target = int(np.argmin(counts))
            chunks[target] += groups[g]
            counts[target] += len(groups[g])
    return [np.array(sorted(c), dtype=np.int64) for c in chunks]
