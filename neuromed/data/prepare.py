"""Перетворення публічних датасетів в єдиний формат проєкту.

Результат (тека ``data/processed/<назва>``):
    images/<id>.png     — квадратний знімок очного дна (обрізаний до кола), prepared_size²
    masks/<id>.png      — маска диска зорового нерва 0/255
    crops/<id>.png      — кропи диска (та «невалідні» кропи поза диском) для класифікатора
    segmentation.csv    — id,image,mask,source,group
    classification.csv  — id,image,source,group,glaucoma,valid_image  (−1 = мітка невідома)
    meta.json           — опис, ліцензія, розміри

Мітка ``valid_image`` (з гілки 16_01_2026) вчить класифікатор відкидати хибні
виявлення сегментатора: кропи поза диском мають valid_image = 0.
"""

from __future__ import annotations

import csv
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from neuromed.data.preprocessing import fundus_square, pad_to_square, read_gray, read_rgb, resize_square, write_image
from neuromed.data.records import CLS_MANIFEST, SEG_MANIFEST, UNKNOWN
from neuromed.utils import get_logger, write_json

SEG_FIELDS = ["id", "image", "mask", "source", "group"]
CLS_FIELDS = ["id", "image", "source", "group", "glaucoma", "valid_image"]


# ---------------------------------------------------------------------- геометрія кропів
def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def disc_box(bbox: tuple[int, int, int, int], expand: float) -> tuple[int, int, int, int]:
    """Квадратна рамка навколо диска: сторона = max(w, h)·(1 + 2·expand)."""
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + 2 * expand)
    half = side / 2
    return int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))


def crop_with_pad(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Кроп рамки; частини поза зображенням заповнюються чорним."""
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    out = np.zeros((y1 - y0, x1 - x0) + img.shape[2:], dtype=img.dtype)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def negative_box(img: np.ndarray, avoid: tuple[int, int, int, int], side: int,
                 rng: np.random.Generator, tries: int = 60) -> tuple[int, int, int, int] | None:
    """Рамка того ж розміру поза диском, переважно всередині кола очного дна."""
    h, w = img.shape[:2]
    if side >= min(h, w):
        return None
    fundus = (img.max(axis=2) if img.ndim == 3 else img) > 15
    for _ in range(tries):
        x0, y0 = int(rng.integers(0, w - side)), int(rng.integers(0, h - side))
        box = (x0, y0, x0 + side, y0 + side)
        if _overlaps(box, avoid):
            continue
        if fundus[y0:y0 + side, x0:x0 + side].mean() > 0.7:
            return box
    return None


# ---------------------------------------------------------------------- запис маніфестів
class DatasetWriter:
    def __init__(self, out_dir: str | Path, prepared: int, crop_size: int, crop_expand: float) -> None:
        self.out = Path(out_dir)
        self.prepared, self.crop_size, self.crop_expand = prepared, crop_size, crop_expand
        self.seg_rows: list[dict] = []
        self.cls_rows: list[dict] = []
        self._lock = threading.Lock()  # рядки додаються з кількох потоків

    def add_fundus(self, sample_id: str, square: np.ndarray, mask: np.ndarray, source: str, group: str,
                   glaucoma: float, negatives: int, rng: np.random.Generator) -> None:
        """Повний знімок: зображення + маска для сегментації, кроп диска (+ невалідні кропи) для класифікатора."""
        img_rel, mask_rel = f"images/{sample_id}.png", f"masks/{sample_id}.png"
        write_image(self.out / img_rel, resize_square(square, self.prepared))
        small = cv2.resize((mask > 0).astype(np.uint8) * 255, (self.prepared, self.prepared),
                           interpolation=cv2.INTER_AREA)
        write_image(self.out / mask_rel, ((small > 127) * 255).astype(np.uint8))
        with self._lock:
            self.seg_rows.append({"id": sample_id, "image": img_rel, "mask": mask_rel, "source": source,
                                  "group": group})
        self.add_disc_crops(sample_id, square, mask, source, group, glaucoma, negatives, rng)

    def add_disc_crops(self, sample_id: str, image: np.ndarray, mask: np.ndarray, source: str, group: str,
                       glaucoma: float, negatives: int, rng: np.random.Generator) -> None:
        bbox = mask_bbox(mask)
        if bbox is None:
            return
        box = disc_box(bbox, self.crop_expand)
        self.add_crop(f"{sample_id}_disc", crop_with_pad(image, box), source, group, glaucoma, 1.0)
        for k in range(negatives):
            neg = negative_box(image, box, box[2] - box[0], rng)
            if neg is not None:
                self.add_crop(f"{sample_id}_neg{k}", crop_with_pad(image, neg), source, group, UNKNOWN, 0.0)

    def add_crop(self, sample_id: str, crop: np.ndarray, source: str, group: str,
                 glaucoma: float, valid: float) -> None:
        rel = f"crops/{sample_id}.png"
        write_image(self.out / rel, resize_square(pad_to_square(crop), self.crop_size))
        with self._lock:
            self.cls_rows.append({"id": sample_id, "image": rel, "source": source, "group": group,
                                  "glaucoma": glaucoma, "valid_image": valid})

    def close(self, meta: dict) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        manifests = ((SEG_MANIFEST, SEG_FIELDS, self.seg_rows), (CLS_MANIFEST, CLS_FIELDS, self.cls_rows))
        for name, fields, rows in manifests:
            if not rows:
                continue
            rows = sorted(rows, key=lambda r: r["id"])
            with open(self.out / name, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        info = {**meta, "prepared_size": self.prepared, "crop_size": self.crop_size, "crop_expand": self.crop_expand,
                "segmentation_samples": len(self.seg_rows), "classification_samples": len(self.cls_rows),
                "created": date.today().isoformat()}
        write_json(self.out / "meta.json", info)
        return info


def _run_parallel(items: list, fn: Callable, label: str) -> None:
    log = get_logger()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, _ in enumerate(pool.map(fn, items), 1):
            if i % 100 == 0 or i == len(items):
                log.info("%s: %d/%d", label, i, len(items))


def _find_root(raw: Path, marker: str) -> Path:
    if (raw / marker).exists():
        return raw
    for child in sorted(raw.rglob(marker)):
        return child.parent
    raise FileNotFoundError(f"Не знайдено '{marker}' у {raw}. Розпакуйте архів датасету в цю теку.")


# ---------------------------------------------------------------------- PAPILA
def _papila_diagnosis(root: Path) -> dict[tuple[str, str], int]:
    import openpyxl

    result = {}
    for eye, fname in (("OD", "patient_data_od.xlsx"), ("OS", "patient_data_os.xlsx")):
        sheet = openpyxl.load_workbook(root / "ClinicalData" / fname, read_only=True, data_only=True).active
        rows = list(sheet.iter_rows(values_only=True))
        column = list(rows[0]).index("Diagnosis")
        for row in rows[1:]:
            if isinstance(row[0], str) and row[0].startswith("#") and row[column] is not None:
                result[(row[0][1:], eye)] = int(row[column])
    return result


def prepare_papila(raw: str | Path, out: str | Path, size: int = 576, crop_size: int = 256, expand: float = 0.3,
                   suspect_as: str = "unknown", negatives: int = 1, seed: int = 0) -> dict:
    """PAPILA (Kovalyk et al., 2022): 488 знімків, контури диска двох експертів (об'єднання), діагноз
    0 — здоровий, 1 — глаукома, 2 — підозра (``suspect_as``: unknown | positive | negative)."""
    root = _find_root(Path(raw), "FundusImages")
    diagnosis = _papila_diagnosis(root)
    suspect = {"unknown": UNKNOWN, "positive": 1.0, "negative": 0.0}[suspect_as]
    label_of = {0: 0.0, 1: 1.0, 2: suspect}
    writer = DatasetWriter(out, size, crop_size, expand)
    images = sorted((root / "FundusImages").glob("RET*.jpg"))
    contours = root / "ExpertsSegmentations" / "Contours"

    def process(item: tuple[int, Path]) -> None:
        i, path = item
        stem = path.stem  # RET002OD
        pid, eye = stem[3:-2], stem[-2:]
        img = read_rgb(path)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for expert in ("exp1", "exp2"):
            file = contours / f"{stem}_disc_{expert}.txt"
            if file.is_file():
                pts = np.round(np.loadtxt(file)).astype(np.int32)
                cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
        if not mask.any():
            get_logger().warning("PAPILA %s: немає контурів диска — пропущено", stem)
            return
        square, mask_sq, _ = fundus_square(img, mask)
        glaucoma = label_of.get(diagnosis.get((pid, eye), -1), UNKNOWN)
        writer.add_fundus(f"papila_{stem}", square, mask_sq, "papila", f"papila_{pid}", glaucoma, negatives,
                          np.random.default_rng(seed + i))

    _run_parallel(list(enumerate(images)), process, "PAPILA")
    return writer.close({"name": "PAPILA", "license": "GPL-3.0+", "url": "https://doi.org/10.6084/m9.figshare.14798004",
                         "citation": "Kovalyk O. et al. PAPILA. Scientific Data 9, 291 (2022)",
                         "suspect_as": suspect_as})


# ---------------------------------------------------------------------- ACRIMA
def prepare_acrima(raw: str | Path, out: str | Path, crop_size: int = 256) -> dict:
    """ACRIMA (Diaz-Pinto et al., 2019): 705 кропів диска; '_g_' у назві — глаукома."""
    root = _find_root(Path(raw), "Images")
    writer = DatasetWriter(out, 0, crop_size, 0.0)
    files = sorted((root / "Images").glob("*.jpg"))

    def process(path: Path) -> None:
        glaucoma = 1.0 if "_g_" in path.name else 0.0
        writer.add_crop(f"acrima_{path.stem}", read_rgb(path), "acrima", f"acrima_{path.stem}", glaucoma, 1.0)

    _run_parallel(files, process, "ACRIMA")
    return writer.close({"name": "ACRIMA", "license": "CC BY 4.0", "url": "https://doi.org/10.6084/m9.figshare.7613135",
                         "citation": "Diaz-Pinto A. et al. BioMedical Engineering OnLine 18, 29 (2019)"})


# ---------------------------------------------------------------------- Zenodo DG (REFUGE, Drishti-GS, ORIGA, RIGA)
def _zdg_disc_mask(raw: Path, rel: str) -> np.ndarray:
    """Маска диска: значення < 200 (128 — кільце диска, 0 — екскавація). RIGA: голосування 6 анотаторів."""
    path = raw / rel
    if path.is_file():
        return (read_gray(path) < 200).astype(np.uint8)
    variants = sorted(path.parent.glob(f"{path.stem}-*{path.suffix}"))
    if not variants:
        raise FileNotFoundError(f"Немає маски {path}")
    votes = sum((read_gray(v) < 200).astype(np.int32) for v in variants)
    return (votes * 2 > len(variants)).astype(np.uint8)


def prepare_zenodo_dg(raw: str | Path, out: str | Path, size: int = 576, crop_size: int = 256,
                      expand: float = 0.3) -> dict:
    """Fundus DG (Chen et al., MICCAI 2023, zenodo 8009107): 1421 ROI 800×800 з масками диска/екскавації
    з REFUGE, Drishti-GS, ORIGA, RIGA (BinRushed, Magrabia). Глаукома відома лише для REFUGE (префікс g/n)."""
    raw = Path(raw)
    root = _find_root(raw, "REFUGE_train.csv")
    items = []
    for csv_path in sorted(root.glob("*_t*.csv")):
        domain = csv_path.stem.rsplit("_", 1)[0]
        with open(csv_path, encoding="utf-8", newline="") as fh:
            items += [(domain, row["image"], row["mask"]) for row in csv.DictReader(fh)]
    writer = DatasetWriter(out, size, crop_size, expand)

    def process(item: tuple[str, str, str]) -> None:
        domain, img_rel, mask_rel = item
        img = read_rgb(root / img_rel)
        mask = _zdg_disc_mask(root, mask_rel)
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        square, mask_sq, _ = fundus_square(img, mask)
        sample_id = "zdg_" + re.sub(r"[^A-Za-z0-9]+", "_", str(Path(img_rel).with_suffix(""))).strip("_")
        name = Path(img_rel).name.lower()
        glaucoma = UNKNOWN
        if domain == "REFUGE" and name[0] in "gn":
            glaucoma = 1.0 if name[0] == "g" else 0.0
        writer.add_fundus(sample_id, square, mask_sq, f"zdg_{domain.lower()}", sample_id, glaucoma, 0,
                          np.random.default_rng(0))

    _run_parallel(items, process, "Zenodo DG")
    return writer.close({"name": "Fundus DG (REFUGE, Drishti-GS, ORIGA, RIGA)", "license": "CC BY 4.0",
                         "url": "https://zenodo.org/records/8009107",
                         "citation": "Chen Z. et al. Treasure in Distribution, MICCAI 2023"})
