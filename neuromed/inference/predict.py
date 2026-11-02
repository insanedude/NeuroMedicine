"""Скринінговий конвеєр: знімок → маска диска → кропи навколо знайдених компонент → класифікатор.

Повторює логіку test_retina_nerve_definitor.py з вихідного репозиторію (виділення
компонент маски, розширена рамка, кроп з оригінального кольорового знімка), але
з тією самою попередньою обробкою, з якою модель навчалась (береться з чекпойнта).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from neuromed.config import PreprocessConfig
from neuromed.data.prepare import crop_with_pad, disc_box
from neuromed.data.preprocessing import fundus_square, model_input, normalize, read_rgb, write_image
from neuromed.training.checkpoint import load_checkpoint
from neuromed.utils import to_builtin

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ScreeningPipeline:
    def __init__(self, seg_path: str | Path, cls_path: str | Path | None = None, device: str = "cpu",
                 threshold: float | None = None) -> None:
        self.device = torch.device(device)
        self.seg, payload = load_checkpoint(seg_path, self.device)
        meta = payload["meta"]
        if meta["task"] != "segmenter":
            raise ValueError(f"{seg_path} не є чекпойнтом сегментатора")
        self.seg_pre = PreprocessConfig(**meta["preprocess"])
        self.seg_size, self.seg_prepared = meta["input_size"], meta.get("prepared_size")
        self.threshold = threshold if threshold is not None else meta.get("eval_threshold", 0.5)
        self.min_component_frac = payload["config"].get("eval", {}).get("min_component_frac", 0.0005)
        self.cls = None
        if cls_path:
            self.cls, payload = load_checkpoint(cls_path, self.device)
            meta = payload["meta"]
            if meta["task"] != "classifier":
                raise ValueError(f"{cls_path} не є чекпойнтом класифікатора")
            self.cls_pre = PreprocessConfig(**meta["preprocess"])
            self.cls_size, self.cls_prepared = meta["input_size"], meta.get("prepared_size")
            self.labels = list(meta["labels"])
            default = meta.get("eval_threshold", 0.5)
            self.cls_thresholds = {name: meta.get("thresholds", {}).get(name, default) for name in self.labels}
            self.crop_expand = meta.get("crop_expand", 0.3)

    def _tensor(self, arr: np.ndarray) -> torch.Tensor:
        return normalize(torch.from_numpy(arr)[None].to(self.device))

    @torch.inference_mode()
    def analyze(self, img_rgb: np.ndarray) -> dict:
        start = time.perf_counter()
        square, _, info = fundus_square(img_rgb)
        x = self._tensor(model_input(square, self.seg_pre, self.seg_size, self.seg_prepared))
        prob = torch.sigmoid(self.seg(x))[0, 0].float().cpu().numpy()
        binary = (prob >= self.threshold).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        min_area = max(1, int(self.min_component_frac * prob.size))
        scale = info.side / self.seg_size
        findings = []
        for i in range(1, n):
            x0, y0, w, h, area = stats[i]
            if area < min_area:
                continue
            box_sq = (x0 * scale, y0 * scale, (x0 + w) * scale, (y0 + h) * scale)
            finding = {"score": float(prob[labels == i].mean()), "area_frac": float(area / prob.size),
                       "bbox_square": [int(round(v)) for v in box_sq]}
            sx0, sy0 = info.to_source(box_sq[0], box_sq[1])
            sx1, sy1 = info.to_source(box_sq[2], box_sq[3])
            finding["bbox"] = [int(round(v)) for v in (sx0, sy0, sx1, sy1)]
            if self.cls is not None:
                crop_box = disc_box(tuple(int(round(v)) for v in box_sq), self.crop_expand)
                crop = crop_with_pad(square, crop_box)
                logits = self.cls(self._tensor(model_input(crop, self.cls_pre, self.cls_size, self.cls_prepared)))
                probs = torch.sigmoid(logits)[0].float().cpu().numpy()
                finding["probs"] = {name: float(p) for name, p in zip(self.labels, probs)}
                finding["flags"] = {name: bool(p >= self.cls_thresholds[name]) for name, p in zip(self.labels, probs)}
                finding["crop_box_square"] = list(crop_box)
                finding["_crop"] = crop
            findings.append(finding)
        findings.sort(key=lambda f: f["score"], reverse=True)
        # на знімку один диск: висновок робиться за найупевненішим валідним виявленням,
        # решта (уламки маски на нечітких знімках) лишається у findings як додаткові виявлення
        valid = [f for f in findings if f.get("flags", {}).get("valid_image", True)]
        for f in findings:
            f["primary"] = bool(valid) and f is valid[0]
        summary = {"disc_found": bool(valid), "extra_detections": max(0, len(findings) - 1)}
        if self.cls is not None and "glaucoma" in self.labels:
            summary["glaucoma_suspected"] = bool(valid) and valid[0]["flags"]["glaucoma"]
        return {"findings": findings, "summary": summary, "prob": prob, "square": square,
                "seconds": time.perf_counter() - start}

    def run_file(self, path: str | Path, out_dir: str | Path) -> dict:
        path, out_dir = Path(path), Path(out_dir)
        result = self.analyze(read_rgb(path))
        stem = path.stem
        square, prob = result["square"], result["prob"]
        side = square.shape[0]
        prob_full = cv2.resize(prob, (side, side), interpolation=cv2.INTER_LINEAR)
        write_image(out_dir / f"{stem}_mask.png", ((prob_full >= self.threshold) * 255).astype(np.uint8))

        overlay = square.copy()
        thickness = max(2, side // 300)
        contours, _ = cv2.findContours((prob_full >= self.threshold).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness)
        for k, f in enumerate(result["findings"]):
            x0, y0, x1, y1 = f.get("crop_box_square", f["bbox_square"])
            color = (255, 255, 0) if f["primary"] else (160, 160, 160)  # основний диск — жовтий, решта — сірі
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, thickness)
            if "probs" in f:
                text = " ".join(f"{name[:1].upper()}{p:.2f}" for name, p in f["probs"].items())
                cv2.putText(overlay, text, (max(0, x0), max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            side / 1200, color, thickness)
                write_image(out_dir / f"{stem}_disc{k}.png", f.pop("_crop"))
                f["crop"] = f"{stem}_disc{k}.png"
        if side > 1024:
            overlay = cv2.resize(overlay, (1024, 1024), interpolation=cv2.INTER_AREA)
        write_image(out_dir / f"{stem}_overlay.jpg", overlay)
        return to_builtin({"image": str(path), "seconds": result["seconds"], **result["summary"],
                           "findings": result["findings"]})


def run_batch(pipeline: ScreeningPipeline, inputs: str | Path, out_dir: str | Path) -> list[dict]:
    inputs, out_dir = Path(inputs), Path(out_dir)
    files = [inputs] if inputs.is_file() else sorted(p for p in inputs.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"Не знайдено зображень у {inputs}")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [pipeline.run_file(p, out_dir) for p in files]
    with open(out_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    with open(out_dir / "results.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        header = ["image", "disc_found", "glaucoma_suspected", "detections", "disc_score", "disc_probs", "seconds"]
        writer.writerow(header)
        for r in results:
            best = next((f for f in r["findings"] if f["primary"]), r["findings"][0] if r["findings"] else {})
            writer.writerow([r["image"], r["disc_found"], r.get("glaucoma_suspected", ""), len(r["findings"]),
                             f"{best.get('score', 0):.3f}" if best else "",
                             json.dumps(best.get("probs", {})), f"{r['seconds']:.3f}"])
    return results
