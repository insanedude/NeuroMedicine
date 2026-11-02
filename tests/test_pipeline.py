"""Наскрізні запуски на крихітному синтетичному датасеті (CPU, секунди): CLI, відновлення, звіт, скринінг."""

from __future__ import annotations

import json

import pytest

from neuromed.cli import main
from neuromed.compression.loop import CompressionRun
from neuromed.config import config_from_dict, save_config
from neuromed.data.preprocessing import read_rgb
from neuromed.data.records import load_records
from neuromed.inference.predict import ScreeningPipeline
from neuromed.utils import read_json

TINY_COMPRESS = {"max_iterations": 1, "spread_epochs": 1, "squeeze_epochs": 1, "stabilize_min_epochs": 1,
                 "stabilize_max_epochs": 1, "stabilize_patience": 1, "min_channels": 2}


def _config(out_dir, dataset, name, model, train, input_size, **extra):
    return config_from_dict({
        "name": name, "output_dir": str(out_dir), "device": "cpu",
        "data": {"datasets": [str(dataset)], "input_size": input_size, "batch_size": 4, "chunks": 2},
        "model": {"base": 8, "pretrained": False, **model},
        "train": {"max_epochs": 2, "min_epochs": 1, "lr": 1e-3, "patience": 1, **train},
        "compress": TINY_COMPRESS,
        "eval": {"benchmark_runs": 1, "edge_cases": 3},
        **extra,
    })


@pytest.fixture(scope="module")
def seg_run(tmp_path_factory, synthetic_dir):
    """`train` (лише ітерація 0) → `compress --resume` через командний рядок."""
    out = tmp_path_factory.mktemp("runs")
    cfg = _config(out, synthetic_dir, "seg", {"type": "segmenter", "labels": ["disc"]},
                  {"target": {"det_recall": 0.5}, "selection": ["det_recall", "dice"]}, 64)
    cfg_path = out / "seg.yaml"
    save_config(cfg, cfg_path)
    assert main(["train", "-c", str(cfg_path)]) == 0
    state = read_json(cfg.run_dir / "state.json")
    assert [r["iteration"] for r in state["records"]] == [0] and not state.get("finished")
    assert main(["train", "-c", str(cfg_path)]) == 1  # повторний запуск без --resume заборонено
    assert main(["compress", "-c", str(cfg_path), "--resume"]) == 0
    return cfg.run_dir


def test_segmentation_run_outputs(seg_run):
    state = read_json(seg_run / "state.json")
    assert state["finished"]
    records = state["records"]
    assert [r["iteration"] for r in records] == [0, 1]
    assert records[1]["params"] < records[0]["params"]
    assert records[1]["norm_ratio"] <= 0.05 + 1e-3  # «both»: норма вирізаних ваг обмежена 5 %
    for name in ("report.md", "report.csv", "compression_curve.png", "history.csv", "log.txt", "splits.json",
                 "checkpoints/final.pt"):
        assert (seg_run / name).is_file(), name
    assert "Повнота" in (seg_run / "report.md").read_text(encoding="utf-8")


def test_cli_evaluate_info_benchmark_predict(seg_run, synthetic_dir, tmp_path):
    final = str(seg_run / "checkpoints" / "final.pt")
    first = str(next((seg_run / "checkpoints").glob("iter00_*.pt")))
    assert main(["evaluate", "--checkpoint", final, "--split", "val", "--device", "cpu"]) == 0
    assert main(["info", "--checkpoint", final]) == 0
    assert main(["benchmark", "--checkpoints", first, final, "--out", str(tmp_path / "bench"),
                 "--device", "cpu"]) == 0
    assert (tmp_path / "bench" / "report.md").is_file()
    image = load_records([synthetic_dir], "segmenter", ["disc"])[0].image
    assert main(["predict", "--seg", final, "--input", str(image), "--out", str(tmp_path / "pred")]) == 0
    results = json.loads((tmp_path / "pred" / "results.json").read_text(encoding="utf-8"))
    assert len(results) == 1 and "disc_found" in results[0]
    assert (tmp_path / "pred" / f"{image.stem}_overlay.jpg").is_file()


def test_classifier_run_with_distillation_and_screening(seg_run, synthetic_dir, tmp_path):
    labels = ["glaucoma", "valid_image"]
    cfg = _config(tmp_path, synthetic_dir, "cls", {"type": "classifier", "labels": labels},
                  {"target": {}, "selection": ["valid_image_f1"]}, 32,
                  distill={"enabled": True, "alpha": 0.5, "temperature": 2.0})
    cfg.data.preprocess.grayscale = False
    cfg.eval.calibrate_recall = {"glaucoma": 0.95}
    final = CompressionRun(cfg).run()
    state = read_json(cfg.run_dir / "state.json")
    assert state["finished"] and len(state["records"]) == 2
    assert "glaucoma" in state["records"][0]["thresholds"]

    pipeline = ScreeningPipeline(seg_run / "checkpoints" / "final.pt", final)
    image = load_records([synthetic_dir], "segmenter", ["disc"])[0].image
    result = pipeline.analyze(read_rgb(image))
    assert set(result["summary"]) == {"disc_found", "glaucoma_suspected"}
    for finding in result["findings"]:
        assert set(finding["probs"]) == set(labels)
