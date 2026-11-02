"""Командний рядок: ``python -m neuromed <команда>``.

prepare   — перетворити публічні датасети (або згенерувати синтетичний) у формат проєкту
train     — лише ітерація 0: transfer learning з VGG-16 і донавчання
compress  — увесь алгоритм зжимання (ітерація 0, якщо її немає, + цикл стиснення)
evaluate  — метрики чекпойнта на вибірці
benchmark — таблиця порівняння кількох чекпойнтів (як у статті)
predict   — скринінг знімків: маска диска, кропи, класифікація
info      — архітектура та метадані чекпойнта
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neuromed.utils import setup_logging

DATASETS = ("papila", "acrima", "zenodo_dg")


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", required=True, help="YAML-конфігурація (див. configs/)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="КЛЮЧ=ЗНАЧЕННЯ",
                   help="перевизначити параметр, напр. --set train.max_epochs=5 (можна кілька разів)")
    p.add_argument("--resume", action="store_true", help="продовжити перерваний запуск з state.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neuromed", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="підготовка датасету")
    p.add_argument("dataset", choices=DATASETS + ("synthetic", "all"))
    p.add_argument("--raw", help="тека з розпакованим архівом (типово data/raw/<датасет>)")
    p.add_argument("--out", help="тека результату (типово data/processed/<датасет>)")
    p.add_argument("--size", type=int, default=576, help="сторона квадратного знімка для сегментації")
    p.add_argument("--crop-size", type=int, default=256, help="сторона кропу диска для класифікатора")
    p.add_argument("--expand", type=float, default=0.3, help="розширення рамки кропу навколо диска")
    p.add_argument("--suspect-as", choices=("unknown", "positive", "negative"), default="unknown",
                   help="PAPILA: як трактувати «підозру на глаукому»")
    p.add_argument("--negatives", type=int, default=1, help="невалідних кропів (поза диском) на знімок")
    p.add_argument("--count", type=int, default=160, help="synthetic: кількість знімків")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("train", help="ітерація 0: transfer learning + донавчання")
    _add_config_args(p)

    p = sub.add_parser("compress", help="увесь алгоритм зжимання")
    _add_config_args(p)
    p.add_argument("--baseline", help="готовий чекпойнт ітерації 0 (інакше буде натренований)")

    p = sub.add_parser("evaluate", help="метрики чекпойнта")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=("val", "test", "train", "all"), default="test")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="КЛЮЧ=ЗНАЧЕННЯ",
                   help="перевизначити параметр збереженої конфігурації (напр. data.datasets=[...])")
    p.add_argument("--device", default="auto")

    p = sub.add_parser("benchmark", help="таблиця порівняння чекпойнтів")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--out", default="runs/benchmark", help="тека для report.md/csv/png")
    p.add_argument("--device", default="auto")

    p = sub.add_parser("predict", help="скринінг знімків")
    p.add_argument("--seg", required=True, help="чекпойнт сегментатора")
    p.add_argument("--cls", help="чекпойнт класифікатора (необов'язково)")
    p.add_argument("--input", required=True, help="зображення або тека")
    p.add_argument("--out", default="runs/predictions")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threshold", type=float, help="поріг маски (типово з чекпойнта)")

    p = sub.add_parser("info", help="опис чекпойнта")
    p.add_argument("--checkpoint", required=True)
    return parser


# ---------------------------------------------------------------------- команди
def cmd_prepare(args) -> None:
    from neuromed.data import prepare
    from neuromed.data.synthetic import generate_synthetic

    names = DATASETS if args.dataset == "all" else (args.dataset,)
    for name in names:
        raw = args.raw if args.raw and args.dataset != "all" else f"data/raw/{name}"
        out = args.out if args.out and args.dataset != "all" else f"data/processed/{name}"
        if name == "papila":
            info = prepare.prepare_papila(raw, out, args.size, args.crop_size, args.expand, args.suspect_as,
                                          args.negatives, args.seed)
        elif name == "acrima":
            info = prepare.prepare_acrima(raw, out, args.crop_size)
        elif name == "zenodo_dg":
            info = prepare.prepare_zenodo_dg(raw, out, args.size, args.crop_size, args.expand)
        else:
            info = generate_synthetic(out, args.count, args.size, args.crop_size, args.expand, args.seed)
        print(f"{name}: сегментація {info['segmentation_samples']}, "
              f"класифікація {info['classification_samples']} → {out}")


def cmd_compress(args, baseline_only: bool) -> None:
    from neuromed.compression.loop import CompressionRun
    from neuromed.config import load_config

    cfg = load_config(args.config, args.overrides)
    run = CompressionRun(cfg, resume=args.resume)
    run.run(baseline_path=getattr(args, "baseline", None), max_iterations=0 if baseline_only else None)


def _load_for_eval(checkpoint: str, overrides: list[str], device_name: str):
    from neuromed.config import apply_overrides, config_from_dict
    from neuromed.data.datamodule import DataModule
    from neuromed.training.checkpoint import load_checkpoint
    from neuromed.training.trainer import Trainer
    from neuromed.utils import resolve_device

    device = resolve_device(device_name)
    model, payload = load_checkpoint(checkpoint, device)
    raw = apply_overrides(dict(payload["config"]), overrides)
    raw["model"] = {**raw.get("model", {}), "pretrained": False}
    cfg = config_from_dict(raw)
    data = DataModule(cfg, device)
    trainer = Trainer(cfg, data, device, Path(checkpoint).parent)
    return model, payload, cfg, data, trainer


def _evaluate_record(checkpoint: str, overrides: list[str], device: str, split: str = "test") -> tuple[dict, dict]:
    from neuromed.evaluation.benchmark import measure_cpu_inference, model_stats

    model, payload, cfg, data, trainer = _load_for_eval(checkpoint, overrides, device)
    val = trainer.evaluate(model, "val")
    result = trainer.evaluate(model, split, thresholds=val.thresholds or None)
    shape = (data.channels, cfg.data.input_size, cfg.data.input_size)
    record = {"iteration": payload["meta"].get("iteration", 0), "status": "accepted", "ratio": None,
              **model_stats(model), "inference_s": measure_cpu_inference(model, shape, cfg.eval.benchmark_threads,
                                                                         cfg.eval.benchmark_runs),
              "val": val.metrics, "test": result.metrics, "thresholds": val.thresholds, "checkpoint": checkpoint}
    return record, payload


def cmd_evaluate(args) -> None:
    record, payload = _evaluate_record(args.checkpoint, args.overrides, args.device, args.split)
    print(f"Чекпойнт: {args.checkpoint} ({payload['arch']['type']}, основа x{record['base']}, "
          f"{record['params']:,} параметрів, {record['size_mb']:.2f} МБ, CPU {record['inference_s']:.3f} с)")
    print(f"Вибірка: {args.split}")
    for key, value in record["test"].items():
        print(f"  {key:24s} {value:.4f}" if isinstance(value, float) else f"  {key:24s} {value}")
    if record["thresholds"]:
        print("Пороги (калібровані на val):", record["thresholds"])


def cmd_benchmark(args) -> None:
    from neuromed.evaluation.report import write_report

    records, task, labels = [], None, None
    for checkpoint in args.checkpoints:
        record, payload = _evaluate_record(checkpoint, [], args.device)
        task, labels = payload["meta"]["task"], payload["meta"]["labels"]
        records.append(record)
    path = write_report(records, task, labels, Path(args.out), "Порівняння чекпойнтів", None,
                        [f"{r['iteration']}: {r['checkpoint']}" for r in records])
    print(f"Звіт: {path}")


def cmd_predict(args) -> None:
    from neuromed.inference.predict import ScreeningPipeline, run_batch

    pipeline = ScreeningPipeline(args.seg, args.cls, args.device, args.threshold)
    results = run_batch(pipeline, args.input, args.out)
    for r in results:
        line = f"{Path(r['image']).name}: диск {'знайдено' if r['disc_found'] else 'НЕ знайдено'}"
        if "glaucoma_suspected" in r:
            line += f", підозра на глаукому: {'ТАК' if r['glaucoma_suspected'] else 'ні'}"
        print(line + f" ({r['seconds']:.2f} с)")
    print(f"Результати: {Path(args.out) / 'results.json'}")


def cmd_info(args) -> None:
    from neuromed.training.checkpoint import load_checkpoint
    from neuromed.utils import count_parameters, state_dict_bytes

    model, payload = load_checkpoint(args.checkpoint)
    meta = payload["meta"]
    print(f"Тип: {payload['arch']['type']}; ітерація {meta.get('iteration')}; вхід "
          f"{payload['arch']['in_channels']}×{meta.get('input_size')}²; мітки {meta.get('labels')}")
    print(f"Параметрів: {count_parameters(model):,}; ваги {state_dict_bytes(model) / 1e6:.2f} МБ")
    print("Ширини шарів:", payload["arch"]["widths"])
    for split in ("val", "test"):
        metrics = meta.get(split) or {}
        keys = [k for k in metrics if k.endswith(("recall", "precision", "dice", "auc", "f1"))]
        print(f"{split}: " + ", ".join(f"{k} {metrics[k]:.4f}" for k in keys))
    if meta.get("thresholds"):
        print("Пороги:", meta["thresholds"])


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            cmd_prepare(args)
        elif args.command in ("train", "compress"):
            cmd_compress(args, baseline_only=args.command == "train")
        elif args.command == "evaluate":
            cmd_evaluate(args)
        elif args.command == "benchmark":
            cmd_benchmark(args)
        elif args.command == "predict":
            cmd_predict(args)
        elif args.command == "info":
            cmd_info(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"Помилка: {exc}", file=sys.stderr)
        return 1
    return 0
