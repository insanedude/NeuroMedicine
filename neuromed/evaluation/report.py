"""Звіт у форматі таблиць статті: стовпчик — ітерація (основа), рядок — метрика."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from neuromed.utils import format_duration

STATUS_UA = {"baseline": "перша ітерація", "accepted": "прийнято", "rejected": "провал → відкат"}


def _rows(task: str, labels: list[str]) -> list[tuple[str, str, str]]:
    common_head = [
        ("Основа (ширина 1-го шару)", "base", "int"),
        ("Частка вирізання", "ratio", "pct"),
        ("Параметри", "params", "int"),
        ("Розмір, МБ", "size_mb", "float"),
        ("Епох до достатнього результату", "epochs_to_goal", "int"),
        ("Епох в ітерації", "epochs", "int"),
        ("Час навчання", "train_seconds", "time"),
        ("Загальний час навчання", "total_seconds", "time"),
        ("Час інференсу (CPU), с", "inference_s", "sec"),
    ]
    if task == "segmenter":
        body = [
            ("Прогнозів (компонент маски)", "test.n_predictions", "int"),
            ("TP (знайдені диски)", "test.det_tp", "int"),
            ("TN", "test.det_tn", "int"),
            ("FP", "test.det_fp", "int"),
            ("FN (пропущені диски)", "test.det_fn", "int"),
            ("Додаткові хибні виявлення", "test.extra_detections", "int"),
            ("Точність виявлення (precision)", "test.det_precision", "pct"),
            ("Повнота виявлення (recall)", "test.det_recall", "pct"),
            ("F1 виявлення", "test.det_f1", "pct"),
            ("Dice маски", "test.dice", "pct"),
            ("Піксельна точність (precision)", "test.pix_precision", "pct"),
            ("Піксельна повнота (recall)", "test.pix_recall", "pct"),
            ("Піксельна влучність (accuracy)", "test.pix_accuracy", "pct"),
        ]
    else:
        main = labels[0]
        body = [
            (f"Позитивних прогнозів ({main})", "test.n_predictions", "int"),
            ("TP", f"test.{main}_tp", "int"),
            ("TN", f"test.{main}_tn", "int"),
            ("FP", f"test.{main}_fp", "int"),
            ("FN", f"test.{main}_fn", "int"),
            ("Поріг (калібрований)", f"test.{main}_threshold", "float"),
            ("Точність (precision)", f"test.{main}_precision", "pct"),
            ("Повнота (recall)", f"test.{main}_recall", "pct"),
            ("Специфічність", f"test.{main}_specificity", "pct"),
            ("Влучність (accuracy)", f"test.{main}_accuracy", "pct"),
            ("F1", f"test.{main}_f1", "pct"),
            ("ROC AUC", f"test.{main}_auc", "pct"),
        ]
        for extra in labels[1:]:
            body += [(f"F1 ({extra})", f"test.{extra}_f1", "pct"),
                     (f"Повнота ({extra})", f"test.{extra}_recall", "pct")]
    tail = [("Виявлення крайніх випадків", "test.edge_detected", "pct"), ("Статус", "status", "status")]
    return common_head + body + tail


def _get(record: dict, path: str):
    value = record
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _fmt(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if kind == "int":
        return f"{int(value):,}".replace(",", " ")
    if kind == "pct":
        return f"{100 * value:.2f}%"
    if kind == "time":
        return format_duration(value)
    if kind == "sec":
        return f"{value:.3f}"
    if kind == "float":
        return f"{value:.3f}"
    if kind == "status":
        return STATUS_UA.get(value, str(value))
    return str(value)


def markdown_table(records: list[dict], task: str, labels: list[str]) -> str:
    header = "| Конфігурація | " + " | ".join(
        f"it{r['iteration']} x{r.get('base', '?')}" for r in records) + " |"
    lines = [header, "|" + "---|" * (len(records) + 1)]
    for title, path, kind in _rows(task, labels):
        lines.append(f"| {title} | " + " | ".join(_fmt(_get(r, path), kind) for r in records) + " |")
    return "\n".join(lines)


def write_csv(records: list[dict], task: str, labels: list[str], path: Path) -> None:
    rows = _rows(task, labels)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric"] + [f"it{r['iteration']}" for r in records])
        for title, key, _ in rows:
            writer.writerow([title] + [_get(r, key) for r in records])


def plot_curve(records: list[dict], task: str, labels: list[str], path: Path) -> None:
    """Дві панелі зі спільною віссю X (параметри, лог-шкала): якість і час інференсу на CPU."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if task == "segmenter":
        series = [("Повнота виявлення", "test.det_recall"), ("Dice", "test.dice")]
    else:
        series = [("ROC AUC", f"test.{labels[0]}_auc"), ("Повнота", f"test.{labels[0]}_recall"),
                  ("Специфічність", f"test.{labels[0]}_specificity")]
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # категоріальні слоти 1–3 еталонної палітри
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=surface, sharex=True)
    params = [r["params"] for r in records]
    accepted = [r["status"] != "rejected" for r in records]
    for ax in axes:
        ax.set_facecolor(surface)
        ax.set_xscale("log")
        ax.grid(True, color=grid, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)
        ax.set_xlabel("Параметри (лог. шкала)", color=muted, fontsize=9)
    axes[0].invert_xaxis()  # вісь спільна, тож обидві панелі: зліва — повна мережа, праворуч — стиснені

    ax = axes[0]
    for (name, key), color in zip(series, colors):
        values = [_get(r, key) for r in records]
        xs = [p for p, v, ok in zip(params, values, accepted) if v is not None and ok]
        ys = [100 * v for v, ok in zip(values, accepted) if v is not None and ok]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6, label=name)
        rx = [p for p, v, ok in zip(params, values, accepted) if v is not None and not ok]
        ry = [100 * v for v, ok in zip(values, accepted) if v is not None and not ok]
        if rx:
            ax.plot(rx, ry, linestyle="none", marker="X", markersize=8, color=color, markeredgecolor=surface)
    ax.set_title("Якість на тестовій вибірці, %", color=ink, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=ink)

    ax = axes[1]
    times = [r.get("inference_s") for r in records]
    ax.plot([p for p, t, ok in zip(params, times, accepted) if ok and t],
            [t for t, ok in zip(times, accepted) if ok and t], color=colors[0], linewidth=2, marker="o", markersize=6)
    ax.set_title("Час інференсу на CPU, с / зображення", color=ink, fontsize=11, loc="left")
    fig.text(0.01, 0.01, "× — ітерація, що провалила критерії (мережу взято з попередньої)", color=muted, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)


def write_report(records: list[dict], task: str, labels: list[str], out_dir: Path, title: str,
                 final_iteration: int | None, notes: list[str] | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(records, task, labels, out_dir / "report.csv")
    plot_curve(records, task, labels, out_dir / "compression_curve.png")
    lines = [f"# {title}", ""]
    if final_iteration is not None:
        final = next(r for r in records if r["iteration"] == final_iteration)
        base = records[0]
        lines += [
            f"**Результат:** ітерація {final_iteration} (основа x{final.get('base')}), "
            f"{final['params'] / base['params']:.1%} параметрів першої ітерації, "
            f"інференс на CPU {final['inference_s']:.3f} с проти {base['inference_s']:.3f} с "
            f"(×{base['inference_s'] / max(final['inference_s'], 1e-9):.1f} швидше).", ""]
    lines += [markdown_table(records, task, labels), "", "![Крива стиснення](compression_curve.png)", ""]
    if notes:
        lines += ["## Примітки", ""] + [f"- {n}" for n in notes] + [""]
    lines += ["Метрики — на тестовій вибірці, яка не брала участі ні в навчанні, ні у виборі моделі. "
              "Час інференсу — медіана для одного зображення на CPU."]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
