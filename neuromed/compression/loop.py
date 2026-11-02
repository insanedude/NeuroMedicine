"""Цикл зжимання — алгоритм з «Інструкцій для зжимання».

1. Трансплантуємо частину VGG-16 (перші чотири блоки) у нову мережу.
2. Донавчаємо, щоб встановились зв'язки на вищих рівнях і мережа спеціалізувалась.
3. Тренуємо до достатніх показників (повнота ~100 % — важливіше за точність).
4. Ітерація стиснення:
   4.1 dropout 0.1–0.2 → 0.5: модель розносить досвід на декілька нейронів;
   4.2 нормування ваг нейронів, які будуть вирізані, — «зганяємо» з них досвід;
   4.3 одночасно dropout 0.5 → 0 і частка нормування 0 → 0.95;
   4.4 вирізаємо непотрібні нейрони, далі — менший крок навчання;
   4.5 дотреновуємо, поки повнота й влучність не стабілізуються після стресу;
   4.6 повторюємо з 4.1, доки мережа не почне фейлити незалежно від кількості
       тренування; тоді беремо мережу з попередньої ітерації — це і є результат.
"""

from __future__ import annotations

import copy
import shutil
from dataclasses import dataclass
from pathlib import Path

from torch import nn

from neuromed.compression.planning import PrunePlan, plan_pruning
from neuromed.compression.regularizer import ChannelNormRegularizer
from neuromed.compression.schedule import constant, spread_schedule, squeeze_schedule
from neuromed.compression.surgery import apply_plan
from neuromed.config import Config, config_to_dict, save_config
from neuromed.data.datamodule import DataModule
from neuromed.evaluation.benchmark import measure_cpu_inference, model_stats
from neuromed.evaluation.report import write_report
from neuromed.models import create_model
from neuromed.training.checkpoint import load_checkpoint, save_checkpoint
from neuromed.training.monitor import PhaseMonitor, acceptance_goal, meets
from neuromed.training.trainer import Trainer
from neuromed.utils import format_duration, read_json, resolve_device, set_seed, setup_logging, write_json


@dataclass
class StepResult:
    model: nn.Module
    accepted: bool
    val: dict
    before_cut: dict
    after_cut: dict
    norm_ratio: float
    epochs: int
    epochs_to_goal: int | None
    seconds: float
    lr: float
    plan: PrunePlan


class CompressionRun:
    def __init__(self, cfg: Config, resume: bool = False) -> None:
        self.cfg = cfg
        self.run_dir = cfg.run_dir
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log = setup_logging(self.run_dir / "log.txt")
        set_seed(cfg.seed)
        # cudnn.benchmark вимкнено: у сталому режимі виграшу не дає, а кожна нова форма шарів
        # (після вирізання, неповні батчі оцінювання) коштує 10–14 с автотюнінгу
        self.device = resolve_device(cfg.device)
        self.state_path = self.run_dir / "state.json"
        if not resume and self.state_path.exists():
            raise FileExistsError(f"{self.run_dir} вже містить запуск. Додайте --resume або змініть name.")
        self.state: dict = read_json(self.state_path) if resume and self.state_path.exists() else {}
        save_config(cfg, self.run_dir / "config.yaml")
        self.log.info("Запуск '%s' на %s", cfg.name, self.device)
        self.data = DataModule(cfg, self.device)
        self.log.info("Дані: %s", self.data.summary())
        write_json(self.run_dir / "splits.json", self.data.splits())
        self.trainer = Trainer(cfg, self.data, self.device, self.run_dir)
        self.selection = list(cfg.train.selection)
        self.show = list(dict.fromkeys(self.selection + list(cfg.train.target) + list(cfg.compress.acceptance.minimum)))

    # ------------------------------------------------------------------ допоміжне
    def _meta(self, iteration: int, val: dict, test: dict, thresholds: dict) -> dict:
        meta = {"task": self.cfg.model.type, "labels": self.cfg.model.labels, "iteration": iteration,
                "input_size": self.cfg.data.input_size, "preprocess": vars(self.cfg.data.preprocess),
                "prepared_size": self.data.prepared_size, "thresholds": thresholds,
                "eval_threshold": self.cfg.eval.threshold, "val": val, "test": test}
        if self.cfg.model.type == "classifier":
            info = Path(self.cfg.data.datasets[0]) / "meta.json"
            meta["crop_expand"] = read_json(info).get("crop_expand", 0.3) if info.is_file() else 0.3
        return meta

    def _save_state(self) -> None:
        self.state["global_epoch"] = self.trainer.global_epoch
        write_json(self.state_path, self.state)

    def _record(self, model: nn.Module, iteration: int, status: str, *, ratio: float, epochs: int,
                epochs_to_goal: int | None, seconds: float, extra: dict | None = None) -> dict:
        """Оцінка на контрольній і тестовій вибірках, розмір, швидкість на CPU; збереження чекпойнта."""
        val_res = self.trainer.evaluate(model, "val")
        test_res = self.trainer.evaluate(model, "test", thresholds=val_res.thresholds or None)
        e = self.cfg.eval
        shape = (self.data.channels, self.cfg.data.input_size, self.cfg.data.input_size)
        self.state["total_seconds"] = self.state.get("total_seconds", 0.0) + seconds
        suffix = "_rejected" if status == "rejected" else ""
        ckpt = self.ckpt_dir / f"iter{iteration:02d}_x{model.widths['enc1.0']}{suffix}.pt"
        save_checkpoint(ckpt, model, self._meta(iteration, val_res.metrics, test_res.metrics, val_res.thresholds),
                        config_to_dict(self.cfg))
        record = {"iteration": iteration, "status": status, "ratio": ratio, **model_stats(model),
                  "epochs": epochs, "epochs_to_goal": epochs_to_goal, "train_seconds": seconds,
                  "total_seconds": self.state["total_seconds"],
                  "inference_s": measure_cpu_inference(model, shape, e.benchmark_threads, e.benchmark_runs),
                  "val": val_res.metrics, "test": test_res.metrics, "thresholds": val_res.thresholds,
                  "checkpoint": str(ckpt), **(extra or {})}
        self.state.setdefault("records", []).append(record)
        return record

    def _report(self) -> Path:
        records = self.state.get("records", [])
        final = self.state.get("final_iteration")
        notes = [f"Задача: {self.cfg.model.type}; датасети: {', '.join(self.cfg.data.datasets)}",
                 f"Частка вирізання на ітерацію: {self.cfg.compress.prune_ratio:.0%} "
                 f"(повтор з меншою після провалу: {'так' if self.cfg.compress.retry_smaller_ratio else 'ні'})",
                 f"Дистиляція знань: {'так' if self.cfg.distill.enabled else 'ні'}; "
                 f"нормування: {self.cfg.compress.norm_method}"]
        for r in records:
            if "cut_before" in r:
                keys = [k for k in self.selection if k in r["cut_before"]]
                change = ", ".join(f"{k} {r['cut_before'][k]:.3f}→{r['cut_after'][k]:.3f}" for k in keys)
                notes.append(f"it{r['iteration']}: норма вирізаних ваг перед вирізанням "
                             f"{r['norm_ratio']:.3f} від початкової; одразу після вирізання {change}")
        return write_report(records, self.cfg.model.type, self.cfg.model.labels, self.run_dir,
                            f"Стиснення: {self.cfg.name}", final, notes)

    # ------------------------------------------------------------------ кроки 1–3
    def train_baseline(self) -> tuple[nn.Module, dict]:
        cfg = self.cfg
        model = self.trainer.prepare(create_model(cfg.model, self.data.channels))
        optimizer = self.trainer.make_optimizer(model, cfg.train.lr)
        goal = (lambda m: meets(m, cfg.train.target)) if cfg.train.target else None
        monitor = PhaseMonitor(self.selection, goal, cfg.train.patience, cfg.train.tolerance, cfg.train.min_epochs)
        freeze = cfg.train.freeze_transferred_epochs if cfg.model.pretrained else 0
        self.log.info("Ітерація 0: transfer learning і донавчання до достатніх показників %s", cfg.train.target)
        result = self.trainer.train_phase(model, optimizer, phase="it0:base", epochs=cfg.train.max_epochs,
                                          schedule=constant(cfg.train.dropout), monitor=monitor,
                                          freeze_epochs=freeze, show=self.show)
        if goal and not result.goal_reached:
            self.log.warning("Достатніх показників %s не досягнуто за %d епох — стиснення продовжиться "
                             "відносно досягнутого рівня", cfg.train.target, result.epochs)
        to_goal = result.first_goal_epoch + 1 if result.first_goal_epoch is not None else None
        return model, {"epochs": result.epochs, "epochs_to_goal": to_goal, "seconds": result.seconds}

    # ------------------------------------------------------------------ крок 4
    def compress_step(self, model: nn.Module, iteration: int, ratio: float, lr: float,
                      teacher: nn.Module | None, reference: dict) -> StepResult:
        c, tag = self.cfg.compress, f"it{iteration}"
        work = copy.deepcopy(model)
        optimizer = self.trainer.make_optimizer(work, lr)
        epochs, seconds = 0, 0.0
        if c.spread_epochs > 0:
            r = self.trainer.train_phase(work, optimizer, phase=f"{tag}:spread", epochs=c.spread_epochs,
                                         schedule=spread_schedule(c), teacher=teacher, show=self.show)
            epochs, seconds = epochs + r.epochs, seconds + r.seconds
        plan = plan_pruning(work, ratio, c.importance, c.round_to, c.min_channels)
        self.log.info("%s: до вирізання позначено %d каналів (%.0f%% кожної групи)", tag,
                      plan.removed_channels(), 100 * ratio)
        regularizer = ChannelNormRegularizer(work, plan, c.norm_include_consumers)
        if c.squeeze_epochs > 0:
            r = self.trainer.train_phase(work, optimizer, phase=f"{tag}:squeeze", epochs=c.squeeze_epochs,
                                         schedule=squeeze_schedule(c), regularizer=regularizer, teacher=teacher,
                                         show=self.show)
            epochs, seconds = epochs + r.epochs, seconds + r.seconds
        norm_ratio = regularizer.mean_ratio()
        before = self.trainer.evaluate(work, "val").metrics
        pruned = self.trainer.prepare(apply_plan(work, plan))
        after = self.trainer.evaluate(pruned, "val").metrics
        self.log.info("%s: вирізано → ширини %s; залишкова норма %.3f; %s", tag, pruned.widths["enc1.0"], norm_ratio,
                      ", ".join(f"{k}: {before[k]:.4f} → {after[k]:.4f}" for k in self.selection))
        new_lr = max(lr * c.lr_scale_after_cut, c.min_lr)
        optimizer = self.trainer.make_optimizer(pruned, new_lr)
        goal = acceptance_goal(c.acceptance.minimum, c.acceptance.max_drop, reference)
        monitor = PhaseMonitor(self.selection, goal, c.stabilize_patience, self.cfg.train.tolerance,
                               c.stabilize_min_epochs)
        r = self.trainer.train_phase(pruned, optimizer, phase=f"{tag}:stabilize", epochs=c.stabilize_max_epochs,
                                     schedule=constant(0.0), teacher=teacher, monitor=monitor, show=self.show)
        to_goal = epochs + r.first_goal_epoch + 1 if r.first_goal_epoch is not None else None
        return StepResult(pruned, monitor.goal_reached, r.best_metrics or {}, before, after, norm_ratio,
                          epochs + r.epochs, to_goal, seconds + r.seconds, new_lr, plan)

    # ------------------------------------------------------------------ увесь алгоритм
    def run(self, baseline_path: str | Path | None = None, max_iterations: int | None = None) -> Path:
        cfg, c = self.cfg, self.cfg.compress
        if "baseline_checkpoint" in self.state:
            self.log.info("Відновлення запуску з %s", self.state_path)
            baseline = self.trainer.prepare(load_checkpoint(self.state["baseline_checkpoint"], self.device)[0])
            current = self.trainer.prepare(load_checkpoint(self.state["current_checkpoint"], self.device)[0])
            self.trainer.edge_ids = self.state.get("edge_ids", [])
            self.trainer.global_epoch = self.state.get("global_epoch", 0)
        else:
            if baseline_path:
                baseline, payload = load_checkpoint(baseline_path, self.device)
                baseline = self.trainer.prepare(baseline)
                if payload["arch"]["type"] != cfg.model.type:
                    raise ValueError("Тип моделі в чекпойнті не збігається з конфігурацією")
                info = {"epochs": 0, "epochs_to_goal": None, "seconds": 0.0}
                self.log.info("Перша ітерація взята з %s", baseline_path)
            else:
                baseline, info = self.train_baseline()
            # «крайні випадки» визначаються на тесті першою ітерацією
            val_res = self.trainer.evaluate(baseline, "val")
            test_res = self.trainer.evaluate(baseline, "test", thresholds=val_res.thresholds or None)
            self.trainer.edge_ids = self.trainer.pick_edge_cases(test_res, cfg.eval.edge_cases)
            record = self._record(baseline, 0, "baseline", ratio=0.0, epochs=info["epochs"],
                                  epochs_to_goal=info["epochs_to_goal"], seconds=info["seconds"])
            current = baseline
            self.state.update({"baseline_checkpoint": record["checkpoint"], "current_checkpoint": record["checkpoint"],
                               "reference": record["val"], "edge_ids": self.trainer.edge_ids, "iteration": 1,
                               "ratio": c.prune_ratio, "lr": cfg.train.lr, "final_iteration": 0})
            self._save_state()
            self._report()

        teacher = None
        if cfg.distill.enabled:
            teacher = copy.deepcopy(baseline).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            self.log.info("Дистиляція знань: вчитель — перша ітерація")
        reference = self.state["reference"]
        limit = c.max_iterations if max_iterations is None else max_iterations

        while self.state["iteration"] <= limit and not self.state.get("finished"):
            it, ratio = self.state["iteration"], self.state["ratio"]
            if plan_pruning(current, ratio, c.importance, c.round_to, c.min_channels).is_empty:
                self.log.info("Досягнуто мінімальної ширини (%d каналів) — зупинка", c.min_channels)
                break
            self.log.info("=== Ітерація %d: вирізаємо %.1f%% нейронів (поточна основа x%d) ===",
                          it, 100 * ratio, current.widths["enc1.0"])
            step = self.compress_step(current, it, ratio, self.state["lr"], teacher, reference)
            extra = {"cut_before": step.before_cut, "cut_after": step.after_cut, "norm_ratio": step.norm_ratio}
            status = "accepted" if step.accepted else "rejected"
            record = self._record(step.model, it, status, ratio=ratio, epochs=step.epochs,
                                  epochs_to_goal=step.epochs_to_goal, seconds=step.seconds, extra=extra)
            self.state["iteration"] = it + 1
            if step.accepted:
                current = step.model
                self.state.update({"current_checkpoint": record["checkpoint"], "lr": step.lr, "final_iteration": it})
                self.log.info("Ітерація %d прийнята: %s параметрів, CPU %.3f с", it,
                              f"{record['params']:,}", record["inference_s"])
            else:
                self.log.warning("Ітерація %d провалила критерії %s (%s)", it, cfg.compress.acceptance.minimum,
                                 ", ".join(f"{k} {step.val.get(k, float('nan')):.4f}" for k in self.show))
                if c.retry_smaller_ratio and ratio / 2 >= c.min_prune_ratio - 1e-9:
                    self.state["ratio"] = ratio / 2
                    self.log.info("Повтор з попередньої мережі з меншою часткою вирізання: %.1f%%", 50 * ratio)
                else:
                    self.state["finished"] = True
                    self.log.info("Мережа почала фейлити — результат: мережа з ітерації %d",
                                  self.state["final_iteration"])
            self._save_state()
            self._report()

        final_record = next(r for r in self.state["records"] if r["iteration"] == self.state["final_iteration"])
        final_path = self.ckpt_dir / "final.pt"
        shutil.copyfile(final_record["checkpoint"], final_path)
        if max_iterations is None:  # `train` (лише ітерація 0) лишає запуск відкритим для `compress --resume`
            self.state["finished"] = True
        self._save_state()
        report = self._report()
        self.log.info("Готово. Фінальна модель: %s (ітерація %d, %s параметрів); звіт: %s; загальний час %s",
                      final_path, self.state["final_iteration"], f"{final_record['params']:,}", report,
                      format_duration(self.state.get("total_seconds", 0.0)))
        return final_path
