"""Конфігурація: перевірка ключів, значень і перевизначень."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuromed.config import ConfigError, config_from_dict, config_to_dict, load_config, save_config

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_load(path):
    cfg = load_config(path)
    assert cfg.name


def test_defaults():
    cfg = load_config(None)
    assert cfg.compress.norm_method == "both"
    assert cfg.compress.norm_max == 0.95
    assert cfg.run_dir == Path("runs") / cfg.name


def test_unknown_key_is_an_error():
    with pytest.raises(ConfigError, match="lrr"):
        config_from_dict({"train": {"lrr": 1e-3}})


def test_invalid_choice_is_an_error():
    with pytest.raises(ConfigError):
        config_from_dict({"compress": {"norm_method": "magic"}})


def test_input_size_must_divide_by_16():
    with pytest.raises(ConfigError):
        config_from_dict({"data": {"input_size": 100}})


def test_overrides_and_float_coercion():
    cfg = load_config(None, ["train.lr=1e-4", "compress.acceptance.minimum={det_recall: 0.9}", "model.base=32"])
    assert isinstance(cfg.train.lr, float) and cfg.train.lr == pytest.approx(1e-4)
    assert cfg.compress.acceptance.minimum == {"det_recall": 0.9}
    assert cfg.model.base == 32


def test_bad_override_syntax():
    with pytest.raises(ConfigError):
        load_config(None, ["train.lr"])


def test_save_and_reload_roundtrip(tmp_path):
    cfg = load_config(None, ["name=roundtrip", "model.labels=[glaucoma, valid_image]", "model.type=classifier"])
    save_config(cfg, tmp_path / "config.yaml")
    assert config_to_dict(load_config(tmp_path / "config.yaml")) == config_to_dict(cfg)
