"""Попередня обробка, аугментація, розбиття, буфер складних зразків."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from neuromed.config import AugmentConfig, PreprocessConfig
from neuromed.data.augment import BatchAugmenter
from neuromed.data.preprocessing import fundus_square, model_input, normalize, read_rgb, write_image
from neuromed.data.records import load_records, make_chunks, split_records
from neuromed.data.sampling import HardExampleBuffer


def _fundus(h=120, w=180):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (w // 2, h // 2), h // 2 - 5, (180, 90, 40), -1)
    return img


def test_fundus_square_crops_to_circle_and_maps_back():
    img = _fundus()
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[50:60, 100:110] = 1
    square, mask_sq, info = fundus_square(img, mask)
    assert square.shape[0] == square.shape[1] == info.side
    assert info.side < img.shape[1]  # чорні поля обрізано
    ys, xs = np.nonzero(mask_sq)
    x, y = info.to_source(float(xs.min()), float(ys.min()))
    assert (x, y) == (100.0, 50.0)


@pytest.mark.parametrize("grayscale, channels", [(True, 1), (False, 3)])
def test_model_input_shape(grayscale, channels):
    out = model_input(_fundus(), PreprocessConfig(grayscale=grayscale), 64, prepared_size=128)
    assert out.shape == (channels, 64, 64) and out.dtype == np.uint8


def test_normalize_range():
    x = normalize(torch.tensor([0, 255], dtype=torch.uint8))
    assert x.tolist() == [-1.0, 1.0]


def test_image_io_with_cyrillic_path(tmp_path):
    img = _fundus()
    path = tmp_path / "знімок ока.png"
    write_image(path, img)
    np.testing.assert_array_equal(read_rgb(path), img)


def test_augmentation_keeps_mask_binary_and_range():
    aug = BatchAugmenter(AugmentConfig(artifact_prob=1.0, blur_prob=1.0))
    x = torch.rand(4, 1, 64, 64)
    mask = torch.zeros(4, 1, 64, 64)
    mask[:, :, 20:40, 20:40] = 1
    out, m = aug(x, mask)
    assert out.shape == x.shape and m.shape == mask.shape
    assert set(torch.unique(m).tolist()) <= {0.0, 1.0}
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0
    assert m.sum() > 0


def test_augmentation_disabled_is_identity():
    x = torch.rand(2, 3, 32, 32)
    out, m = BatchAugmenter(AugmentConfig(enabled=False))(x, None)
    assert out is x and m is None


def test_splits_are_grouped_and_disjoint(synthetic_dir):
    records = load_records([synthetic_dir], "classifier", ["glaucoma", "valid_image"])
    train, val, test = split_records(records, 0.2, 0.2, seed=0, primary_label="glaucoma")
    assert len(set(train) | set(val) | set(test)) == len(records)
    groups = [{records[i].group for i in part} for part in (train, val, test)]
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])
    chunks = make_chunks(records, train, 3, seed=0, primary_label="glaucoma")
    assert sorted(np.concatenate(chunks).tolist()) == sorted(train.tolist())


def test_hard_example_buffer():
    buf = HardExampleBuffer(max_size=4)
    buf.update(np.arange(7), np.array([0.1, 5.0, 0.2, 4.0, 0.3, 3.0, 2.0]))
    assert len(buf) == 4
    assert set(buf.losses) == {1, 3, 5, 6}
    buf.update([1], [100.0])
    samples = buf.sample(1000, np.random.default_rng(0))
    assert (samples == 1).mean() > 0.8
