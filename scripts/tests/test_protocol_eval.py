"""Sanity tests for scripts/protocol_eval.py (run: python -m pytest scripts/tests -q,
or python scripts/tests/test_protocol_eval.py)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from protocol_eval import RunAccumulator, load, metrics  # noqa: E402

BG = 255


def mmseg_like_miou(preds, gts, C, ignore=255):
    """Reference implementation of MMSeg IoUMetric (mask applied first, nanmean)."""
    inter = np.zeros(C); union = np.zeros(C)
    for p, g in zip(preds, gts):
        m = g != ignore
        p, g = p[m], g[m]
        for c in range(C):
            inter[c] += np.sum((p == c) & (g == c))
            union[c] += np.sum((p == c) | (g == c))
    with np.errstate(invalid="ignore"):
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return 100 * np.nanmean(iou)


def make_single_object_images(n, C, rng, size=32):
    imgs = []
    for i in range(n):
        g = np.full((size, size), BG, np.int64)
        c = rng.integers(C)
        h, w = rng.integers(4, 12, 2)
        y, x = rng.integers(0, size - 12, 2)
        g[y:y + h, x:x + w] = c
        imgs.append((g, c))
    return imgs


def test_p0_matches_mmseg_reference():
    rng = np.random.default_rng(0)
    C = 7
    preds, gts = [], []
    acc = RunAccumulator(C)
    for _ in range(40):
        g = rng.integers(0, C, (16, 16))
        g[rng.random((16, 16)) < 0.3] = BG
        p = rng.integers(0, C, (16, 16))
        preds.append(p); gts.append(g)
        acc.add(p, g, bg_codes=(BG,), void_codes=())
    with tempfile.TemporaryDirectory() as t:
        acc.save(os.path.join(t, "r.npz"))
        m = metrics(load(os.path.join(t, "r.npz")))
    ref = mmseg_like_miou(preds, gts, C)
    assert abs(m["P0_mIoU"] - ref) < 1e-9, (m["P0_mIoU"], ref)


def test_whole_image_fill_is_perfect_only_under_official_protocol():
    rng = np.random.default_rng(1)
    C = 10
    acc = RunAccumulator(C)
    fg_frac = []
    for g, c in make_single_object_images(50, C, rng):
        pred = np.full_like(g, c)  # oracle-name whole-image fill
        acc.add(pred, g, bg_codes=(BG,), void_codes=())
        fg_frac.append(np.mean(g != BG))
    m = metrics({**acc.arrays(), "num_classes": C})
    assert abs(m["P0_mIoU"] - 100) < 1e-9
    assert m["P1b_fg_mIoU"] < 50
    assert abs(m["P2_recog_acc"] - 100) < 1e-9
    # class-gated IoU of a full-image fill equals the foreground fraction per image
    assert abs(m["P2_gated_IoU"] - 100 * np.mean(fg_frac)) < 1e-6


def test_perfect_masks_wrong_class_half_the_time():
    rng = np.random.default_rng(2)
    C = 10
    acc = RunAccumulator(C)
    for i, (g, c) in enumerate(make_single_object_images(100, C, rng)):
        pred = np.full_like(g, C)  # predicts background explicitly
        cls = c if i % 2 == 0 else (c + 1) % C
        pred[g != BG] = cls
        acc.add(pred, g, bg_codes=(BG,), void_codes=())
    m = metrics({**acc.arrays(), "num_classes": C})
    assert abs(m["P2_recog_acc"] - 50) < 1e-9
    assert abs(m["P2_fg_IoU"] - 100) < 1e-9
    assert abs(m["P2_gated_IoU"] - 50) < 1e-9
    assert abs(m["P1_bg_IoU"] - 100) < 1e-9


if __name__ == "__main__":
    test_p0_matches_mmseg_reference()
    test_whole_image_fill_is_perfect_only_under_official_protocol()
    test_perfect_masks_wrong_class_half_the_time()
    print("all tests passed")
