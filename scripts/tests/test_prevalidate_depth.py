"""Synthetic tests for scripts/prevalidate_depth_figure_ground.py
(run: python scripts/tests/test_prevalidate_depth.py)."""
import json
import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import prevalidate_depth_figure_ground as P  # noqa: E402


def brute_auroc(score, pos):
    s1, s0 = score[pos], score[~pos]
    gt = (s1[:, None] > s0[None, :]).sum() + 0.5 * (s1[:, None] == s0[None, :]).sum()
    return gt / (len(s1) * len(s0))


def test_auroc_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        score = rng.integers(0, 5, 60).astype(float)  # many ties
        pos = rng.random(60) < 0.4
        assert abs(P.auroc(score, pos) - brute_auroc(score, pos)) < 1e-12


def test_otsu_splits_bimodal():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 0.1, 500), rng.normal(1, 0.1, 500)])
    assert 0.3 < P.otsu(x) < 0.7


def make_dataset(root, n, informative, rng, size=64):
    os.makedirs(os.path.join(root, "mask"))
    os.makedirs(os.path.join(root, "depth"))
    os.makedirs(os.path.join(root, "img"))
    for i in range(n):
        lab = np.zeros((size, size), np.uint8)  # 0 = background
        h, w = rng.integers(12, 24, 2)
        y, x = rng.integers(0, size - 24, 2)  # random position: center prior is uninformative
        lab[y:y + h, x:x + w] = 1 + rng.integers(3)
        lab[:2, :] = 255  # void strip
        if informative:
            depth = np.where(lab != 0, 1.0, 0.2) + rng.normal(0, 0.05, lab.shape)
        else:
            depth = rng.random(lab.shape)
        img = rng.integers(0, 255, (size, size, 3)).astype(np.uint8)  # luminance carries no edges
        stem = f"im{i:03d}"
        Image.fromarray(lab).save(os.path.join(root, "mask", stem + ".png"))
        np.save(os.path.join(root, "depth", stem + ".npy"), depth.astype(np.float16))
        Image.fromarray(img).save(os.path.join(root, "img", stem + ".jpg"))


def run(root):
    out = os.path.join(root, "out.json")
    argv = sys.argv
    sys.argv = ["x", "--mask_dir", os.path.join(root, "mask"), "--depth_dir", os.path.join(root, "depth"),
                "--img_dir", os.path.join(root, "img"), "--bg_codes", "0", "--void_codes", "255",
                "--out", out]
    try:
        P.main()
    finally:
        sys.argv = argv
    return json.load(open(out))["summary"]


def test_informative_depth_passes_and_noise_is_abandoned():
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as good, tempfile.TemporaryDirectory() as bad:
        make_dataset(good, 30, True, rng)
        make_dataset(bad, 30, False, rng)
        s = run(good)
        assert s["auroc_depth"]["mean"] > 0.95 and s["order"]["mean"] == 1.0
        assert s["auroc_center"]["mean"] < 0.85 and abs(s["auroc_shuf"]["mean"] - 0.5) < 0.2
        assert s["iou_otsu_depth"]["mean"] > 0.9 > s["iou_allfg"]["mean"]
        assert s["ring_contrast"]["mean"] > 3
        assert s["bF_depth"]["mean"] > s["bF_lum"]["mean"] + 0.3
        assert s["pair_auroc_depth_given_spatial"]["mean"] > 0.8
        assert s["G1_pass"] and not s["G1_abandon"] and s["G3_pass"] and s["G2_pass"]
        t = run(bad)
        assert abs(t["auroc_depth"]["mean"] - 0.5) < 0.1
        assert abs(t["pair_auroc_depth_given_spatial"]["mean"] - 0.5) < 0.1
        assert t["G1_abandon"] and not t["G1_pass"] and not t["G2_pass"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
