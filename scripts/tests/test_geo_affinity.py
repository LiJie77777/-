"""Tests for the numpy reference in scripts/e2o_patches/geo_affinity.py
(run: python scripts/tests/test_geo_affinity.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2o_patches"))
import geo_affinity as G  # noqa: E402


def make(rng, B=2, H=6, W=7, C=16, beta=1.2, gamma=3.0):
    f = rng.normal(size=(B, H * W, C))
    f /= np.linalg.norm(f, axis=-1, keepdims=True)
    S = np.einsum("bmc,bnc->bmn", f, f)
    sim = (S - S.mean() * beta) * gamma  # E2O uses one global mean
    z = rng.random((B, H, W))
    return sim, z, (H, W)


def test_zero_weights_reproduce_e2o():
    sim, z, hw = make(np.random.default_rng(0))
    ref = np.where(sim > 0, sim, -np.inf)
    for mode in ("soft", "hard"):
        out = G.geo_adjust_similarity_np(sim, z, hw, mode=mode, lam_z=0.0, lam_s=0.0)
        assert np.array_equal(np.isinf(out), np.isinf(ref))
        assert np.allclose(out[np.isfinite(out)], ref[np.isfinite(ref)])


def test_diagonal_untouched_and_kept():
    sim, z, hw = make(np.random.default_rng(1))
    out = G.geo_adjust_similarity_np(sim, z, hw, mode="hard", lam_z=1.0, lam_s=1.0)
    d = np.einsum("bii->bi", out)
    assert np.allclose(d, np.einsum("bii->bi", sim))


def test_soft_equals_multiplicative_decay():
    # softmax(S~ - pen) == softmax(S~) * exp(-pen), renormalised (DFormerv2 form A * beta^G)
    sim, z, hw = make(np.random.default_rng(2))
    base = np.where(sim > 0, sim, -np.inf)
    out = G.geo_adjust_similarity_np(sim, z, hw, mode="soft", lam_z=0.5, lam_s=0.3)
    with np.errstate(invalid="ignore"):
        pen = np.where(np.isfinite(base), base - out, 0.0)
    A = G.softmax_np(base)
    dec = A * np.exp(-pen)
    dec /= dec.sum(-1, keepdims=True)
    assert np.allclose(G.softmax_np(out), dec)


def test_invariant_to_affine_depth():
    # DA predicts disparity only up to scale and shift; the term must not depend on them
    sim, z, hw = make(np.random.default_rng(3))
    a = G.geo_adjust_similarity_np(sim, z, hw, lam_z=0.5)
    b = G.geo_adjust_similarity_np(sim, 7.0 * z + 3.0, hw, lam_z=0.5)
    assert np.allclose(a[np.isfinite(a)], b[np.isfinite(b)])


def test_depth_term_downweights_cross_layer_pairs():
    rng = np.random.default_rng(4)
    sim, _, hw = make(rng)
    z = np.zeros((2,) + hw)
    z[:, :, : hw[1] // 2] = 1.0  # two depth layers
    out = G.geo_adjust_similarity_np(sim, z, hw, mode="soft", lam_z=1.0)
    layer = z.reshape(2, -1)
    cross = layer[:, :, None] != layer[:, None, :]
    fin = np.isfinite(out)
    delta = sim - out
    assert np.all(delta[fin & cross] > 0) and np.allclose(delta[fin & ~cross], 0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
