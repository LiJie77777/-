"""Offline go/no-go checks for the depth cards G1 (figure-ground / background channel) and
G3 (depth-guided boundary refinement) in docs/geometry_fusion_research.md.

Needs only numpy + Pillow, the GT masks and a depth cache (one <stem>.npy per image,
relative inverse depth as written by scripts/dump_depth_cache.py: larger = nearer).
No inference and no tuning: GT is used only for scoring, every rule below is fixed.

Per image, then mean and 95% bootstrap CI over images:
  F1 order    : 1 if median disparity of GT foreground > that of GT background
  F1 auroc    : AUROC of disparity as a per-pixel foreground score
  F2 iou_otsu : class-agnostic fg IoU of the Otsu-thresholded disparity mask (near = fg)
     iou_allfg: fg IoU of "everything is foreground" = what E2O outputs today, i.e. its
                P2 class-agnostic fg IoU (it has no background output)
  F3 ring     : (median disparity in GT fg - median in a ring of width --ring around it)
                / MAD of disparity over the image; > 0 means the object stands out as nearer
  B1 bF       : (only with --img_dir) boundary F-score vs GT semantic boundaries, 2 px
                tolerance, of the top-k gradient pixels of depth vs of luminance, k = number
                of GT boundary pixels (equal edge density)
  A1 pair     : random pixel pairs; positive = same GT label (background is its own label).
                AUROC of -|disparity difference| and of -L1 spatial distance, and of the
                depth score within spatial-distance tertiles (does depth separate same- vs
                different-region pairs at a fixed distance?). Necessary, not sufficient, for
                card G2: the real test is whether depth adds to DINOv2 similarity.
Controls through the same pipeline: center prior and shuffled depth (the depth map of the
next image in the list), both area-matched for F2; luminance edges for B1.

Usage:
    python scripts/prevalidate_depth_figure_ground.py --mask_dir datasets/MAS3K/mask \
        --depth_dir cache/depth_vitb/MAS3K --bg_codes 0 --void_codes 255 \
        [--img_dir datasets/MAS3K/img] [--max_images 500] --out mas3k_fg.json
Background / void codes differ per dataset: check them with
scripts/dedup_and_mask_consistency.py --census_only first.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def resize_float(a, hw):
    im = Image.fromarray(np.asarray(a, np.float32), mode="F")
    return np.asarray(im.resize((hw[1], hw[0]), Image.BILINEAR), np.float64)


def resize_label(a, hw):
    return np.asarray(Image.fromarray(a.astype(np.int32), mode="I").resize((hw[1], hw[0]), Image.NEAREST))


def target_hw(hw, max_side):
    s = min(1.0, max_side / max(hw))
    return max(1, int(round(hw[0] * s))), max(1, int(round(hw[1] * s)))


def auroc(score, pos):
    """Mann-Whitney AUROC with average ranks for ties; nan if one class is empty."""
    n1 = int(pos.sum())
    n0 = pos.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    rank = ((cum - cnt + 1 + cum) / 2.0)[inv]
    return float((rank[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def otsu(x, bins=256):
    hist, edges = np.histogram(x, bins=bins)
    p = hist / max(hist.sum(), 1)
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    m = np.cumsum(p * c)
    w1 = 1 - w0
    var = (m[-1] * w0 - m) ** 2 / np.maximum(w0 * w1, 1e-12)
    return float(edges[int(np.argmax(var[:-1])) + 1])


def dilate(m, r):
    out = m.copy()
    for _ in range(r):
        p = np.pad(out, 1)
        out = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def semantic_boundary(lab, valid):
    b = np.zeros(lab.shape, bool)
    for dy, dx in ((0, 1), (1, 0)):
        a, c = lab[: lab.shape[0] - dy, : lab.shape[1] - dx], lab[dy:, dx:]
        va, vc = valid[: lab.shape[0] - dy, : lab.shape[1] - dx], valid[dy:, dx:]
        d = (a != c) & va & vc
        b[: lab.shape[0] - dy, : lab.shape[1] - dx] |= d
        b[dy:, dx:] |= d
    return b


def boundary_f(score, gt_b, valid, tol=2):
    k = int(gt_b.sum())
    if k == 0:
        return float("nan")
    s = np.where(valid, score, -np.inf).ravel()
    pred = np.zeros(s.size, bool)
    pred[np.argpartition(-s, k - 1)[:k]] = True
    pred = pred.reshape(score.shape)
    prec = (pred & dilate(gt_b, tol)).sum() / max(pred.sum(), 1)
    rec = (gt_b & dilate(pred, tol)).sum() / k
    return float(2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0


def grad_mag(a):
    gy, gx = np.gradient(a)
    return np.hypot(gx, gy)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def area_matched(score, valid, frac):
    s = np.where(valid, score, -np.inf).ravel()
    k = max(1, int(round(frac * valid.sum())))
    m = np.zeros(s.size, bool)
    m[np.argpartition(-s, k - 1)[:k]] = True
    return m.reshape(score.shape) & valid


def pair_aurocs(depth, lab, valid, rng, n_pairs=4000):
    idx = np.flatnonzero(valid.ravel())
    if idx.size < 2:
        return {}
    a, b = rng.choice(idx, n_pairs), rng.choice(idx, n_pairs)
    W = lab.shape[1]
    same = lab.ravel()[a] == lab.ravel()[b]
    dz = -np.abs(depth.ravel()[a] - depth.ravel()[b])
    dp = -(np.abs(a // W - b // W) + np.abs(a % W - b % W)).astype(np.float64)
    r = {"pair_auroc_depth": auroc(dz, same), "pair_auroc_spatial": auroc(dp, same)}
    edges = np.quantile(dp, [1 / 3, 2 / 3])
    bins = np.digitize(dp, edges)
    within = [auroc(dz[bins == k], same[bins == k]) for k in range(3)]
    within = [w for w in within if np.isfinite(w)]
    r["pair_auroc_depth_given_spatial"] = float(np.mean(within)) if within else float("nan")
    return r


def analyse_image(depth, lab, fg, bg, valid, shuf, lum, ring_w, rng):
    H, W = lab.shape
    yy, xx = np.mgrid[0:H, 0:W]
    center = -(((yy - (H - 1) / 2) / H) ** 2 + ((xx - (W - 1) / 2) / W) ** 2)
    r = {}
    pos = fg[valid]
    if fg.any() and bg.any():
        r["order"] = float(np.median(depth[fg]) > np.median(depth[bg]))
        r["auroc_depth"] = auroc(depth[valid], pos)
        r["auroc_center"] = auroc(center[valid], pos)
        r["auroc_shuf"] = auroc(shuf[valid], pos)
        th = otsu(depth[valid])
        m = (depth > th) & valid
        frac = m.sum() / valid.sum()
        r["iou_otsu_depth"] = iou(m, fg)
        r["iou_otsu_center_area_matched"] = iou(area_matched(center, valid, frac), fg)
        r["iou_otsu_shuf"] = iou((shuf > otsu(shuf[valid])) & valid, fg)
        r["iou_allfg"] = iou(valid, fg)
        ring = dilate(fg, ring_w) & valid & ~fg
        if ring.any():
            mad = np.median(np.abs(depth[valid] - np.median(depth[valid]))) + 1e-6
            r["ring_contrast"] = float((np.median(depth[fg]) - np.median(depth[ring])) / mad)
    r.update(pair_aurocs(depth, lab, valid, rng))
    if lum is not None:
        gt_b = semantic_boundary(lab, valid)
        r["bF_depth"] = boundary_f(grad_mag(depth), gt_b, valid)
        r["bF_lum"] = boundary_f(grad_mag(lum), gt_b, valid)
    return r


def bootstrap_ci(x, n=1000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--depth_dir", required=True)
    ap.add_argument("--img_dir", default=None, help="enables the boundary check B1")
    ap.add_argument("--bg_codes", type=int, nargs="*", default=[])
    ap.add_argument("--void_codes", type=int, nargs="*", default=[])
    ap.add_argument("--max_images", type=int, default=0, help="0 = all; else a fixed-seed subsample")
    ap.add_argument("--max_side", type=int, default=320)
    ap.add_argument("--ring", type=int, default=8, help="ring width in pixels at --max_side scale")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(args.mask_dir) if f.endswith(".png"))
    stems = [s for s in stems if os.path.exists(os.path.join(args.depth_dir, s + ".npy"))]
    if args.max_images and len(stems) > args.max_images:
        rng = np.random.default_rng(args.seed)
        stems = sorted(rng.choice(stems, args.max_images, replace=False).tolist())
    if len(stems) < 2:
        raise SystemExit("need at least two images with both a mask and a cached depth map")
    img_files = {}
    if args.img_dir:
        img_files = {os.path.splitext(f)[0]: f for f in os.listdir(args.img_dir) if f.lower().endswith(IMG_EXT)}

    rng = np.random.default_rng(args.seed)
    rows = []
    skipped = {"no_fg": 0, "no_bg": 0}
    for i, s in enumerate(stems):
        raw = np.array(Image.open(os.path.join(args.mask_dir, s + ".png")))
        if raw.ndim != 2:
            raise SystemExit(f"{s}.png is not a single-channel label mask; convert colour masks first")
        hw = target_hw(raw.shape, args.max_side)
        lab = resize_label(raw.astype(np.int64), hw).astype(np.int64)
        void = np.isin(lab, args.void_codes)
        bg = np.isin(lab, args.bg_codes) & ~void
        fg = ~void & ~bg
        valid = ~void
        skipped["no_fg"] += int(not fg.any())
        skipped["no_bg"] += int(not bg.any())
        depth = resize_float(np.load(os.path.join(args.depth_dir, s + ".npy")).astype(np.float32), hw)
        nxt = stems[(i + 1) % len(stems)]
        shuf = resize_float(np.load(os.path.join(args.depth_dir, nxt + ".npy")).astype(np.float32), hw)
        lum = None
        if args.img_dir and s in img_files:
            rgb = np.asarray(Image.open(os.path.join(args.img_dir, img_files[s])).convert("RGB"), np.float32)
            lum = resize_float(rgb @ np.array([0.299, 0.587, 0.114], np.float32), hw)
        r = analyse_image(depth, lab, fg, bg, valid, shuf, lum, args.ring, rng)
        r["image"] = s
        rows.append(r)

    keys = ["order", "auroc_depth", "auroc_center", "auroc_shuf", "iou_otsu_depth",
            "iou_otsu_center_area_matched", "iou_otsu_shuf", "iou_allfg", "ring_contrast",
            "bF_depth", "bF_lum", "pair_auroc_depth", "pair_auroc_spatial",
            "pair_auroc_depth_given_spatial"]
    summary = {"n_images": len(rows), "skipped_for_F1_F3": skipped}
    for k in keys:
        mean, lo, hi, n = bootstrap_ci([r.get(k, np.nan) for r in rows], seed=args.seed)
        summary[k] = {"mean": mean, "ci95": [lo, hi], "n": n}
    paired = {
        "auroc_depth_minus_best_control": [r["auroc_depth"] - max(r["auroc_center"], r["auroc_shuf"])
                                           for r in rows if "auroc_depth" in r],
        "iou_otsu_depth_minus_allfg": [r["iou_otsu_depth"] - r["iou_allfg"] for r in rows if "iou_otsu_depth" in r],
        "iou_otsu_depth_minus_center": [r["iou_otsu_depth"] - r["iou_otsu_center_area_matched"]
                                        for r in rows if "iou_otsu_depth" in r],
        "bF_depth_minus_lum": [r["bF_depth"] - r["bF_lum"] for r in rows if "bF_depth" in r],
    }
    for k, v in paired.items():
        mean, lo, hi, n = bootstrap_ci(v, seed=args.seed)
        summary[k] = {"mean": mean, "ci95": [lo, hi], "n": n}

    # Pre-registered rules from the report (card G1 / G3); printed, never tuned.
    a = summary["auroc_depth"]["mean"]
    d = summary["auroc_depth_minus_best_control"]
    g = summary["iou_otsu_depth_minus_allfg"]
    summary["G1_pass"] = bool(a >= 0.70 and d["mean"] >= 0.05 and d["ci95"][0] > 0 and g["ci95"][0] > 0.05)
    summary["G1_abandon"] = bool(a < 0.60 or d["mean"] <= 0)
    b = summary["bF_depth_minus_lum"]
    summary["G3_pass"] = bool(b["n"] > 0 and b["mean"] >= 0.02 and b["ci95"][0] > 0)
    p = summary["pair_auroc_depth_given_spatial"]
    summary["G2_pass"] = bool(p["mean"] >= 0.60 and p["ci95"][0] > 0.55)

    for k, v in summary.items():
        if isinstance(v, dict) and "mean" in v:
            print(f"{k:34s} {v['mean']:8.4f}  CI95 [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]  n={v['n']}")
        else:
            print(f"{k:34s} {v}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "per_image": rows}, f, indent=1)


if __name__ == "__main__":
    main()
