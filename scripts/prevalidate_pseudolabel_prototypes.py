"""Offline go/no-go check for Line B experiment B3 (cross-image visual prototypes).

Works on the existing GT-Uniform probe dump (AquaFree --probe-dump): one row per
(sliding window, GT class) region with keys
    feat (N,512)  cls_logits (N,C)  gt (N,)  text_pred (N,)  count (N,)  image (N,)
It asks: if class prototypes are built WITHOUT labels (from MLLM Objects and/or the
text head's own confident predictions) on one half of the images, how well do they
classify regions of the other half, compared with the text head (and with the GT-label
centroid upper bound, which the hand-off package reports as 82.60 / 80.94)?

Caveat: probe regions are pooled inside GT regions (perfect grouping), so every number
here is optimistic for the full pipeline. Use it only as a filter: if even this fails,
drop B3.

Usage:
    python scripts/prevalidate_pseudolabel_prototypes.py --probe probe.npz \
        --mllm_json configs/llm_reasoning_AquaOV255_qwen7b.json \
        --class_file configs/cls_AquaOV255.txt [--logit_scale 40]
"""
import argparse
import hashlib
import json
import re

import numpy as np

ALIASES = {"mask": "surgicalmask", "weedyseadragon": "seadragon"}  # verify against category.txt


def norm(s):
    k = re.sub(r"[^a-z]", "", s.lower())
    return ALIASES.get(k, k)


def load_probe(path):
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    import torch  # probe dumps saved with torch.save
    d = torch.load(path, map_location="cpu")
    return {k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v)) for k, v in d.items()}


def fold_of(name):
    return int(hashlib.md5(str(name).encode()).hexdigest(), 16) % 2


def softmax(x, axis=-1):
    x = x - x.max(axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis, keepdims=True)


def accuracy(pred, gt, w):
    hit = (pred == gt).astype(np.float64)
    cls = np.unique(gt)
    per_class = np.mean([hit[gt == c].mean() for c in cls])
    return 100 * (w @ hit) / w.sum(), 100 * hit.mean(), 100 * per_class


def build_prototypes(feat, labels, C):
    D = feat.shape[1]
    proto = np.zeros((C, D))
    has = np.zeros(C, bool)
    for c in np.unique(labels[labels >= 0]):
        m = labels == c
        v = feat[m].mean(0)
        proto[c] = v / (np.linalg.norm(v) + 1e-8)
        has[c] = True
    return proto, has


def tpa_fuse(logp_text, feat, proto, has, alpha=0.5, lam=5.0):
    """TPA-style late fusion (arXiv 2608.08290, Eq. 6-8): mean-centred cosine to prototypes."""
    f = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
    q = f @ proto.T
    q[:, ~has] = 0.0
    qbar = q[:, has].mean(1, keepdims=True) if has.any() else 0.0
    qt = np.where(has[None, :], q - qbar, 0.0)
    return logp_text + (1 - alpha) * lam * qt


def auto_lambda(logp_text, feat, proto, has, k=5):
    """Label-free scale matching: make the spread of the prototype term over the top-k
    text classes comparable to the spread of the text log-probabilities (median ratio)."""
    f = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
    q = f @ proto.T
    top = np.argsort(-logp_text, 1)[:, :k]
    lp = np.take_along_axis(logp_text, top, 1)
    qq = np.take_along_axis(np.where(has[None, :], q, np.nan), top, 1)
    spread_lp = np.nanmax(lp, 1) - np.nanmin(lp, 1)
    spread_q = np.nanmax(qq, 1) - np.nanmin(qq, 1)
    ok = np.isfinite(spread_q) & (spread_q > 1e-6)
    return float(np.median(spread_lp[ok]) / np.median(spread_q[ok])) if ok.any() else 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--mllm_json", required=True)
    ap.add_argument("--class_file", required=True)
    ap.add_argument("--logit_scale", type=float, default=40.0,
                    help="applied to cls_logits if they are raw cosine similarities")
    args = ap.parse_args()

    P = load_probe(args.probe)
    feat = P["feat"].astype(np.float64)
    logits = P["cls_logits"].astype(np.float64) * args.logit_scale
    gt = P["gt"].astype(np.int64)
    w = P["count"].astype(np.float64)
    images = np.array([str(x).split("/")[-1] for x in P["image"]])
    C = logits.shape[1]
    classes = [x.strip() for x in open(args.class_file, encoding="utf-8").read().split("\n") if x.strip()]
    cidx = {norm(c): i for i, c in enumerate(classes)}
    mllm = json.load(open(args.mllm_json, encoding="utf-8"))

    logp = np.log(softmax(logits) + 1e-12)
    text_pred = logits.argmax(1)
    prob_max = softmax(logits).max(1)

    # candidate sets per region (image-level Objects mapped to class ids)
    cand = []
    for im in images:
        objs = (mllm.get(im) or {}).get("Objects") or []
        cand.append(sorted({cidx[norm(o)] for o in objs if isinstance(o, str) and norm(o) in cidx}))
    # MLLM pseudo-label per region = candidate with highest text logit (if any)
    mllm_pl = np.array([c[int(np.argmax(logits[i, c]))] if c else -1 for i, c in enumerate(cand)])
    top5 = np.argsort(-logits, 1)[:, :5]
    agree = np.array([mllm_pl[i] >= 0 and mllm_pl[i] in top5[i] for i in range(len(gt))])
    conf_text = np.where(prob_max > 2.0 / C, text_pred, -1)  # TPA anchor rule tau = 2/C

    seeds = {
        "GT labels (upper bound)": gt,
        "MLLM pseudo-labels": mllm_pl,
        "MLLM ∩ text-top5": np.where(agree, mllm_pl, -1),
        "text confident (p>2/C), no MLLM": conf_text,
    }
    folds = np.array([fold_of(im) for im in images])
    print(f"regions={len(gt)} images={len(set(images))} classes={C}")
    a, r, pc = accuracy(text_pred, gt, w)
    print(f"{'text head':38s} area={a:6.2f} region={r:6.2f} class-mean={pc:6.2f}")
    for name, lab in seeds.items():
        valid = lab >= 0
        prec = 100 * np.mean(lab[valid] == gt[valid]) if valid.any() else float("nan")
        pred_proto = np.empty_like(gt)
        pred_fuse = np.empty_like(gt)
        pred_auto = np.empty_like(gt)
        lam_used = []
        coverage = []
        for f in (0, 1):
            tr, te = folds != f, folds == f
            proto, has = build_prototypes(feat[tr], np.where(tr, lab, -1)[tr], C)
            coverage.append(has.sum())
            fe = feat[te] / (np.linalg.norm(feat[te], axis=1, keepdims=True) + 1e-8)
            s = fe @ proto.T
            s[:, ~has] = -np.inf
            pred_proto[te] = s.argmax(1)
            pred_fuse[te] = tpa_fuse(logp[te], feat[te], proto, has).argmax(1)
            lam = auto_lambda(logp[tr], feat[tr], proto, has)  # computed on the other fold
            lam_used.append(lam)
            pred_auto[te] = tpa_fuse(logp[te], feat[te], proto, has, alpha=0.0, lam=lam).argmax(1)
        a1, r1, p1 = accuracy(pred_proto, gt, w)
        a2, r2, p2 = accuracy(pred_fuse, gt, w)
        print(f"{name:38s} seed-precision={prec:6.2f} seed-coverage={100 * valid.mean():5.1f}% "
              f"classes-with-proto={min(coverage)}/{C}")
        print(f"{'  -> prototype only':38s} area={a1:6.2f} region={r1:6.2f} class-mean={p1:6.2f}")
        print(f"{'  -> TPA fusion (a=0.5, lam=5)':38s} area={a2:6.2f} region={r2:6.2f} class-mean={p2:6.2f}")
        a3, r3, p3 = accuracy(pred_auto, gt, w)
        lam_s = "/".join(f"{x:.1f}" for x in lam_used)
        print(f"{'  -> fusion, lam scale-matched':38s} area={a3:6.2f} region={r3:6.2f} class-mean={p3:6.2f} (lam={lam_s})")


if __name__ == "__main__":
    main()
