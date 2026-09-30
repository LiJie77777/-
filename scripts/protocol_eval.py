"""Offline multi-protocol evaluator for UOVSBench-style runs.

One inference run -> one NPZ of per-image area statistics -> every protocol below is
computed offline (no re-inference). Protocols (see docs/uovs_research_plan.md, Line C):

  P0  official      : MMSeg IoUMetric semantics. Background/void GT pixels are dropped
                      before any statistic (ignore_index), C foreground classes.
  P1a bg-scored     : background is a real class (index C); mIoU over C+1 classes
                      (VOC21 / COCO-Object style).
  P1b bg-scored fg  : same confusion as P1a, but the mean is taken over the C
                      foreground classes only (background pixels still produce FP).
  P2  class-gated   : per image, main GT class c* (largest valid area) vs predicted main
                      class (largest predicted foreground area over all non-void pixels).
                      Reports recognition accuracy, class-agnostic foreground IoU and
                      class-gated IoU (IoU of c* if recognized, else 0), mean over images.
                      Inspired by the class-gated metrics of OVCamo (ECCV 2024).

Prediction convention: labels 0..C-1 are foreground classes; label C means the method
explicitly predicted "background". Methods without a background concept never output C.

Mask convention: pass the dataset's background codes and void codes explicitly
(e.g. AquaOV255 after MMSeg's uint8 cast: background 65535 -> 255). Verify the codes of
every dataset on the server before use (pre-verification item C1-3).

Usage:
    # 1) inside the inference loop, per image:
    acc = RunAccumulator(num_classes=C)
    acc.add(pred_map, gt_map, bg_codes=(255,), void_codes=())
    acc.save("run.npz", meta={...})
    # 2) offline:
    python scripts/protocol_eval.py runA.npz runB.npz --bootstrap 1000
"""
import argparse
import json

import numpy as np


class RunAccumulator:
    def __init__(self, num_classes):
        self.C = int(num_classes)
        self.rows = {k: [] for k in ("inter_off", "gt_off", "pred_off",
                                      "inter_bg", "gt_bg", "pred_bg")}
        self.image_ids = []

    def add(self, pred, gt, bg_codes=(), void_codes=(255,), image_id=None):
        C = self.C
        pred = np.asarray(pred).astype(np.int64).ravel()
        gt = np.asarray(gt).astype(np.int64).ravel()
        if pred.shape != gt.shape:
            raise ValueError("pred and gt must have the same shape")
        if pred.min() < 0 or pred.max() > C:
            raise ValueError("pred labels must be in [0, C]")
        void = np.isin(gt, list(void_codes)) if len(void_codes) else np.zeros_like(gt, bool)
        bg = np.isin(gt, list(bg_codes)) if len(bg_codes) else np.zeros_like(gt, bool)
        fg = (~void) & (~bg) & (gt >= 0) & (gt < C)
        unknown = (~void) & (~bg) & (~fg)
        if unknown.any():
            bad = np.unique(gt[unknown])[:10]
            raise ValueError(f"GT codes {bad} are neither classes, background nor void")

        # P0: official. Only foreground GT pixels count. A pixel predicted as C
        # (background) is a false negative for its GT class and FP for nobody.
        g, p = gt[fg], pred[fg]
        gt_off = np.bincount(g, minlength=C)[:C]
        pred_off = np.bincount(p, minlength=C + 1)[:C]
        inter_off = np.bincount(g[g == p], minlength=C)[:C]

        # P1: background is class C.
        valid = fg | bg
        g1 = np.where(bg, C, gt)[valid]
        p1 = pred[valid]
        gt_bg = np.bincount(g1, minlength=C + 1)
        pred_bg = np.bincount(p1, minlength=C + 1)
        inter_bg = np.bincount(g1[g1 == p1], minlength=C + 1)

        for k, v in (("inter_off", inter_off), ("gt_off", gt_off), ("pred_off", pred_off),
                     ("inter_bg", inter_bg), ("gt_bg", gt_bg), ("pred_bg", pred_bg)):
            self.rows[k].append(v.astype(np.int64))
        self.image_ids.append(image_id if image_id is not None else str(len(self.image_ids)))

    def arrays(self):
        out = {k: np.stack(v) for k, v in self.rows.items()}
        out["image_ids"] = np.array(self.image_ids)
        out["num_classes"] = np.array(self.C)
        return out

    def save(self, path, meta=None):
        np.savez_compressed(path, **self.arrays(), meta=json.dumps(meta or {}))


def load(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["num_classes"] = int(d["num_classes"])
    d["meta"] = json.loads(str(d["meta"])) if "meta" in d else {}
    return d


def _iou(inter, gt, pred):
    union = gt + pred - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return iou


def metrics(d, weights=None):
    """weights: per-image multiplicities (bootstrap); None = all ones."""
    C = d["num_classes"]
    w = np.ones(len(d["gt_off"])) if weights is None else weights
    s = {k: w @ d[k].astype(np.float64) for k in ("inter_off", "gt_off", "pred_off",
                                                   "inter_bg", "gt_bg", "pred_bg")}
    iou0 = _iou(s["inter_off"], s["gt_off"], s["pred_off"])
    with np.errstate(invalid="ignore", divide="ignore"):
        acc0 = np.where(s["gt_off"] > 0, s["inter_off"] / np.maximum(s["gt_off"], 1), np.nan)
    iou1 = _iou(s["inter_bg"], s["gt_bg"], s["pred_bg"])
    out = {
        "P0_mIoU": 100 * np.nanmean(iou0),
        "P0_aAcc": 100 * s["inter_off"].sum() / max(s["gt_off"].sum(), 1),
        "P0_mAcc": 100 * np.nanmean(acc0),
        "P1a_mIoU_C+1": 100 * np.nanmean(iou1),
        "P1b_fg_mIoU": 100 * np.nanmean(iou1[:C]),
        "P1_bg_IoU": 100 * iou1[C] if not np.isnan(iou1[C]) else float("nan"),
    }
    out.update(class_gated(d, w))
    return out


def class_gated(d, w):
    C = d["num_classes"]
    gt_fg = d["gt_bg"][:, :C]
    has_fg = gt_fg.sum(1) > 0
    main_gt = gt_fg.argmax(1)
    pred_fg = d["pred_bg"][:, :C]
    main_pred = np.where(pred_fg.sum(1) > 0, pred_fg.argmax(1), -1)
    recog = (main_pred == main_gt) & has_fg
    idx = np.arange(len(main_gt))
    inter_c = d["inter_bg"][idx, main_gt].astype(np.float64)
    union_c = (d["gt_bg"][idx, main_gt] + d["pred_bg"][idx, main_gt]).astype(np.float64) - inter_c
    iou_c = np.where(union_c > 0, inter_c / np.maximum(union_c, 1), 0.0)
    # class-agnostic foreground IoU: fg = any class < C
    gt_fg_area = gt_fg.sum(1).astype(np.float64)
    pred_fg_area = pred_fg.sum(1).astype(np.float64)
    # pixels that are fg in both = all GT-fg pixels minus those predicted as bg
    both_fg = gt_fg_area - _gt_fg_pred_bg(d)
    fg_union = gt_fg_area + pred_fg_area - both_fg
    fg_iou = np.where(fg_union > 0, both_fg / np.maximum(fg_union, 1), 0.0)
    ww = w * has_fg
    n = max(ww.sum(), 1e-9)
    return {
        "P2_recog_acc": 100 * (ww @ recog) / n,
        "P2_fg_IoU": 100 * (ww @ fg_iou) / n,
        "P2_gated_IoU": 100 * (ww @ (iou_c * recog)) / n,
        "P2_n_images": float(has_fg.sum()),
    }


def _gt_fg_pred_bg(d):
    """Per image: #GT-foreground pixels predicted as background (label C).

    From P1 statistics: pred_bg[:, C] counts pixels predicted C among fg+bg GT;
    inter_bg[:, C] counts those whose GT is also bg. The difference is GT-fg->pred-bg.
    """
    C = d["num_classes"]
    return (d["pred_bg"][:, C] - d["inter_bg"][:, C]).astype(np.float64)


def bootstrap(d, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(d["gt_off"])
    samples = []
    for _ in range(n_boot):
        w = rng.multinomial(N, np.full(N, 1.0 / N)).astype(np.float64)
        samples.append(metrics(d, w))
    keys = samples[0].keys()
    return {k: (np.percentile([s[k] for s in samples], 2.5),
                np.percentile([s[k] for s in samples], 97.5)) for k in keys}


def kendall_tau(a, b):
    n = len(a)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            conc += s > 0
            disc += s < 0
    return (conc - disc) / max(conc + disc, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--bootstrap", type=int, default=0)
    args = ap.parse_args()
    names = args.names or args.runs
    table = []
    for name, path in zip(names, args.runs):
        d = load(path)
        m = metrics(d)
        if args.bootstrap:
            ci = bootstrap(d, args.bootstrap)
            m.update({f"{k}_CI": ci[k] for k in ("P0_mIoU", "P1b_fg_mIoU", "P2_gated_IoU")})
        table.append((name, m, d["meta"]))
    keys = ["P0_mIoU", "P0_aAcc", "P0_mAcc", "P1a_mIoU_C+1", "P1b_fg_mIoU", "P1_bg_IoU",
            "P2_recog_acc", "P2_fg_IoU", "P2_gated_IoU"]
    print("run\t" + "\t".join(keys))
    for name, m, _ in table:
        print(name + "\t" + "\t".join(f"{m[k]:.2f}" for k in keys))
        for k in ("P0_mIoU", "P1b_fg_mIoU", "P2_gated_IoU"):
            if f"{k}_CI" in m:
                lo, hi = m[f"{k}_CI"]
                print(f"    {k} 95% CI [{lo:.2f}, {hi:.2f}]")
    if len(table) >= 3:
        p0 = [m["P0_mIoU"] for _, m, _ in table]
        for k in ("P1a_mIoU_C+1", "P1b_fg_mIoU", "P2_gated_IoU"):
            print(f"Kendall tau(P0, {k}) = {kendall_tau(p0, [m[k] for _, m, _ in table]):.3f}")


if __name__ == "__main__":
    main()
