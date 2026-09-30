"""Whole-image "oracle-name" fill on any mask folder, under all protocols of
protocol_eval.py (Line C, experiment C3-ref).

For every image, predict the largest valid GT class on every pixel (a diagnostic upper
bound, not a model). Run it on the six UOVSBench datasets (should reproduce hand-off
§6.1: e.g. AquaOV255 official 99.66) and on accepted benchmarks without scored background
(PASCAL VOC20: SegmentationClass, bg=0 -> use --class_offset 1 --bg_codes 0 --void_codes 255)
to calibrate how "degenerate" UOVSBench is relative to them.

Examples:
    python scripts/oracle_fill_reference.py --mask_dir datasets/AquaOV255/masks \
        --num_classes 254 --bg_codes 65535 --void_codes
    python scripts/oracle_fill_reference.py --mask_dir VOC2012/SegmentationClass \
        --list VOC2012/ImageSets/Segmentation/val.txt --num_classes 20 \
        --class_offset 1 --bg_codes 0 --void_codes 255
"""
import argparse
import os

import numpy as np
from PIL import Image

from protocol_eval import RunAccumulator, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--list", default=None, help="optional file with one stem per line")
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--class_offset", type=int, default=0,
                    help="subtract from raw codes to get 0-based classes (VOC: 1)")
    ap.add_argument("--bg_codes", type=int, nargs="*", default=[])
    ap.add_argument("--void_codes", type=int, nargs="*", default=[])
    args = ap.parse_args()

    if args.list:
        stems = [x.strip() for x in open(args.list) if x.strip()]
        files = [s + ".png" for s in stems]
    else:
        files = sorted(f for f in os.listdir(args.mask_dir) if f.endswith(".png"))
    C = args.num_classes
    acc = RunAccumulator(C)
    n_fg_classes = []
    fg_frac = []
    for f in files:
        raw = np.array(Image.open(os.path.join(args.mask_dir, f))).astype(np.int64)
        bg = np.isin(raw, args.bg_codes)
        void = np.isin(raw, args.void_codes)
        gt = raw - args.class_offset
        gt = np.where(bg, -2, np.where(void, -1, gt))  # -2 background, -1 void
        fg = gt >= 0
        cls, cnt = np.unique(gt[fg], return_counts=True)
        n_fg_classes.append(len(cls))
        fg_frac.append(fg.sum() / max((~void).sum(), 1))
        main = int(cls[np.argmax(cnt)]) if len(cls) else 0
        pred = np.full(gt.shape, main, np.int64)
        acc.add(pred, gt, bg_codes=(-2,), void_codes=(-1,), image_id=f)
    n = np.array(n_fg_classes)
    print(f"images={len(files)}  0 fg class: {np.mean(n == 0) * 100:.2f}%  "
          f"1 fg class: {np.mean(n == 1) * 100:.2f}%  >=2: {np.mean(n >= 2) * 100:.2f}%  "
          f"mean fg fraction of non-void pixels: {100 * np.mean(fg_frac):.2f}%")
    m = metrics({**acc.arrays(), "num_classes": C})
    for k, v in m.items():
        print(f"{k:14s} {v:8.2f}")


if __name__ == "__main__":
    main()
