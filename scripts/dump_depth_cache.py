"""Cache Depth Anything V2 relative inverse depth (disparity; larger = nearer) for every
image of a dataset, one <stem>.npy (float16, original image size) per image.

Input to scripts/prevalidate_depth_figure_ground.py and to the depth cards in
docs/geometry_fusion_research.md. Uses the depth_anything_v2 package shipped with the
official Earth2Ocean code and its own infer_image() (whole image, short side 518), so the
map is the DPT head output that E2O computes and discards in get_geo_feats().

Note: E2O itself runs DA per 336x336 sliding window, so its in-model depth is normalised
per window. This cache is per whole image (one consistent scale/shift per image).

Requires torch + opencv + the DAv2 checkpoint; not runnable without a GPU environment.

Usage:
    python scripts/dump_depth_cache.py --e2o_root /path/Earth2Ocean-code \
        --ckpt /path/Earth2Ocean-code/pretrained_ckpt/depth_anything_v2_vitb.pth \
        --img_dir datasets/AquaOV255/images --out_dir cache/depth_vitb/AquaOV255
"""
import argparse
import os
import sys

import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2o_root", required=True, help="folder that contains depth_anything_v2/")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--encoder", default="vitb", choices=sorted(CONFIGS))
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--list", default=None, help="optional file with one image file name per line")
    ap.add_argument("--input_size", type=int, default=518)
    args = ap.parse_args()

    sys.path.insert(0, args.e2o_root)
    import cv2
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(**CONFIGS[args.encoder])
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    if args.list:
        files = [x.strip() for x in open(args.list) if x.strip()]
    else:
        files = sorted(f for f in os.listdir(args.img_dir) if f.lower().endswith(IMG_EXT))
    os.makedirs(args.out_dir, exist_ok=True)
    for i, f in enumerate(files):
        out = os.path.join(args.out_dir, os.path.splitext(f)[0] + ".npy")
        if os.path.exists(out):
            continue
        raw = cv2.imread(os.path.join(args.img_dir, f))  # BGR, as infer_image expects
        if raw is None:
            print(f"skip unreadable {f}")
            continue
        with torch.no_grad():
            d = model.infer_image(raw, args.input_size)
        np.save(out, d.astype(np.float16))
        if i % 200 == 0:
            print(f"{i}/{len(files)} {f} range=({d.min():.3f},{d.max():.3f})")


if __name__ == "__main__":
    main()
