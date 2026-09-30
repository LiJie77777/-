"""Server-side data-integrity audit for UOVSBench (Line C, experiment C1).

1. Mask value census per dataset: which codes appear, which one is background / void.
2. Exact duplicates (MD5 of image bytes) and near duplicates (64-bit dHash, Hamming
   distance <= --max_hamming) within and across the six datasets.
3. For images shared by AquaOV255 and USIS16K (same file stem): compare the two GT
   masks (USIS16K class i corresponds to AquaOV255 class i for i < 158, see
   scripts/audit_official_package.py). Reports foreground IoU, class agreement and
   pixel agreement, i.e. an annotation-consistency estimate.

Only numpy + Pillow are required. Default layout follows the official configs:
    datasets/AquaOV255/{images,masks}      datasets/USIS16K/{images,annotations}
    datasets/MAS3K/{img,mask}              datasets/DUT-USEG/{img,mask}
    datasets/USIS10K_val_test/{img,mask}   datasets/SUIM_train_val/{img,mask}

Usage:
    python scripts/dedup_and_mask_consistency.py --root /home/jli/datasets --out audit_c1.json
    python scripts/dedup_and_mask_consistency.py --root ... --census_only
"""
import argparse
import collections
import hashlib
import json
import os

import numpy as np
from PIL import Image

LAYOUT = {
    "AquaOV255": ("AquaOV255/images", "AquaOV255/masks"),
    "USIS16K": ("USIS16K/images", "USIS16K/annotations"),
    "MAS3K": ("MAS3K/img", "MAS3K/mask"),
    "DUT-USEG": ("DUT-USEG/img", "DUT-USEG/mask"),
    "USIS10K": ("USIS10K_val_test/img", "USIS10K_val_test/mask"),
    "SUIM": ("SUIM_train_val/img", "SUIM_train_val/mask"),
}
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(p, size=8):
    im = Image.open(p).convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).ravel()
    return int(np.packbits(bits).view(">u8")[0])


def load_mask(p):
    # 'unchanged' semantics: keep 16-bit values (AquaOV255 uses I;16 with 65535 background)
    return np.array(Image.open(p))


def census(root, max_images=2000):
    out = {}
    for ds, (_, mdir) in LAYOUT.items():
        d = os.path.join(root, mdir)
        if not os.path.isdir(d):
            continue
        files = sorted(os.listdir(d))[:max_images]
        vals = collections.Counter()
        dtypes = collections.Counter()
        n_valid_classes = collections.Counter()
        for f in files:
            m = load_mask(os.path.join(d, f))
            dtypes[str(m.dtype) + ("x3" if m.ndim == 3 else "")] += 1
            u, c = np.unique(m if m.ndim == 2 else m[..., 0], return_counts=True)
            vals.update(dict(zip(u.tolist(), c.tolist())))
        out[ds] = {"n_masks_scanned": len(files), "dtypes": dict(dtypes),
                   "top_values_by_pixels": vals.most_common(12)}
    return out


def hash_all(root):
    table = {}
    for ds, (idir, _) in LAYOUT.items():
        d = os.path.join(root, idir)
        if not os.path.isdir(d):
            continue
        for f in list_images(d):
            p = os.path.join(d, f)
            table[(ds, f)] = (md5_file(p), dhash(p))
    return table


def near_duplicates(table, max_hamming):
    keys = list(table)
    hashes = np.array([table[k][1] for k in keys], dtype=np.uint64)
    pairs = []
    for i in range(len(keys)):
        x = np.bitwise_xor(hashes[i + 1:], hashes[i])
        dist = np.bitwise_count(x) if hasattr(np, "bitwise_count") else \
            np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        for j in np.nonzero(dist <= max_hamming)[0]:
            pairs.append((keys[i], keys[i + 1 + j], int(dist[j])))
    return pairs


def mask_consistency(root, bg_aqua, bg_usis, n_shared_classes=158, offset=0):
    a_img = {os.path.splitext(f)[0] for f in list_images(os.path.join(root, LAYOUT["AquaOV255"][0]))}
    u_img = {os.path.splitext(f)[0] for f in list_images(os.path.join(root, LAYOUT["USIS16K"][0]))}
    shared = sorted(a_img & u_img)
    stats = collections.Counter()
    fg_ious, pix_agree = [], []
    for s in shared:
        pa = os.path.join(root, LAYOUT["AquaOV255"][1], s + ".png")
        pu = os.path.join(root, LAYOUT["USIS16K"][1], s + ".png")
        if not (os.path.exists(pa) and os.path.exists(pu)):
            stats["missing_mask"] += 1
            continue
        ma, mu = load_mask(pa).astype(np.int64), load_mask(pu).astype(np.int64)
        if ma.shape != mu.shape:
            stats["shape_mismatch"] += 1
            continue
        fa, fu = ~np.isin(ma, bg_aqua), ~np.isin(mu, bg_usis)
        mu = mu - offset
        inter, union = np.sum(fa & fu), np.sum(fa | fu)
        fg_ious.append(inter / union if union else 1.0)
        both = fa & fu
        if both.any():
            pix_agree.append(np.mean(ma[both] == mu[both]))
        ca = set(np.unique(ma[fa]).tolist())
        cu = set(np.unique(mu[fu]).tolist())
        stats["same_class_set" if ca == cu else "diff_class_set"] += 1
        stats["aqua_class_outside_shared_range"] += any(c >= n_shared_classes for c in ca)
    return {"n_shared": len(shared), "counts": dict(stats),
            "fg_IoU_mean": float(np.mean(fg_ious)) if fg_ious else None,
            "fg_IoU_p10": float(np.percentile(fg_ious, 10)) if fg_ious else None,
            "frac_fg_IoU_eq_1": float(np.mean(np.isclose(fg_ious, 1.0))) if fg_ious else None,
            "class_pixel_agreement_mean": float(np.mean(pix_agree)) if pix_agree else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="audit_c1.json")
    ap.add_argument("--max_hamming", type=int, default=4)
    ap.add_argument("--census_only", action="store_true")
    ap.add_argument("--bg_aqua", type=int, nargs="+", default=[65535, 255],
                    help="background codes of AquaOV255 masks (verify with --census_only first)")
    ap.add_argument("--bg_usis16k", type=int, nargs="+", default=[255],
                    help="background codes of USIS16K masks; if USIS16K is 1-based with 0 as "
                         "background, pass 0 and set --usis16k_offset 1")
    ap.add_argument("--usis16k_offset", type=int, default=0,
                    help="subtract this from USIS16K class codes before comparing to AquaOV255")
    args = ap.parse_args()

    report = {"mask_census": census(args.root)}
    print(json.dumps(report["mask_census"], indent=1))
    if not args.census_only:
        table = hash_all(args.root)
        by_md5 = collections.defaultdict(list)
        for k, (m, _) in table.items():
            by_md5[m].append(k)
        exact = [v for v in by_md5.values() if len(v) > 1]
        cross = collections.Counter()
        for group in exact:
            dss = sorted({ds for ds, _ in group})
            if len(dss) > 1:
                cross[" & ".join(dss)] += 1
        report["exact_duplicate_groups"] = len(exact)
        report["exact_duplicate_groups_across_datasets"] = dict(cross)
        near = near_duplicates(table, args.max_hamming)
        near_cross = collections.Counter(" & ".join(sorted({a[0], b[0]})) for a, b, _ in near if a[0] != b[0])
        report["near_duplicate_pairs_across_datasets"] = dict(near_cross)
        report["near_duplicate_examples"] = [(a, b, d) for a, b, d in near[:50]]
        report["aqua_vs_usis16k_masks"] = mask_consistency(
            args.root, tuple(args.bg_aqua), tuple(args.bg_usis16k), offset=args.usis16k_offset)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in report.items() if k != "mask_census"}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
