"""Audit the official Earth2Ocean code package (no GPU, no dataset needed).

Reproduces the package-level facts recorded in docs/uovs_research_plan.md (section 1):
  * class-list lineage between AquaOV255 and USIS16K,
  * image-list overlap between the six UOVSBench datasets,
  * class-name formatting (trailing spaces / missing word boundaries),
  * MLLM JSON coverage and Objects statistics,
  * accuracy of the MLLM "Objects" field against the class encoded in the
    AquaOV255 file name (a proxy label; ~98.8% of images have one valid class).

Usage:
    python scripts/audit_official_package.py --code_dir /path/to/Earth2Ocean-code
"""
import argparse
import collections
import json
import os
import re

DATASETS = {
    # name: (class file, image list, json prefix)
    "AquaOV255": ("cls_AquaOV255.txt", "imglist_AquaOV255.txt", "llm_reasoning_AquaOV255"),
    "USIS16K": ("cls_usis16k.txt", "imglist_usis16k.txt", "llm_reasoning_usis16k"),
    "MAS3K": ("cls_mas3k.txt", "imglist_mask3k.txt", "llm_reasoning_mas3k"),
    "USIS10K": ("cls_usis10kvaltest.txt", "imglist_usis10kvaltest.txt", "llm_reasoning_usis10kvaltest"),
    "SUIM": ("cls_suimtrainval.txt", "imglist_suimtrainval.txt", "llm_reasoning_suimtrainval"),
    "DUT-USEG": ("cls_dutuseg.txt", "imglist_dutuseg.txt", "llm_reasoning_dutuseg"),
}
MLLMS = ["4o", "qwen7b", "qwen3b"]


def norm(s):
    """Letters only, lower case: 'Twin-spot goby' -> 'twinspotgoby'."""
    return re.sub(r"[^a-z]", "", s.lower())


def read_raw_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [x for x in f.read().split("\n") if x.strip()]


def read_list(path):
    return [x.strip() for x in read_raw_lines(path)]


def filename_class_key(fname):
    """'AlligatorGar_004.jpg' -> 'alligatorgar'; 'Abalone0830001.jpg' -> 'abalone'."""
    stem = fname.rsplit(".", 1)[0]
    stem = re.sub(r"_?\d+$", "", stem)
    return norm(stem)


def section(title):
    print("\n" + "=" * 8 + " " + title + " " + "=" * 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_dir", required=True)
    args = ap.parse_args()
    cfg = os.path.join(args.code_dir, "configs")

    classes, imgs = {}, {}
    for ds, (cls_f, img_f, _) in DATASETS.items():
        classes[ds] = read_raw_lines(os.path.join(cfg, cls_f))
        imgs[ds] = set(read_list(os.path.join(cfg, img_f)))

    section("Class-name formatting")
    for ds, raw in classes.items():
        n_trailing = sum(1 for x in raw if x != x.rstrip())
        n_space_inside = sum(1 for x in raw if " " in x.strip())
        n_camel = sum(1 for x in raw if re.search(r"[a-z][A-Z]", x))
        print(f"{ds:10s} classes={len(raw):4d} trailing_space={n_trailing:4d} "
              f"internal_space={n_space_inside:4d} camelcase={n_camel:4d}")

    section("Class-list lineage AquaOV255 vs USIS16K")
    a = [x.strip() for x in classes["AquaOV255"]]
    u = [x.strip() for x in classes["USIS16K"]]
    same_idx = sum(1 for i, x in enumerate(u) if i < len(a) and norm(x) == norm(a[i]))
    print(f"USIS16K classes whose normalized name equals AquaOV255 class at the same index: {same_idx}/{len(u)}")
    print("mismatches:", [(u[i], a[i]) for i in range(len(u)) if norm(u[i]) != norm(a[i])])

    section("Image-list overlap (by file name)")
    names = list(DATASETS)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            inter = len(imgs[x] & imgs[y])
            if inter:
                print(f"{x} & {y}: {inter}  ({x}={len(imgs[x])}, {y}={len(imgs[y])})")
    print(f"AquaOV255 images not in USIS16K: {len(imgs['AquaOV255'] - imgs['USIS16K'])}")
    print(f"USIS16K images not in AquaOV255: {len(imgs['USIS16K'] - imgs['AquaOV255'])}")

    section("MLLM JSON coverage / Objects statistics")
    for ds, (_, _, jprefix) in DATASETS.items():
        for m in MLLMS:
            p = os.path.join(cfg, f"{jprefix}_{m}.json")
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            missing = len(imgs[ds] - set(d))
            cnt = collections.Counter()
            empty = multi = 0
            for v in d.values():
                objs = [o for o in (v.get("Objects") or []) if isinstance(o, str)]
                empty += not objs
                multi += len(set(objs)) > 1
                cnt.update(objs)
            tot = sum(cnt.values()) or 1
            camel = sum(c for w, c in cnt.items() if re.search(r"[a-z][A-Z]", w))
            cls_keys = {norm(c) for c in classes[ds]}
            in_list = sum(c for w, c in cnt.items() if norm(w) in cls_keys)
            print(f"{ds:10s} {m:7s} keys={len(d):6d} missing_vs_imglist={missing:4d} "
                  f"empty={100 * empty / len(d):5.1f}% multi={100 * multi / len(d):5.1f}% "
                  f"camelcase_mentions={100 * camel / tot:5.1f}% in_class_list={100 * in_list / tot:5.1f}%")

    section("AquaOV255: MLLM first Object vs class encoded in file name")
    cls_index = {norm(c): i for i, c in enumerate(classes["AquaOV255"])}
    fname_label = {}
    unmapped = collections.Counter()
    for f in imgs["AquaOV255"]:
        k = filename_class_key(f)
        if k in cls_index:
            fname_label[f] = cls_index[k]
        else:
            unmapped[k] += 1
    print(f"images with file-name class mapped: {len(fname_label)}/{len(imgs['AquaOV255'])}; "
          f"unmapped prefixes: {unmapped.most_common(5)}")
    for m in MLLMS:
        p = os.path.join(cfg, f"llm_reasoning_AquaOV255_{m}.json")
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        tot = first = anyhit = nonempty = single = 0
        for f_, y in fname_label.items():
            if f_ not in d:
                continue
            tot += 1
            mapped = [cls_index[norm(o)] for o in (d[f_].get("Objects") or [])
                      if isinstance(o, str) and norm(o) in cls_index]
            nonempty += bool(mapped)
            single += len(set(mapped)) == 1
            first += bool(mapped) and mapped[0] == y
            anyhit += y in mapped
        print(f"{m:7s} n={tot} first==fname {100 * first / tot:.2f}% | any {100 * anyhit / tot:.2f}% | "
              f"non-empty {100 * nonempty / tot:.2f}% | single-candidate {100 * single / tot:.2f}%")

    section("Same image, two JSONs (AquaOV255 vs USIS16K, 4o)")
    with open(os.path.join(cfg, "llm_reasoning_AquaOV255_4o.json"), encoding="utf-8") as f:
        ja = json.load(f)
    with open(os.path.join(cfg, "llm_reasoning_usis16k_4o.json"), encoding="utf-8") as f:
        ju = json.load(f)
    both = [k for k in imgs["AquaOV255"] & imgs["USIS16K"] if k in ja and k in ju]
    same_cap = sum(1 for k in both if ja[k].get("Captioning") == ju[k].get("Captioning"))
    print(f"shared images present in both JSONs: {len(both)}; identical captions: {same_cap}")


if __name__ == "__main__":
    main()
