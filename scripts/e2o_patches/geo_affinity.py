"""Card G2 of docs/geometry_fusion_research.md: 3D-aware proxy affinity for E2O's GMG.

E2O (open_clip/transformer.py, VisionTransformer.custom_attn) builds
    S~ = gamma * (S - beta * mean(S)),  S~[S~ < 0] = -inf,  A = softmax(S~)
from cosine self-similarity S of DAv2 block-11 tokens. This module adds an explicit
geometric term built from the depth map that E2O already computes and discards
(get_geo_feats returns it as geo_map):

    pen_ij = scale * (lam_z * |z_i - z_j| / med|dz|  +  lam_s * ||p_i - p_j||_1 / med||dp||_1)

z = per-window median/MAD-normalised disparity (invariant to DA's unknown scale/shift),
p = token grid coordinates, scale = median of the positive entries of S~ (label-free).
  mode "soft": cut on S~ as before, then subtract pen from the surviving entries.
               softmax(S~ - pen) == A * exp(-pen) renormalised, i.e. the DFormerv2
               geometry-prior form A * beta^G with a fixed (not learned) decay.
  mode "hard": subtract pen before the cut, so far-apart-in-depth pairs can be cut.
lam_z = 0 with lam_s > 0 is the spatial-only control. Shuffled depth, blurred depth or
a non-geometric smooth map (pca1_map of DINOv2 tokens) are the other controls: pass them
as `zmap` instead of the depth.

Hook (three edits, see the report, card G2):
  1. Earth2Ocean.forward_feature: keep geo_map from get_geo_feats, pool it to the token
     grid: zmap = F.adaptive_avg_pool2d(geo_map.float().reshape(B,1,*geo_map.shape[-2:]),
     (vfm_h, vfm_w))[:, 0]; pass zmap and geo_cfg through clip.encode_image.
  2. open_clip/model.py encode_image and VisionTransformer.forward: forward the two args.
  3. custom_attn: after `similarity = (similarity - mean*beta) * gamma` and BEFORE the
     `< 0 -> -inf` line, call geo_adjust_similarity_torch(similarity, zmap, (H, W), **geo_cfg)
     and drop the original -inf line (the function applies the cut itself).

The numpy functions are the reference (tested in scripts/tests/test_geo_affinity.py);
the torch function is a line-by-line port that has not been run in this environment.
"""
import numpy as np

_EPS = 1e-6


def _norm_z(d):
    med = np.median(d, axis=1, keepdims=True)
    mad = np.median(np.abs(d - med), axis=1, keepdims=True) + _EPS
    return (d - med) / mad


def geo_adjust_similarity_np(sim, zmap, hw, mode="soft", lam_z=0.25, lam_s=0.0):
    """sim: (B, L, L) = gamma*(S - beta*mean S) before the cut; zmap: (B, H, W).
    Returns the adjusted similarity with -inf for cut pairs."""
    B, L, _ = sim.shape
    H, W = hw
    keep = sim > 0
    pos = np.where(keep, sim, np.nan)
    scale = np.nanmedian(pos.reshape(B, -1), axis=1)[:, None, None]
    pen = np.zeros_like(sim)
    if lam_z:
        z = _norm_z(zmap.reshape(B, L).astype(np.float64))
        dz = np.abs(z[:, :, None] - z[:, None, :])
        pen = pen + lam_z * dz / (np.median(dz.reshape(B, -1), axis=1)[:, None, None] + _EPS)
    if lam_s:
        ys, xs = np.mgrid[0:H, 0:W]
        p = np.stack([ys.ravel(), xs.ravel()], 1).astype(np.float64)
        ds = np.abs(p[:, None, :] - p[None, :, :]).sum(-1)
        pen = pen + lam_s * ds / (np.median(ds) + _EPS)
    pen = scale * pen
    if mode == "soft":
        return np.where(keep, sim - pen, -np.inf)
    if mode == "hard":
        out = sim - pen
        return np.where(out > 0, out, -np.inf)
    raise ValueError(mode)


def softmax_np(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def geo_adjust_similarity_torch(sim, zmap, hw, mode="soft", lam_z=0.25, lam_s=0.0):
    """Torch port of geo_adjust_similarity_np (same arguments, same semantics)."""
    import torch

    B, L, _ = sim.shape
    H, W = hw
    s = sim.float()
    keep = s > 0
    scale = torch.stack([s[b][keep[b]].median() if keep[b].any() else s.new_tensor(1.0)
                         for b in range(B)])[:, None, None]
    pen = torch.zeros_like(s)
    if lam_z:
        d = zmap.reshape(B, L).float()
        med = d.median(dim=1, keepdim=True).values
        mad = (d - med).abs().median(dim=1, keepdim=True).values + _EPS
        z = (d - med) / mad
        dz = (z[:, :, None] - z[:, None, :]).abs()
        pen = pen + lam_z * dz / (dz.reshape(B, -1).median(dim=1).values[:, None, None] + _EPS)
    if lam_s:
        ys, xs = torch.meshgrid(torch.arange(H, device=s.device), torch.arange(W, device=s.device), indexing="ij")
        p = torch.stack([ys.flatten(), xs.flatten()], 1).float()
        ds = torch.cdist(p, p, p=1)
        pen = pen + lam_s * ds / (ds.median() + _EPS)
    pen = scale * pen
    if mode == "soft":
        out = torch.where(keep, s - pen, torch.full_like(s, float("-inf")))
    elif mode == "hard":
        out = s - pen
        out = torch.where(out > 0, out, torch.full_like(s, float("-inf")))
    else:
        raise ValueError(mode)
    return out.to(sim.dtype)


def pca1_map_torch(feats):
    """Non-geometric smooth-map control: first principal component of token features
    (B, C, H, W) -> (B, H, W), sign fixed so that the map correlates positively with the
    mean token norm (an arbitrary but label-free convention)."""
    import torch

    B, C, H, W = feats.shape
    x = feats.float().flatten(2).transpose(1, 2)  # B, L, C
    x = x - x.mean(1, keepdim=True)
    _, _, v = torch.pca_lowrank(x, q=1, center=False)
    m = (x @ v[..., :1]).squeeze(-1)  # B, L
    norm = feats.float().flatten(2).norm(dim=1)
    sign = torch.sign(((m - m.mean(1, keepdim=True)) * (norm - norm.mean(1, keepdim=True))).sum(1, keepdim=True))
    return (m * torch.where(sign == 0, torch.ones_like(sign), sign)).reshape(B, H, W)
