# gleve_train/losses_gleve.py
from __future__ import annotations
import math
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F

def dice_loss(prob: torch.Tensor, gt: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    # prob, gt: (B,1,H,W,D)
    inter = (prob * gt).sum(dim=(2,3,4))
    union = prob.sum(dim=(2,3,4)) + gt.sum(dim=(2,3,4))
    dice = (2*inter + eps) / (union + eps)
    return (1 - dice).mean()


def foreground_recall_loss(prob: torch.Tensor, gt: torch.Tensor, min_ratio: float = 0.25, eps: float = 1e-6) -> torch.Tensor:
    """
    Encourage non-empty prediction on positive samples by enforcing a minimum predicted foreground mass.
    """
    pred_mass = prob.sum(dim=(2, 3, 4))
    gt_mass = gt.sum(dim=(2, 3, 4))
    has_fg = gt_mass > 0
    target_mass = gt_mass * min_ratio
    penalty = F.relu(target_mass - pred_mass) / (gt_mass + eps)
    penalty = penalty * has_fg.float()
    return penalty.sum() / (has_fg.float().sum() + 1.0)


def foreground_presence_loss(prob: torch.Tensor, gt: torch.Tensor, target_peak: float = 0.5) -> torch.Tensor:
    """
    Encourage at least some voxels to reach a confident foreground probability on positive samples.
    """
    gt_mass = gt.sum(dim=(2, 3, 4))
    has_fg = gt_mass > 0
    peak_prob = prob.amax(dim=(2, 3, 4))
    penalty = F.relu(target_peak - peak_prob) * has_fg.float()
    return penalty.sum() / (has_fg.float().sum() + 1.0)


def overseg_penalty(prob: torch.Tensor, gt: torch.Tensor, margin_ratio: float = 1.5, eps: float = 1e-6) -> torch.Tensor:
    """
    Penalize predictions whose foreground mass is much larger than GT.
    """
    pred_mass = prob.sum(dim=(2, 3, 4))
    gt_mass = gt.sum(dim=(2, 3, 4))
    has_fg = gt_mass > 0
    allowed_mass = torch.clamp(gt_mass * margin_ratio, min=64.0)
    penalty = F.relu(torch.log1p(pred_mass) - torch.log1p(allowed_mass))
    penalty = penalty * has_fg.float()
    return penalty.sum() / (has_fg.float().sum() + 1.0)


def false_positive_penalty(prob: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Directly penalize predicted foreground outside GT on supervised samples.
    """
    fp_mass = (prob * (1.0 - gt)).sum(dim=(2, 3, 4))
    pred_mass = prob.sum(dim=(2, 3, 4))
    gt_mass = gt.sum(dim=(2, 3, 4))
    has_fg = gt_mass > 0
    penalty = fp_mass / (pred_mass + gt_mass + eps)
    penalty = penalty * has_fg.float()
    return penalty.sum() / (has_fg.float().sum() + 1.0)


def false_negative_penalty(prob: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Directly penalize missed foreground on supervised samples.
    """
    fn_mass = ((1.0 - prob) * gt).sum(dim=(2, 3, 4))
    gt_mass = gt.sum(dim=(2, 3, 4))
    has_fg = gt_mass > 0
    penalty = fn_mass / (gt_mass + eps)
    penalty = penalty * has_fg.float()
    return penalty.sum() / (has_fg.float().sum() + 1.0)


def tversky_loss(
    prob: torch.Tensor,
    gt: torch.Tensor,
    alpha: float = 0.25,
    beta: float = 0.75,
    eps: float = 1e-6,
) -> torch.Tensor:
    tp = (prob * gt).sum(dim=(2, 3, 4))
    fp = (prob * (1.0 - gt)).sum(dim=(2, 3, 4))
    fn = ((1.0 - prob) * gt).sum(dim=(2, 3, 4))
    score = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1.0 - score).mean()

def bce_logits_loss(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pos = gt.sum()
    neg = gt.numel() - pos
    pos_weight = torch.clamp(neg / (pos + 1e-6), min=1.0, max=50.0)
    return F.binary_cross_entropy_with_logits(logits, gt, pos_weight=pos_weight)

def unimodal_verification_loss(scores: torch.Tensor, tau: float=1.0, eps: float=1e-8) -> torch.Tensor:
    """
    scores: (B,N,K) or (N,K) verification logits.
    Encourage a single dominant candidate via entropy minimization.
    """
    if scores.dim() == 2:
        scores = scores.unsqueeze(0)
    scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)
    p = F.softmax(scores / tau, dim=-1)
    ent = -(p * torch.log(p + eps)).sum(dim=-1)  # (B,N)
    return torch.nan_to_num(ent.mean(), nan=0.0, posinf=1e4, neginf=0.0)

def organ_consistency_loss(pred_mask: torch.Tensor, organ_mask: torch.Tensor) -> torch.Tensor:
    """
    Penalize predicted lesion mass outside target organ.
    pred_mask, organ_mask: (B,1,H,W,D) in [0,1]
    """
    outside = pred_mask * (1.0 - organ_mask)
    return outside.mean()

def exclusivity_loss(pred_masks: torch.Tensor) -> torch.Tensor:
    """
    pred_masks: (B,N,1,H,W,D) prob masks for multiple lesions (same case).
    Penalize overlap among lesions.
    """
    if pred_masks.shape[1] <= 1:
        return pred_masks.new_zeros(())
    # sum over lesions, overlap where sum>1
    s = pred_masks.sum(dim=1)  # (B,1,H,W,D)
    overlap = F.relu(s - 1.0)
    return overlap.mean()


def smoothness_loss(pred_masks: torch.Tensor) -> torch.Tensor:
    """
    Penalize salt-and-pepper predictions by total variation in 3D.
    pred_masks: (B,N,1,H,W,D)
    """
    dh = (pred_masks[:, :, :, 1:, :, :] - pred_masks[:, :, :, :-1, :, :]).abs().mean()
    dw = (pred_masks[:, :, :, :, 1:, :] - pred_masks[:, :, :, :, :-1, :]).abs().mean()
    dd = (pred_masks[:, :, :, :, :, 1:] - pred_masks[:, :, :, :, :, :-1]).abs().mean()
    return (dh + dw + dd) / 3.0


def compactness_loss(pred_masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Encourage compact, near-isotropic blobs using weighted second moments.
    pred_masks: (B,N,1,H,W,D)
    """
    B, N, _, H, W, D = pred_masks.shape
    prob = pred_masks.squeeze(2)

    gh = torch.linspace(-1.0, 1.0, H, device=prob.device, dtype=prob.dtype).view(1, 1, H, 1, 1)
    gw = torch.linspace(-1.0, 1.0, W, device=prob.device, dtype=prob.dtype).view(1, 1, 1, W, 1)
    gd = torch.linspace(-1.0, 1.0, D, device=prob.device, dtype=prob.dtype).view(1, 1, 1, 1, D)

    mass = prob.sum(dim=(2, 3, 4), keepdim=True) + eps
    ch = (prob * gh).sum(dim=(2, 3, 4), keepdim=True) / mass
    cw = (prob * gw).sum(dim=(2, 3, 4), keepdim=True) / mass
    cd = (prob * gd).sum(dim=(2, 3, 4), keepdim=True) / mass

    vh = (prob * (gh - ch).square()).sum(dim=(2, 3, 4)) / mass.squeeze(-1).squeeze(-1).squeeze(-1)
    vw = (prob * (gw - cw).square()).sum(dim=(2, 3, 4)) / mass.squeeze(-1).squeeze(-1).squeeze(-1)
    vd = (prob * (gd - cd).square()).sum(dim=(2, 3, 4)) / mass.squeeze(-1).squeeze(-1).squeeze(-1)

    mean_var = (vh + vw + vd) / 3.0
    anis = ((vh - mean_var).abs() + (vw - mean_var).abs() + (vd - mean_var).abs()) / (mean_var.abs() + eps)
    has_fg = prob.sum(dim=(2, 3, 4)) > eps
    return (anis * has_fg.float()).sum() / (has_fg.float().sum() + 1.0)

def weak_loss(
    ver_scores: torch.Tensor,
    pred_masks: torch.Tensor,      # (B,N,1,H,W,D) probs
    ct: torch.Tensor,              # (B,1,H,W,D) normalized 0..1 after HU windowing
    organ_masks: torch.Tensor,     # (B,N,1,H,W,D) organ prior per lesion
    report_targets: Dict[str, torch.Tensor],  # {"V":(B,N), "mu":(B,N)} with NaNs allowed
    voxel_volume_cc: torch.Tensor,
    hu_min: float,
    hu_max: float,
    hu_loss_scale: float,
    report_term_alpha: float = 0.0,
    eps: float=1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    B,N = pred_masks.shape[0], pred_masks.shape[1]

    pred_masks = torch.nan_to_num(pred_masks, nan=0.0, posinf=1.0, neginf=0.0)
    ct = torch.nan_to_num(ct, nan=0.0, posinf=0.0, neginf=0.0)
    organ_masks = torch.nan_to_num(organ_masks, nan=0.0, posinf=1.0, neginf=0.0)
    report_term_alpha = float(max(report_term_alpha, 0.0))

    Lexcl = exclusivity_loss(pred_masks)
    Lsmooth = smoothness_loss(pred_masks)
    Lcompact = compactness_loss(pred_masks)
    outside_mass = (pred_masks * (1.0 - organ_masks)).sum(dim=(2,3,4,5))
    pred_mass_full = pred_masks.sum(dim=(2,3,4,5))
    has_pred = pred_mass_full > eps
    Lorg = outside_mass / (pred_mass_full + eps)
    Lorg = (Lorg * has_pred.float()).sum() / (has_pred.float().sum() + 1.0)

    Lver = pred_masks.new_zeros(())
    Lv = pred_masks.new_zeros(())
    Lv_under = pred_masks.new_zeros(())
    Lmu = pred_masks.new_zeros(())
    if report_term_alpha > 0.0:
        voxel_volume_cc = torch.as_tensor(voxel_volume_cc, device=pred_masks.device, dtype=pred_masks.dtype).view(B, 1)
        pred_volume_cc = pred_mass_full * voxel_volume_cc
        target_volume_cc = report_targets["V"].to(device=pred_masks.device, dtype=pred_masks.dtype)
        valid_v = torch.isfinite(target_volume_cc) & (target_volume_cc > 0)
        if valid_v.any():
            pred_log_v = torch.log1p(torch.clamp(pred_volume_cc, min=0.0))
            target_log_v = torch.log1p(torch.clamp(target_volume_cc, min=0.0))
            vol_err = F.smooth_l1_loss(pred_log_v[valid_v], target_log_v[valid_v], reduction="mean")
            vol_under = F.relu(target_log_v[valid_v] - pred_log_v[valid_v]).mean()
            Lv = torch.nan_to_num(vol_err, nan=0.0, posinf=1e4, neginf=0.0)
            Lv_under = torch.nan_to_num(vol_under, nan=0.0, posinf=1e4, neginf=0.0)

        ct_hu = ct * float(hu_max - hu_min) + float(hu_min)
        ct_hu = torch.nan_to_num(ct_hu, nan=float(hu_min), posinf=float(hu_max), neginf=float(hu_min))
        pred_mu = (pred_masks * ct_hu.unsqueeze(1)).sum(dim=(2, 3, 4, 5)) / (pred_mass_full + eps)
        target_mu = report_targets["mu"].to(device=pred_masks.device, dtype=pred_masks.dtype)
        target_mu = target_mu.clamp(min=float(hu_min), max=float(hu_max))
        valid_mu = torch.isfinite(target_mu) & (pred_mass_full > 16.0)
        if valid_mu.any():
            mu_err = F.smooth_l1_loss(
                pred_mu[valid_mu] / max(float(hu_loss_scale), 1.0),
                target_mu[valid_mu] / max(float(hu_loss_scale), 1.0),
                reduction="mean",
            )
            Lmu = torch.nan_to_num(mu_err, nan=0.0, posinf=1e4, neginf=0.0)

        Lver = unimodal_verification_loss(ver_scores)

    Lexcl = torch.nan_to_num(Lexcl, nan=0.0, posinf=1e4, neginf=0.0)
    Lsmooth = torch.nan_to_num(Lsmooth, nan=0.0, posinf=1e4, neginf=0.0)
    Lcompact = torch.nan_to_num(Lcompact, nan=0.0, posinf=1e4, neginf=0.0)
    Lorg = torch.nan_to_num(Lorg, nan=0.0, posinf=1e4, neginf=0.0)
    Lver = torch.nan_to_num(Lver, nan=0.0, posinf=1e4, neginf=0.0)
    Lv = torch.nan_to_num(Lv, nan=0.0, posinf=1e4, neginf=0.0)
    Lv_under = torch.nan_to_num(Lv_under, nan=0.0, posinf=1e4, neginf=0.0)
    Lmu = torch.nan_to_num(Lmu, nan=0.0, posinf=1e4, neginf=0.0)

    base_total = torch.nan_to_num(
        0.25 * Lexcl + 2.5 * Lorg + 0.10 * Lsmooth + 0.05 * Lcompact,
        nan=0.0,
        posinf=100.0,
        neginf=0.0,
    )
    report_total = torch.nan_to_num(
        0.15 * Lver + 0.70 * Lv + 0.40 * Lv_under + 0.45 * Lmu,
        nan=0.0,
        posinf=100.0,
        neginf=0.0,
    )
    total = torch.nan_to_num(
        base_total + report_term_alpha * report_total,
        nan=0.0,
        posinf=100.0,
        neginf=0.0,
    )
    parts = {
        "Lver": Lver.detach(),
        "Lv": Lv.detach(),
        "Lv_under": Lv_under.detach(),
        "Lmu": Lmu.detach(),
        "Lexcl": Lexcl.detach(),
        "Lsmooth": Lsmooth.detach(),
        "Lcompact": Lcompact.detach(),
        "Lorg": Lorg.detach(),
        "Lweak_base": base_total,
        "Lweak_report": report_total,
    }
    return total, parts


def _dilate_mask(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if kernel_size <= 1:
        return (mask > 0).float()
    pad = kernel_size // 2
    return F.max_pool3d(mask.float(), kernel_size=kernel_size, stride=1, padding=pad)


def _bbox_stats(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    mask: (..., H, W, D) binary-ish tensor
    returns:
      center: (..., 3) normalized to [0,1]
      size:   (..., 3) normalized by spatial size
      valid:  (...) float in {0,1}
    """
    lead_shape = mask.shape[:-3]
    H, W, D = mask.shape[-3:]
    flat = (mask > 0).reshape(-1, H, W, D)
    device = mask.device
    dtype = mask.dtype
    centers = torch.zeros((flat.shape[0], 3), device=device, dtype=dtype)
    sizes = torch.zeros((flat.shape[0], 3), device=device, dtype=dtype)
    valid = torch.zeros((flat.shape[0],), device=device, dtype=dtype)
    center_norm = torch.tensor([max(H - 1, 1), max(W - 1, 1), max(D - 1, 1)], device=device, dtype=dtype)
    size_norm = torch.tensor([max(H, 1), max(W, 1), max(D, 1)], device=device, dtype=dtype)

    for i in range(flat.shape[0]):
        idx = flat[i].nonzero(as_tuple=False)
        if idx.numel() == 0:
            continue
        mins = idx.min(dim=0).values.to(dtype)
        maxs = idx.max(dim=0).values.to(dtype)
        centers[i] = ((mins + maxs) * 0.5) / center_norm
        sizes[i] = (maxs - mins + 1.0) / size_norm
        valid[i] = 1.0

    return (
        centers.view(*lead_shape, 3),
        sizes.view(*lead_shape, 3),
        valid.view(*lead_shape),
    )


def candidate_region_loss(
    scores: torch.Tensor,
    cand_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    beta: float = 2.0,
    dilate_kernel: int = 5,
    target_temp: float = 0.25,
    score_temp: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Supervise verification scores with a hybrid target:
    1. coverage against a slightly dilated GT mask
    2. candidate-vs-GT bbox center distance
    3. candidate-vs-GT bbox size agreement

    Pure mask IoU is too harsh for coarse top-k candidates over irregular
    lesions. Pure distance is also insufficient because it can reward a small
    blob near the lesion center without actually covering the lesion. The target
    below keeps both location and coverage.

    scores: (B,N,K)
    cand_masks: (B,N,K,1,h,w,d)
    gt_masks: (B,N,1,h,w,d)
    """
    B, N, K = scores.shape
    cand = cand_masks.squeeze(3).float()
    gt = gt_masks.float()
    _, _, _, H, W, D = cand.shape

    gt_dil = _dilate_mask(gt.view(B * N, 1, H, W, D), kernel_size=dilate_kernel).view(B, N, 1, H, W, D)
    inter = (cand * gt).sum(dim=(3, 4, 5))
    inter_dil = (cand * gt_dil).sum(dim=(3, 4, 5))
    cand_mass = cand.sum(dim=(3, 4, 5))
    gt_mass = gt.sum(dim=(2, 3, 4, 5))
    gt_dil_mass = gt_dil.sum(dim=(2, 3, 4, 5))
    gt_mass_u = gt_mass.unsqueeze(-1)
    gt_dil_mass_u = gt_dil_mass.unsqueeze(-1)
    union = cand_mass + gt_mass_u - inter
    iou = inter / (union + eps)
    recall = inter / (gt_mass_u + eps)
    precision = inter / (cand_mass + eps)
    beta2 = float(beta) * float(beta)
    fbeta = ((1.0 + beta2) * precision * recall) / (beta2 * precision + recall + eps)

    recall_dil = inter_dil / (gt_dil_mass_u + eps)
    inside_dil = inter_dil / (cand_mass + eps)
    cand_center, cand_size, cand_valid = _bbox_stats(cand)
    gt_center, gt_size, gt_valid = _bbox_stats(gt_dil.squeeze(2))
    gt_center = gt_center.unsqueeze(2)
    gt_size = gt_size.unsqueeze(2)
    gt_valid = gt_valid.unsqueeze(-1)

    center_dist = torch.linalg.norm(cand_center - gt_center, dim=-1) / math.sqrt(3.0)
    center_quality = torch.exp(-4.0 * center_dist)
    size_gap = (torch.log(cand_size + 1e-3) - torch.log(gt_size + 1e-3)).abs().mean(dim=-1)
    size_quality = torch.exp(-2.0 * size_gap)

    # Favor candidates that stay close to the dilated GT box and still cover the lesion area.
    quality = 0.40 * recall_dil + 0.20 * inside_dil + 0.30 * center_quality + 0.10 * size_quality
    quality = torch.where((cand_valid > 0) & (gt_valid > 0), quality, quality.new_full(quality.shape, -1e4))
    has_fg = gt_mass > 0
    if not has_fg.any():
        zero = scores.new_zeros(())
        return zero, {
            "Lloc_rank_soft": zero,
            "best_iou": zero,
            "best_recall": zero,
            "best_fbeta": zero,
        }

    scores = torch.nan_to_num(scores, nan=0.0, posinf=8.0, neginf=-8.0)
    target_prob = torch.softmax(quality / max(float(target_temp), 1e-3), dim=-1).detach()
    log_prob = F.log_softmax(scores.view(B * N, K) / max(float(score_temp), 1e-3), dim=-1)
    target_prob = target_prob.view(B * N, K)
    has_fg_flat = has_fg.view(B * N)
    soft_ce = -(target_prob[has_fg_flat] * log_prob[has_fg_flat]).sum(dim=-1)
    loss = soft_ce.mean()
    loss = torch.nan_to_num(loss, nan=0.0, posinf=100.0, neginf=0.0)
    parts = {
        "Lloc_rank_soft": loss.detach(),
        "best_iou": (iou.max(dim=-1).values * has_fg.float()).sum() / (has_fg.float().sum() + 1.0),
        "best_recall": (recall.max(dim=-1).values * has_fg.float()).sum() / (has_fg.float().sum() + 1.0),
        "best_fbeta": (fbeta.max(dim=-1).values * has_fg.float()).sum() / (has_fg.float().sum() + 1.0),
        "best_center_quality": (center_quality.max(dim=-1).values * has_fg.float()).sum() / (has_fg.float().sum() + 1.0),
    }
    return loss, parts

def dense_candidate_localization_loss(
    sim_scores: torch.Tensor,
    gt_masks: torch.Tensor,
    logit_scale: float = 8.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Directly supervise the dense similarity map so the query/cosine branch
    receives gradients. This avoids relying only on hard-threshold candidates
    and verifier ranking.

    sim_scores: (B,N,h,w,d)
    gt_masks: (B,N,1,h,w,d)
    """
    B, N, H, W, D = sim_scores.shape
    logits = (sim_scores * logit_scale).reshape(B * N, 1, H, W, D)
    gt = gt_masks.reshape(B * N, 1, H, W, D).float()
    has_fg = gt.sum(dim=(1, 2, 3, 4)) > 0
    if not has_fg.any():
        zero = sim_scores.new_zeros(())
        return zero, {"Lloc_dense_bce": zero, "Lloc_dense_dice": zero, "dense_iou": zero}

    logits = logits[has_fg]
    gt = gt[has_fg]
    prob = torch.sigmoid(logits)
    Ldice = dice_loss(prob, gt)
    Lbce = bce_logits_loss(logits, gt)
    loss = 0.5 * Ldice + 0.5 * Lbce

    pred_bin = (prob > 0.5).float()
    inter = (pred_bin * gt).sum(dim=(2, 3, 4))
    union = pred_bin.sum(dim=(2, 3, 4)) + gt.sum(dim=(2, 3, 4)) - inter
    dense_iou = (inter / (union + eps)).mean()
    parts = {
        "Lloc_dense_bce": Lbce.detach(),
        "Lloc_dense_dice": Ldice.detach(),
        "dense_iou": dense_iou.detach(),
    }
    return loss, parts

def seg_loss(logits: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    logits, gt: (B,1,H,W,D)
    """
    logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
    gt = torch.nan_to_num(gt, nan=0.0, posinf=1.0, neginf=0.0)
    prob = torch.sigmoid(logits)
    Ldice = dice_loss(prob, gt)
    Ltv = tversky_loss(prob, gt, alpha=0.25, beta=0.75)
    Lbce = bce_logits_loss(logits, gt)
    Lrec = false_negative_penalty(prob, gt)
    Lpre = foreground_recall_loss(prob, gt, min_ratio=0.75)
    Lover = overseg_penalty(prob, gt, margin_ratio=3.0)
    Lfp = false_positive_penalty(prob, gt)
    loss = 0.28 * Ldice + 0.18 * Ltv + 0.20 * Lbce + 0.18 * Lrec + 0.10 * Lpre + 0.04 * Lover + 0.02 * Lfp
    loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)
    parts = {
        "Ldice": Ldice.detach(),
        "Ltv": Ltv.detach(),
        "Lbce": Lbce.detach(),
        "Lrec": Lrec.detach(),
        "Lpre": Lpre.detach(),
        "Lover": Lover.detach(),
        "Lfp": Lfp.detach(),
    }
    return loss, parts
