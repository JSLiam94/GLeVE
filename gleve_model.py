# gleve_model.py
from __future__ import annotations
from collections import deque
from functools import lru_cache
import math
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_rel_encoder import LeQuEncoder
from losses_gleve import candidate_region_loss, dense_candidate_localization_loss, seg_loss, weak_loss
from medformer3d import MedFormer3D
from octree_refiner import OcReOctree
from utils_nii import HU_MAX, HU_MIN

try:
    from scipy import ndimage as scipy_ndimage
except ImportError:
    scipy_ndimage = None

try:
    from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment
except ImportError:
    scipy_linear_sum_assignment = None


def load_model_state_flexible(model: torch.nn.Module, state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load as many parameters as possible from a checkpoint even if the architecture
    changed slightly. Parameters with missing keys or mismatched tensor shapes are skipped.
    """
    model_state = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped_missing = []
    skipped_shape = []

    for k, v in state_dict.items():
        if k not in model_state:
            skipped_missing.append(k)
            continue
        if model_state[k].shape != v.shape:
            skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        filtered[k] = v

    missing_after = [k for k in model_state.keys() if k not in filtered]
    model_state.update(filtered)
    model.load_state_dict(model_state, strict=False)
    return {
        "loaded": len(filtered),
        "total": len(model_state),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "missing_after": missing_after,
    }


class FiLMAnatomyModulation(nn.Module):
    def __init__(self, C: int, n_organs: int=3, d_o: int=64):
        super().__init__()
        self.organ2id = {"kidney":0, "pancreas":1, "liver":2}
        self.emb = nn.Embedding(n_organs, d_o)
        self.phi_o = nn.Linear(d_o, C)
        self.psi = nn.Sequential(nn.Linear(C, C*2), nn.GELU(), nn.Linear(C*2, C*2))

    def forward(self, Fv: torch.Tensor, organ_masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fv: (B,C,h,w,d)
        organ_masks: dict organ -> (B,1,H,W,D) at full-res
        """
        B,C,h,w,d = Fv.shape
        device = Fv.device

        E = torch.zeros((B,h,w,d, C), device=device)
        for organ, m in organ_masks.items():
            if organ not in self.organ2id:
                continue
            oid = self.organ2id[organ]
            e = self.emb.weight[oid]              # (d_o,)
            eC = self.phi_o(e).view(1,1,1,1,C)    # (1,1,1,1,C)
            md = F.interpolate(m, size=(h,w,d), mode="trilinear", align_corners=False)  # (B,1,h,w,d)
            E = E + md.permute(0,2,3,4,1) * eC

        E = E.permute(0,4,1,2,3).contiguous()  # (B,C,h,w,d)
        gb = self.psi(E.permute(0,2,3,4,1)).view(B,h,w,d,2*C)
        gb = gb.permute(0,4,1,2,3).contiguous()
        gamma, beta = torch.chunk(gb, 2, dim=1)
        return gamma * Fv + beta


def _bbox_from_mask(mask: torch.Tensor, pad: int=8) -> Tuple[int,int,int,int,int,int]:
    """
    mask: (B,1,H,W,D) binary/prob
    returns bbox for batch=1: (h0,h1,w0,w1,d0,d1)
    """
    assert mask.shape[0] == 1, "This implementation assumes batch_size=1 (as your setup)."
    m = mask[0,0]  # (H,W,D)
    idx = (m > 0.5).nonzero(as_tuple=False)
    H,W,D = m.shape
    if idx.numel() == 0:
        return (0,H,0,W,0,D)
    h0 = int(idx[:,0].min().item()) - pad
    h1 = int(idx[:,0].max().item()) + pad + 1
    w0 = int(idx[:,1].min().item()) - pad
    w1 = int(idx[:,1].max().item()) + pad + 1
    d0 = int(idx[:,2].min().item()) - pad
    d1 = int(idx[:,2].max().item()) + pad + 1
    h0 = max(0,h0); w0=max(0,w0); d0=max(0,d0)
    h1 = min(H,h1); w1=min(W,w1); d1=min(D,d1)
    return (h0,h1,w0,w1,d0,d1)


def _largest_connected_component(mask_3d: torch.Tensor) -> torch.Tensor:
    """
    Keep only the largest 6-connected component for a single 3D binary mask.
    mask_3d: (H,W,D)
    """
    mask_np = (mask_3d > 0.5).detach().cpu().numpy()
    if not mask_np.any():
        return torch.zeros_like(mask_3d)

    if scipy_ndimage is not None:
        structure = scipy_ndimage.generate_binary_structure(rank=3, connectivity=1)
        labeled, num = scipy_ndimage.label(mask_np, structure=structure)
        if num == 0:
            return torch.zeros_like(mask_3d)
        counts = np.bincount(labeled.ravel())[1:]
        if counts.size == 0:
            return torch.zeros_like(mask_3d)
        largest_label = int(np.argmax(counts)) + 1
        out_np = (labeled == largest_label)
        return torch.from_numpy(out_np).to(device=mask_3d.device, dtype=mask_3d.dtype)

    H, W, D = mask_np.shape
    visited = torch.zeros((H, W, D), dtype=torch.bool)
    best_coords = []
    neighbors = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    fg = mask_np.nonzero()
    for h, w, d in zip(fg[0], fg[1], fg[2]):
        if visited[h, w, d]:
            continue
        q = deque([(int(h), int(w), int(d))])
        visited[h, w, d] = True
        coords = []
        while q:
            ch, cw, cd = q.popleft()
            coords.append((ch, cw, cd))
            for dh, dw, dd in neighbors:
                nh, nw, nd = ch + dh, cw + dw, cd + dd
                if nh < 0 or nh >= H or nw < 0 or nw >= W or nd < 0 or nd >= D:
                    continue
                if visited[nh, nw, nd] or not mask_np[nh, nw, nd]:
                    continue
                visited[nh, nw, nd] = True
                q.append((nh, nw, nd))
        if len(coords) > len(best_coords):
            best_coords = coords

    out = torch.zeros_like(mask_3d)
    if best_coords:
        hh, ww, dd = zip(*best_coords)
        out[list(hh), list(ww), list(dd)] = 1.0
    return out


def _connected_components(mask_3d: torch.Tensor) -> List[torch.Tensor]:
    """
    Return all 6-connected foreground components for a single 3D binary mask.
    Components are sorted by size descending.
    """
    mask_np = (mask_3d > 0.5).detach().cpu().numpy()
    if not mask_np.any():
        return []

    if scipy_ndimage is not None:
        structure = scipy_ndimage.generate_binary_structure(rank=3, connectivity=1)
        labeled, num = scipy_ndimage.label(mask_np, structure=structure)
        if num == 0:
            return []
        counts = np.bincount(labeled.ravel())[1:]
        if counts.size == 0:
            return []
        labels = np.argsort(counts)[::-1] + 1
        return [
            torch.from_numpy(labeled == lab).to(device=mask_3d.device, dtype=mask_3d.dtype)
            for lab in labels
        ]

    H, W, D = mask_np.shape
    visited = torch.zeros((H, W, D), dtype=torch.bool)
    neighbors = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    comps: List[List[Tuple[int, int, int]]] = []

    fg = mask_np.nonzero()
    for h, w, d in zip(fg[0], fg[1], fg[2]):
        if visited[h, w, d]:
            continue
        q = deque([(int(h), int(w), int(d))])
        visited[h, w, d] = True
        coords = []
        while q:
            ch, cw, cd = q.popleft()
            coords.append((ch, cw, cd))
            for dh, dw, dd in neighbors:
                nh, nw, nd = ch + dh, cw + dw, cd + dd
                if nh < 0 or nh >= H or nw < 0 or nw >= W or nd < 0 or nd >= D:
                    continue
                if visited[nh, nw, nd] or not mask_np[nh, nw, nd]:
                    continue
                visited[nh, nw, nd] = True
                q.append((nh, nw, nd))
        comps.append(coords)

    comps.sort(key=len, reverse=True)
    out = []
    for coords in comps:
        comp = torch.zeros_like(mask_3d)
        hh, ww, dd = zip(*coords)
        comp[list(hh), list(ww), list(dd)] = 1.0
        out.append(comp)
    return out


def _binary_iou(mask_a: torch.Tensor, mask_b: torch.Tensor, eps: float = 1e-6) -> float:
    a = (mask_a > 0.5).float()
    b = (mask_b > 0.5).float()
    inter = (a * b).sum()
    union = a.sum() + b.sum() - inter
    return float((inter / (union + eps)).item())


def _greedy_match_by_iou(pred_masks: torch.Tensor, gt_components: List[torch.Tensor]) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    pred_masks: (Np,H,W,D) probabilities for one organ
    gt_components: list of (H,W,D) binary masks
    """
    if pred_masks.shape[0] == 0:
        return [], [], list(range(len(gt_components)))
    if len(gt_components) == 0:
        return [], list(range(pred_masks.shape[0])), []

    candidates: List[Tuple[float, int, int]] = []
    for pi in range(pred_masks.shape[0]):
        for gi, gt_comp in enumerate(gt_components):
            iou = _binary_iou(pred_masks[pi], gt_comp.to(pred_masks.device))
            if iou > 0.0:
                candidates.append((iou, pi, gi))

    candidates.sort(key=lambda x: x[0], reverse=True)
    used_pred = set()
    used_gt = set()
    matches: List[Tuple[int, int, float]] = []
    for iou, pi, gi in candidates:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((pi, gi, iou))

    unmatched_pred = [pi for pi in range(pred_masks.shape[0]) if pi not in used_pred]
    unmatched_gt = [gi for gi in range(len(gt_components)) if gi not in used_gt]
    return matches, unmatched_pred, unmatched_gt


def _report_volume_sort_key(report_volumes: torch.Tensor, lesion_idx: int) -> Tuple[int, float]:
    if lesion_idx >= int(report_volumes.numel()):
        return (0, float("-inf"))
    v = float(report_volumes[lesion_idx].item())
    if math.isfinite(v):
        return (1, v)
    return (0, float("-inf"))


def _downsample_binary_mask(mask_full: torch.Tensor, out_size: Tuple[int, int, int]) -> torch.Tensor:
    mask_full = (mask_full > 0.5).float()
    if mask_full.dim() == 3:
        mask_full = mask_full.unsqueeze(0).unsqueeze(0)
    pool_ks = tuple(
        max(int(math.ceil(mask_full.shape[2 + ax] / float(out_size[ax]))), 1)
        for ax in range(3)
    )
    mask_ds = F.max_pool3d(mask_full, kernel_size=pool_ks, stride=pool_ks, ceil_mode=True)
    if mask_ds.shape[2:] != out_size:
        mask_ds = F.interpolate(mask_ds, size=out_size, mode="nearest")
    return (mask_ds > 0).float()


def _seed_connected_component(mask_3d: torch.Tensor, seed_flat_idx: int) -> torch.Tensor:
    mask_bool = mask_3d.bool()
    if not mask_bool.any():
        return mask_3d.new_zeros(mask_3d.shape)
    seed = torch.zeros_like(mask_bool)
    seed.view(-1)[int(seed_flat_idx)] = True
    if not mask_bool.view(-1)[int(seed_flat_idx)]:
        return seed.float()

    comp = seed
    max_iter = int(sum(mask_bool.shape))
    for _ in range(max_iter):
        grown = F.max_pool3d(comp.float().unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)[0, 0] > 0
        new_comp = grown & mask_bool
        if torch.equal(new_comp, comp):
            break
        comp = new_comp
    return comp.float()


def _grow_region_to_target(
    region_3d: torch.Tensor,
    score_3d: torch.Tensor,
    valid_3d: torch.Tensor,
    target_count: int,
) -> torch.Tensor:
    region = region_3d.bool().clone()
    valid = valid_3d.bool()
    target = max(1, min(int(target_count), int(valid.sum().item())))
    if target <= 0:
        return region_3d.new_zeros(region_3d.shape)

    if not region.any():
        flat_scores = score_3d.reshape(-1).masked_fill(~valid.reshape(-1), -1e4)
        best_idx = int(flat_scores.argmax().item())
        region.view(-1)[best_idx] = True

    max_iter = int(sum(valid.shape))
    for _ in range(max_iter):
        curr = int(region.sum().item())
        if curr >= target:
            break
        frontier = (
            F.max_pool3d(region.float().unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)[0, 0] > 0
        ) & valid & (~region)
        if not frontier.any():
            frontier = valid & (~region)
            if not frontier.any():
                break
        frontier_flat = frontier.reshape(-1)
        frontier_scores = score_3d.reshape(-1).masked_fill(~frontier_flat, -1e4)
        add_k = min(target - curr, int(frontier_flat.sum().item()))
        add_idx = torch.topk(frontier_scores, k=add_k, largest=True, sorted=False).indices
        region.view(-1)[add_idx] = True
    return region.float()


def _mask_center_size(mask_3d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    idx = (mask_3d > 0.5).nonzero(as_tuple=False)
    if idx.numel() == 0:
        zero = mask_3d.new_zeros(3)
        return zero, zero, 0.0
    mins = idx.min(dim=0).values.float()
    maxs = idx.max(dim=0).values.float()
    H, W, D = mask_3d.shape
    center_norm = mask_3d.new_tensor([max(H - 1, 1), max(W - 1, 1), max(D - 1, 1)]).float()
    size_norm = mask_3d.new_tensor([max(H, 1), max(W, 1), max(D, 1)]).float()
    center = ((mins + maxs) * 0.5) / center_norm
    size = (maxs - mins + 1.0) / size_norm
    return center, size, 1.0


def _candidate_component_quality(
    cand_masks_feat: torch.Tensor,
    gt_comp_feat: torch.Tensor,
    dilate_kernel: int = 5,
    eps: float = 1e-6,
) -> float:
    """
    cand_masks_feat: (K,h,w,d)
    gt_comp_feat:    (h,w,d)
    Returns the best candidate-vs-component quality for assignment.
    """
    if cand_masks_feat.numel() == 0:
        return 0.0

    gt = (gt_comp_feat > 0.5).float()
    if gt.sum().item() <= 0:
        return 0.0

    if dilate_kernel > 1:
        pad = dilate_kernel // 2
        gt_dil = F.max_pool3d(gt.unsqueeze(0).unsqueeze(0), kernel_size=dilate_kernel, stride=1, padding=pad)[0, 0]
    else:
        gt_dil = gt

    gt_dil_mass = float(gt_dil.sum().item())
    if gt_dil_mass <= 0:
        return 0.0

    gt_center, gt_size, gt_valid = _mask_center_size(gt_dil)
    best_quality = 0.0
    for k in range(cand_masks_feat.shape[0]):
        cand = (cand_masks_feat[k] > 0.5).float()
        cand_mass = float(cand.sum().item())
        if cand_mass <= 0:
            continue
        inter_dil = float((cand * gt_dil).sum().item())
        recall_dil = inter_dil / (gt_dil_mass + eps)
        inside_dil = inter_dil / (cand_mass + eps)

        cand_center, cand_size, cand_valid = _mask_center_size(cand)
        if gt_valid > 0.0 and cand_valid > 0.0:
            center_dist = torch.linalg.norm(cand_center - gt_center, ord=2).item() / math.sqrt(3.0)
            center_quality = math.exp(-4.0 * center_dist)
            size_gap = (torch.log(cand_size + 1e-3) - torch.log(gt_size + 1e-3)).abs().mean().item()
            size_quality = math.exp(-2.0 * size_gap)
        else:
            center_quality = 0.0
            size_quality = 0.0

        quality = 0.40 * recall_dil + 0.20 * inside_dil + 0.30 * center_quality + 0.10 * size_quality
        best_quality = max(best_quality, quality)
    return float(best_quality)


def _optimal_assignment_max(weight: torch.Tensor) -> List[Tuple[int, int]]:
    """
    Maximum-weight bipartite matching for lesion/component sets.
    Returns (pred_idx, gt_idx) pairs.

    The previous recursive exact solver is exponential in the number of
    lesions/components. A single lesion-heavy crop could therefore stretch one
    training step from milliseconds to minutes and look like the whole run had
    frozen. Use a polynomial-time Hungarian solver when scipy is available, and
    fall back to a cheap greedy matcher for larger problems otherwise.
    """
    if weight.numel() == 0:
        return []
    weight = weight.detach().cpu().float()
    n_pred, n_gt = weight.shape
    if n_pred == 0 or n_gt == 0:
        return []

    if scipy_linear_sum_assignment is not None:
        weight_np = weight.numpy()
        try:
            rows, cols = scipy_linear_sum_assignment(weight_np, maximize=True)
        except TypeError:
            rows, cols = scipy_linear_sum_assignment(-weight_np)
        return [(int(pi), int(gi)) for pi, gi in zip(rows.tolist(), cols.tolist())]

    # Exact recursion is still fine for tiny problems when scipy is unavailable.
    if max(n_pred, n_gt) <= 8:
        if n_pred > n_gt:
            pairs_t = _optimal_assignment_max(weight.t())
            return [(pi, gi) for gi, pi in pairs_t]

        weight_list = weight.tolist()

        @lru_cache(maxsize=None)
        def _solve(pred_idx: int, used_mask: int) -> Tuple[float, Tuple[Tuple[int, int], ...]]:
            if pred_idx >= n_pred:
                return 0.0, tuple()

            best_score = float("-inf")
            best_pairs: Tuple[Tuple[int, int], ...] = tuple()
            for gt_idx in range(n_gt):
                if (used_mask >> gt_idx) & 1:
                    continue
                next_score, next_pairs = _solve(pred_idx + 1, used_mask | (1 << gt_idx))
                total = float(weight_list[pred_idx][gt_idx]) + next_score
                if total > best_score:
                    best_score = total
                    best_pairs = ((pred_idx, gt_idx),) + next_pairs
            return best_score, best_pairs

        _, pairs = _solve(0, 0)
        return list(pairs)

    # Greedy fallback keeps training moving even without scipy.
    edges: List[Tuple[float, int, int]] = []
    for pi in range(n_pred):
        for gi in range(n_gt):
            edges.append((float(weight[pi, gi].item()), pi, gi))
    edges.sort(key=lambda x: x[0], reverse=True)

    used_pred = set()
    used_gt = set()
    pairs: List[Tuple[int, int]] = []
    for _, pi, gi in edges:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        pairs.append((pi, gi))
        if len(used_pred) == n_pred or len(used_gt) == n_gt:
            break
    return pairs


def _assign_gt_components_to_lesions(
    ct: torch.Tensor,
    lesion_masks: Dict[str, torch.Tensor],
    organ_to_indices: Dict[str, List[int]],
    cand_masks_feat: torch.Tensor,
    feat_size: Tuple[int, int, int],
) -> List[torch.Tensor]:
    """
    Assign lesion nodes to GT connected components with optimal one-to-one
    matching inside each organ/support group. The matching score is based on the
    candidate pool geometry instead of report-volume sorting.
    """
    device = ct.device
    assigned = [torch.zeros_like(ct) for _ in range(int(cand_masks_feat.shape[1]))]
    for org, idxs in organ_to_indices.items():
        if len(idxs) == 0:
            continue
        gt_components = _connected_components(lesion_masks[org][0, 0])
        if len(gt_components) == 0:
            continue
        gt_components_feat = [
            _downsample_binary_mask(comp.to(device), feat_size)[0, 0]
            for comp in gt_components
        ]
        quality = ct.new_zeros((len(idxs), len(gt_components_feat)))
        for local_i, lesion_idx in enumerate(idxs):
            cand_i = cand_masks_feat[0, lesion_idx, :, 0]
            for gi, gt_comp_feat in enumerate(gt_components_feat):
                quality[local_i, gi] = _candidate_component_quality(cand_i, gt_comp_feat)

        if len(idxs) <= len(gt_components_feat):
            pairs_local = _optimal_assignment_max(quality)
        else:
            pairs_local = _optimal_assignment_max(quality.t())
            pairs_local = [(li, gi) for gi, li in pairs_local]

        for local_i, gi in pairs_local:
            lesion_idx = idxs[local_i]
            assigned[lesion_idx] = gt_components[gi].to(device).unsqueeze(0).unsqueeze(0)
    return assigned


def _select_support_mask(mask_dict: Dict[str, torch.Tensor], support_key: str) -> torch.Tensor:
    if support_key in mask_dict:
        return mask_dict[support_key]
    if support_key.startswith("kidney"):
        return mask_dict["kidney"]
    raise KeyError(f"Unknown support key: {support_key}")


class GLeVETrainModel(nn.Module):
    def __init__(
        self,
        d_text: int=768,
        M: int=8,
        topK: int=4,
        num_classes: int=1,
        base_chan: int=32,
        feat_dim: int=128,
        ver_hidden_dim: int=256,
        ocre_hidden_dim: int=128,
        oc_depth: int=3,
        oc_min_size: int=16,
        candidate_warmup_epochs: int=10,
        refine_ramp_epochs: int=5,
        physical_stats: Dict[str, Any] | None = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.M = M
        self.topK = topK
        self.candidate_warmup_epochs = max(int(candidate_warmup_epochs), 0)
        self.refine_ramp_epochs = max(int(refine_ramp_epochs), 1)
        self.physical_stats = dict(physical_stats or {})
        self.hu_min = float(self.physical_stats.get("hu_min", HU_MIN))
        self.hu_max = float(self.physical_stats.get("hu_max", HU_MAX))
        self.hu_loss_scale = float(self.physical_stats.get("report_hu_scale", max(self.hu_max - self.hu_min, 80.0)))
        self.candidate_voxel_ratios = (0.0015, 0.004, 0.010, 0.025, 0.050, 0.100, 0.160, 0.240)
        self.candidate_softmax_temp = 0.15

        self.lequ = LeQuEncoder(d=d_text, M=M, n_layers=2)

        # MedFormer3D backbone
        self.visual = MedFormer3D(
            in_chan=1,
            num_classes=num_classes,
            base_chan=base_chan,
            mid_channels=feat_dim,
            aux_loss=True,
            use_checkpoint=use_checkpoint,
        )

        # FiLM on mid features
        self.film = FiLMAnatomyModulation(C=feat_dim)

        self.q_proj = nn.Linear(d_text, feat_dim)

        # verification scorer
        self.ver_mlp = nn.Sequential(
            nn.Linear(feat_dim + d_text, ver_hidden_dim),
            nn.GELU(),
            nn.Linear(ver_hidden_dim, 1),
        )
        self.e_proj = nn.Linear(3, 1)

        # OcRe octree refiner
        self.ocre = OcReOctree(
            feat_channels=feat_dim,
            hidden_channels=ocre_hidden_dim,
            depth=oc_depth,
            min_size=oc_min_size,
            active_thr=0.005,
            final_highres_channels=base_chan * 2,
            boundary_channels=base_chan,
            boundary_band_width=1,
            boundary_refine_strength=0.35,
        )

    def _candidate_ratios_for_k(self, K: int) -> List[float]:
        base = list(self.candidate_voxel_ratios)
        if K <= 0:
            return []
        if K == 1:
            return [base[len(base) // 2]]
        if K <= len(base):
            idx = np.linspace(0, len(base) - 1, num=K)
            idx = np.round(idx).astype(np.int64).tolist()
            for i in range(1, len(idx)):
                idx[i] = max(idx[i], idx[i - 1] + 1)
            idx[-1] = min(idx[-1], len(base) - 1)
            for i in range(len(idx) - 2, -1, -1):
                idx[i] = min(idx[i], idx[i + 1] - 1)
            return [base[max(0, min(int(j), len(base) - 1))] for j in idx]
        return base + [base[-1]] * (K - len(base))

    def _cosine_sim(self, q: torch.Tensor, Fv: torch.Tensor) -> torch.Tensor:
        """
        q: (N,M,C), Fv: (B,C,h,w,d) -> S: (B,N,h,w,d)
        """
        B,C,h,w,d = Fv.shape
        N,M,_ = q.shape
        F_n = F.normalize(Fv, dim=1)
        q_n = F.normalize(q, dim=-1)

        S = []
        for i in range(N):
            qi = q_n[i]  # (M,C)
            sm = torch.einsum("mc,bchwd->bmhwd", qi, F_n)  # (B,M,h,w,d)
            si = sm.max(dim=1).values
            S.append(si)
        return torch.stack(S, dim=1)

    def _topk_candidate_masks(
        self,
        S: torch.Tensor,
        K: int,
        organ_priors: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        S: (B,N,h,w,d) -> hard/soft candidates: (B,N,K,1,h,w,d)
        """
        B,N,h,w,d = S.shape
        total_voxels = h * w * d
        ratios = self._candidate_ratios_for_k(K)
        while len(ratios) < K:
            ratios.append(ratios[-1] if ratios else 0.02)

        flat_scores = S.float().reshape(B, N, total_voxels)
        smooth_scores = F.avg_pool3d(S.view(B * N, 1, h, w, d), kernel_size=5, stride=1, padding=2).view(B, N, h, w, d)
        smooth_flat = smooth_scores.float().reshape(B, N, total_voxels)
        if organ_priors is not None:
            organ_flat = organ_priors.reshape(B, N, total_voxels) > 0.05
            support_count = organ_flat.sum(dim=-1)
            has_organ_support = support_count > 0
            masked_scores = torch.where(organ_flat, smooth_flat, smooth_flat.new_full(smooth_flat.shape, -1e4))
            score_source = torch.where(has_organ_support.unsqueeze(-1), masked_scores, smooth_flat)
            valid_source = torch.where(has_organ_support.unsqueeze(-1), organ_flat, torch.ones_like(organ_flat))
            support_count = torch.where(
                has_organ_support,
                support_count,
                torch.full_like(support_count, total_voxels),
            )
        else:
            score_source = smooth_flat
            valid_source = torch.ones_like(flat_scores, dtype=torch.bool)
            support_count = torch.full((B, N), total_voxels, device=S.device, dtype=torch.long)

        ratio_tensor = S.new_tensor(ratios, dtype=torch.float32).view(1, 1, K)
        eff_ks = torch.clamp(
            torch.round(support_count.unsqueeze(-1).float() * ratio_tensor).long(),
            min=1,
        )
        eff_ks = torch.minimum(eff_ks, support_count.unsqueeze(-1))
        for ki in range(1, K):
            can_grow = support_count > eff_ks[..., ki - 1]
            next_k = torch.maximum(eff_ks[..., ki], eff_ks[..., ki - 1] + can_grow.long())
            eff_ks[..., ki] = torch.minimum(next_k, support_count)
        hard = S.new_zeros((B, N, K, 1, h, w, d))
        soft = S.new_zeros((B, N, K, 1, h, w, d))

        grid_h = torch.linspace(-1.0, 1.0, h, device=S.device, dtype=S.dtype).view(h, 1, 1)
        grid_w = torch.linspace(-1.0, 1.0, w, device=S.device, dtype=S.dtype).view(1, w, 1)
        grid_d = torch.linspace(-1.0, 1.0, d, device=S.device, dtype=S.dtype).view(1, 1, d)
        for b in range(B):
            for n_idx in range(N):
                valid_flat = valid_source[b, n_idx]
                if not valid_flat.any():
                    continue
                smooth_bn = smooth_scores[b, n_idx]
                raw_flat = flat_scores[b, n_idx]
                seed_flat = int(score_source[b, n_idx].argmax().item())
                seed_h = seed_flat // (w * d)
                seed_w = (seed_flat // d) % w
                seed_d = seed_flat % d
                dist = torch.sqrt(
                    (grid_h - grid_h[seed_h]) ** 2
                    + (grid_w - grid_w[0, seed_w]) ** 2
                    + (grid_d - grid_d[0, 0, seed_d]) ** 2
                ) / math.sqrt(3.0)
                smooth_valid = smooth_bn.reshape(-1)[valid_flat]
                smooth_min = smooth_valid.min()
                smooth_max = smooth_valid.max()
                smooth_norm = (smooth_bn - smooth_min) / (smooth_max - smooth_min + 1e-6)

                valid_3d = valid_flat.reshape(h, w, d)
                for ki in range(K):
                    target_k = int(eff_ks[b, n_idx, ki].item())
                    if target_k <= 0:
                        continue
                    ratio = max(float(ratios[ki]), 1e-4)
                    sigma = max((3.0 * ratio) ** (1.0 / 3.0), 0.12)
                    spatial_bias = torch.exp(-0.5 * (dist / sigma) ** 2)
                    combined = smooth_norm + 0.35 * spatial_bias
                    combined_flat = combined.reshape(-1).masked_fill(~valid_flat, -1e4)

                    core_k = min(max(4, target_k // 4), int(valid_flat.sum().item()), target_k)
                    core_idx = torch.topk(combined_flat, k=core_k, largest=True, sorted=False).indices
                    core_region = torch.zeros_like(combined_flat, dtype=S.dtype)
                    core_region[core_idx] = 1.0
                    core_region = core_region.reshape(h, w, d)
                    core_region = (
                        F.max_pool3d(core_region.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)[0, 0] > 0
                    ).float()
                    core_region = _seed_connected_component(core_region * valid_3d.float(), seed_flat)
                    cand_region = _grow_region_to_target(core_region, combined, valid_3d, target_k)
                    cand_flat = cand_region.reshape(-1).bool()
                    hard[b, n_idx, ki, 0] = cand_region

                    soft_logits = (raw_flat / self.candidate_softmax_temp).masked_fill(~cand_flat, -1e4)
                    soft_w = torch.softmax(soft_logits, dim=-1).to(dtype=S.dtype) * cand_flat.to(dtype=S.dtype)
                    soft[b, n_idx, ki, 0] = soft_w.reshape(h, w, d)
        return hard, soft

    def _pool_region_feat(self, Fv: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
        w = region
        num = (Fv * w).sum(dim=(2,3,4))
        den = w.sum(dim=(2,3,4)) + 1e-6
        return num / den

    def forward(self, batch: Dict[str, Any], epoch: int = 1, return_vis: bool = False) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device

        ct = batch["ct"].to(device, non_blocking=True)  # (B,1,H,W,D)  B=1
        spacing = batch["spacing"].to(device, non_blocking=True)
        organ_masks = {k: v.to(device, non_blocking=True) for k,v in batch["organ_masks"].items()}
        lesion_masks = {k: v.to(device, non_blocking=True) for k,v in batch["lesion_masks"].items()}
        has_mask = batch["has_mask"].to(device, non_blocking=True)  # (B,1)

        graph = batch["graph"]
        text_emb = batch["text_emb"].to(device, non_blocking=True)  # (V,768)

        # ---- LeQu ----
        lesion_ids, z, q_text, support_of_lesion, targets = self.lequ(graph, text_emb)
        if z.shape[0] == 0:
            zero = ct.new_zeros(())
            return {
                "loss_total": zero,
                "loss_seg": zero,
                "loss_weak": zero,
                "valid": False,
                "num_lesions": 0,
            }

        q = self.q_proj(q_text)  # (N,M,128)

        candidate_only = epoch <= self.candidate_warmup_epochs
        refine_alpha = 0.0
        if not candidate_only:
            refine_alpha = min(
                max(epoch - self.candidate_warmup_epochs, 1),
                self.refine_ramp_epochs,
            ) / float(self.refine_ramp_epochs)
        report_term_alpha = 0.0
        if not candidate_only:
            stable_epoch = self.candidate_warmup_epochs + self.refine_ramp_epochs
            if epoch > stable_epoch:
                report_term_alpha = min(
                    max(epoch - stable_epoch, 1),
                    self.refine_ramp_epochs,
                ) / float(self.refine_ramp_epochs)

        # ---- Visual features ----
        # Use the encoder features directly for candidate localization and
        # refinement. The single-channel MedFormer decoder predicts only a global
        # lesionness map, which conflicts with per-lesion node training in
        # multi-lesion cases and was destabilizing optimization.
        feat_mid, visual_aux = self.visual.forward_features(ct)
        coarse_logits_full = None
        coarse_aux_logits_full = None
        coarse_prob_full = None
        feat_mid = self.film(feat_mid, organ_masks)
        feat_final = visual_aux["x1"]
        feat_boundary = visual_aux["x0"]

        # ---- Similarity response ----
        S = self._cosine_sim(q, feat_mid)  # (B,N,h,w,d)
        B, N = S.shape[0], S.shape[1]

        organ_to_indices: Dict[str, List[int]] = {}
        for i, support_key in enumerate(support_of_lesion):
            organ_to_indices.setdefault(support_key, []).append(i)

        # per-lesion organ prior (downsample to feature-res)
        organ_priors = []
        for i in range(N):
            support_key = support_of_lesion[i]
            om = _select_support_mask(organ_masks, support_key)
            omd = F.interpolate(om, size=feat_mid.shape[2:], mode="trilinear", align_corners=False)
            organ_priors.append(omd)
        organ_priors = torch.stack(organ_priors, dim=1)  # (B,N,1,h,w,d)

        # ---- Candidate generation ----
        cand, cand_soft = self._topk_candidate_masks(S, K=self.topK, organ_priors=organ_priors)  # (B,N,K,1,h,w,d)
        K = cand.shape[2]

        assigned_gt_full = [torch.zeros_like(ct) for _ in range(N)]
        if has_mask.item() > 0.5:
            assigned_masks = _assign_gt_components_to_lesions(
                ct=ct,
                lesion_masks=lesion_masks,
                organ_to_indices=organ_to_indices,
                cand_masks_feat=cand,
                feat_size=feat_mid.shape[2:],
            )
            for i in range(min(N, len(assigned_masks))):
                assigned_gt_full[i] = assigned_masks[i]

        gt_masks_feat = []
        for i in range(N):
            gt_masks_feat.append(_downsample_binary_mask(assigned_gt_full[i], feat_mid.shape[2:]))
        gt_masks_feat = torch.stack(gt_masks_feat, dim=1)  # (B,N,1,h,w,d)

        # ---- Region-level verification ----
        ver_scores = []
        ct_d = F.interpolate(ct, size=feat_mid.shape[2:], mode="trilinear", align_corners=False)
        for i in range(N):
                zi = z[i].view(1,-1).expand(B,-1)
                for k in range(K):
                    rk_hard = cand[:, i, k]      # (B,1,h,w,d)
                    rk_soft = cand_soft[:, i, k] # (B,1,h,w,d)
                    vk = self._pool_region_feat(feat_mid, rk_soft)  # (B,128)
                    vk = F.normalize(vk, dim=-1)
                    zi_n = F.normalize(zi, dim=-1)

                    cover = (rk_hard * organ_priors[:, i]).sum(dim=(1,2,3,4)) / (rk_hard.sum(dim=(1,2,3,4)) + 1e-6)
                    support_vol = organ_priors[:, i].sum(dim=(1,2,3,4)) + 1e-6
                    vol = rk_hard.sum(dim=(1,2,3,4)) / support_vol
                    mu = (rk_soft * ct_d).sum(dim=(1,2,3,4)) / (rk_soft.sum(dim=(1,2,3,4)) + 1e-6)
                    e = torch.stack([cover, vol, mu], dim=-1)  # (B,3)

                    raw_s = self.ver_mlp(torch.cat([vk, zi_n], dim=-1)).squeeze(-1) + self.e_proj(e).squeeze(-1)
                    s = 8.0 * torch.tanh(raw_s / 8.0)
                    ver_scores.append(s)
        ver_scores = torch.stack(ver_scores, dim=-1).view(B, N, K)

        k_star = ver_scores.argmax(dim=-1)  # (B,N)
        chosen = []
        for i in range(N):
            idxk = k_star[:, i]
            ri = []
            for b in range(B):
                chosen_mask = cand[b, i, idxk[b], 0]
                chosen_mask = _largest_connected_component(chosen_mask).to(device)
                ri.append(chosen_mask.unsqueeze(0).unsqueeze(0))
            chosen.append(torch.cat(ri, dim=0))
        chosen = torch.stack(chosen, dim=1)  # (B,N,1,h,w,d)  (feature-res mask)
        chosen_union = chosen.amax(dim=1)
        chosen_voxels = chosen_union.sum()
        chosen_iou = ct.new_zeros(())
        if has_mask.item() > 0.5:
            chosen_inter = (chosen * gt_masks_feat).sum(dim=(2, 3, 4, 5))
            chosen_union_den = chosen.sum(dim=(2, 3, 4, 5)) + gt_masks_feat.sum(dim=(2, 3, 4, 5)) - chosen_inter
            chosen_valid = (gt_masks_feat.sum(dim=(2, 3, 4, 5)) > 0).float()
            chosen_iou = (
                (chosen_inter / (chosen_union_den + 1e-6)) * chosen_valid
            ).sum() / (chosen_valid.sum() + 1e-6)

        gt_union_full = torch.maximum(
            torch.maximum(lesion_masks["kidney"], lesion_masks["pancreas"]),
            lesion_masks["liver"],
        ).float()

        # ---- Build full-res cond0 from chosen and refine with OcRe-Octree ----
        H,W,D = ct.shape[2:]
        pred_logits_full_list = []
        organ_masks_per_lesion_full = []
        chosen_full_list = []
        cond0_full_list = []
        refinement_enabled = not candidate_only
        mean_spacing_mm = float(spacing.view(B, -1).mean().item())
        seed_expand_vox = max(2, int(round(8.0 / max(mean_spacing_mm, 1e-3))))
        seed_expand_vox = min(seed_expand_vox, 12)
        seed_kernel = int(seed_expand_vox * 2 + 1)
        roi_pad = max(12, int(round(20.0 / max(mean_spacing_mm, 1e-3))))
        roi_pad = min(roi_pad, 32)

        for i in range(N):
            chosen_up = F.interpolate(chosen[:, i], size=(H,W,D), mode="trilinear", align_corners=False)
            chosen_full_list.append(chosen_up.clamp(0.0, 1.0))
            support_key = support_of_lesion[i]
            organ_prior_full = _select_support_mask(organ_masks, support_key)

            cond0_core = chosen_up.clamp(0.0, 1.0)
            cond0_expand = F.max_pool3d((cond0_core > 0.10).float(), kernel_size=seed_kernel, stride=1, padding=seed_expand_vox)
            cond0 = torch.maximum(cond0_core, 0.35 * cond0_expand)
            cond0_full_list.append(cond0)
            if refinement_enabled:
                roi_seed = F.max_pool3d((cond0 > 0.08).float(), kernel_size=seed_kernel, stride=1, padding=seed_expand_vox)
                roi = _bbox_from_mask(roi_seed, pad=roi_pad)
                logits_full = self.ocre(
                    feat_full=feat_mid,
                    cond0_full=cond0,
                    roi_box=roi,
                    out_size=(H,W,D),
                    final_feat_full=feat_final,
                    boundary_feat_full=feat_boundary,
                )
                pred_logits_full_list.append(logits_full)

            # organ prior full-res for L_org (match paper)
            organ_masks_per_lesion_full.append(organ_prior_full)

        chosen_full = torch.stack(chosen_full_list, dim=1)  # (B,N,1,H,W,D)
        chosen_union_full = (chosen_full > 0.5).float().amax(dim=1)
        cond0_full = torch.stack(cond0_full_list, dim=1)  # (B,N,1,H,W,D)
        cond0_union = (cond0_full > 0.5).float().amax(dim=1)
        cond0_voxels = cond0_union.sum()

        if refinement_enabled:
            pred_logits_full = torch.stack(pred_logits_full_list, dim=1)  # (B,N,1,H,W,D)
            pred_prob_full = torch.sigmoid(pred_logits_full)
        else:
            pred_prob_full = cond0_full.clamp(min=0.0, max=1.0)
            pred_logits_full = None
        pred_union = (pred_prob_full > 0.5).float().amax(dim=1)
        pred_voxels = pred_union.sum()
        pred_voxels_01 = (pred_prob_full > 0.1).float().sum()
        pred_mass = pred_prob_full.sum()
        cond0_iou = ct.new_zeros(())
        pred_union_iou = ct.new_zeros(())
        if has_mask.item() > 0.5:
            cond0_iou = ct.new_tensor(_binary_iou(cond0_union[0, 0], gt_union_full[0, 0]))
            pred_union_iou = ct.new_tensor(_binary_iou(pred_union[0, 0], gt_union_full[0, 0]))

        organ_masks_per_lesion_full = torch.stack(organ_masks_per_lesion_full, dim=1)  # (B,N,1,H,W,D)

        # ---- Weak loss (paper-aligned terms) ----
        weak_parts = {
            "Lver": ct.new_zeros(()),
            "Lv": ct.new_zeros(()),
            "Lv_under": ct.new_zeros(()),
            "Lmu": ct.new_zeros(()),
            "Lexcl": ct.new_zeros(()),
            "Lsmooth": ct.new_zeros(()),
            "Lcompact": ct.new_zeros(()),
            "Lorg": ct.new_zeros(()),
            "Lweak_base": ct.new_zeros(()),
            "Lweak_report": ct.new_zeros(()),
        }
        loss_weak_base = ct.new_zeros(())
        loss_weak_report = ct.new_zeros(())
        if candidate_only:
            loss_weak = ct.new_zeros(())
        else:
            targets_batched = {
                "V": targets["V"].view(1, -1),
                "mu": targets["mu"].view(1, -1),
            }
            loss_weak, weak_parts = weak_loss(
                ver_scores=ver_scores,
                pred_masks=pred_prob_full,
                ct=ct,
                organ_masks=organ_masks_per_lesion_full,
                report_targets=targets_batched,
                voxel_volume_cc=torch.prod(spacing.view(B, -1), dim=-1),
                hu_min=self.hu_min,
                hu_max=self.hu_max,
                hu_loss_scale=self.hu_loss_scale,
                report_term_alpha=report_term_alpha,
            )
            loss_weak_base = weak_parts["Lweak_base"]
            loss_weak_report = weak_parts["Lweak_report"]

        loss_loc = ct.new_zeros(())
        loss_loc_rank = ct.new_zeros(())
        loss_loc_dense = ct.new_zeros(())
        loc_best_iou = ct.new_zeros(())
        loc_best_recall = ct.new_zeros(())
        loc_best_fbeta = ct.new_zeros(())
        dense_loc_iou = ct.new_zeros(())
        if has_mask.item() > 0.5:
            loss_loc_rank, loc_parts = candidate_region_loss(
                scores=ver_scores,
                cand_masks=cand,
                gt_masks=gt_masks_feat,
            )
            loss_loc_dense, dense_loc_parts = dense_candidate_localization_loss(
                sim_scores=S,
                gt_masks=gt_masks_feat,
            )
            loss_loc = 0.15 * loss_loc_rank + 0.85 * loss_loc_dense
            loc_best_iou = loc_parts["best_iou"]
            loc_best_recall = loc_parts["best_recall"]
            loc_best_fbeta = loc_parts["best_fbeta"]
            dense_loc_iou = dense_loc_parts["dense_iou"]

        loss_seg = ct.new_zeros(())
        loss_seg_match = ct.new_zeros(())
        loss_seg_union = ct.new_zeros(())
        seg_parts = {
            "Ldice": ct.new_zeros(()),
            "Ltv": ct.new_zeros(()),
            "Lbce": ct.new_zeros(()),
            "Lrec": ct.new_zeros(()),
            "Lpre": ct.new_zeros(()),
            "Lover": ct.new_zeros(()),
            "Lfp": ct.new_zeros(()),
        }
        gt_voxels = gt_union_full.sum() if has_mask.item() > 0.5 else ct.new_zeros(())
        lesion_match_iou = ct.new_zeros(())
        matched_lesions = ct.new_zeros(())
        unmatched_pred_lesions = ct.new_zeros(())
        unmatched_gt_lesions = ct.new_zeros(())
        lesion_precision = ct.new_zeros(())
        lesion_recall = ct.new_zeros(())
        if (not candidate_only) and has_mask.item() > 0.5:
            gt_union = torch.zeros_like(ct)
            loss_seg = 0.0
            loss_seg_match = 0.0
            loss_seg_union = 0.0
            seg_parts_accum = {k: ct.new_zeros(()) for k in seg_parts}
            used_organs = 0
            matched_iou_sum = 0.0
            matched_count = 0
            unmatched_pred_count = 0
            unmatched_gt_count = 0

            for org, idxs in organ_to_indices.items():
                if len(idxs) == 0:
                    continue
                gt_org = _select_support_mask(lesion_masks, org)
                gt_union = torch.maximum(gt_union, gt_org)

                pred_org_nodes = pred_prob_full[:, idxs, 0]  # (B,n_org,H,W,D), B=1
                gt_components = _connected_components(gt_org[0, 0])
                matches, unmatched_pred, unmatched_gt = _greedy_match_by_iou(pred_org_nodes[0], gt_components)

                match_loss_sum = ct.new_zeros(())
                match_terms = 0
                for pi, gi, miou in matches:
                    gt_comp = gt_components[gi].to(device).unsqueeze(0).unsqueeze(0)
                    pred_prob_i = pred_prob_full[:, idxs[pi]]
                    if pred_logits_full is not None:
                        pred_logits_i = pred_logits_full[:, idxs[pi]]
                    else:
                        pred_logits_i = torch.logit(pred_prob_i.clamp(min=1e-4, max=1.0 - 1e-4))
                    seg_i, seg_i_parts = seg_loss(pred_logits_i, gt_comp)
                    match_loss_sum = match_loss_sum + seg_i
                    for k in seg_parts_accum:
                        seg_parts_accum[k] = seg_parts_accum[k] + seg_i_parts[k]
                    match_terms += 1
                    matched_iou_sum += miou
                    matched_count += 1

                for pi in unmatched_pred:
                    zero_gt = torch.zeros_like(gt_org)
                    pred_prob_i = pred_prob_full[:, idxs[pi]]
                    if pred_logits_full is not None:
                        pred_logits_i = pred_logits_full[:, idxs[pi]]
                    else:
                        pred_logits_i = torch.logit(pred_prob_i.clamp(min=1e-4, max=1.0 - 1e-4))
                    seg_i, seg_i_parts = seg_loss(pred_logits_i, zero_gt)
                    match_loss_sum = match_loss_sum + seg_i
                    for k in seg_parts_accum:
                        seg_parts_accum[k] = seg_parts_accum[k] + seg_i_parts[k]
                    match_terms += 1
                    unmatched_pred_count += 1

                for gi in unmatched_gt:
                    gt_comp = gt_components[gi].to(device).unsqueeze(0).unsqueeze(0)
                    inside_score = (pred_prob_full[:, idxs, 0] * gt_comp).sum(dim=(2, 3, 4))
                    inside_score = inside_score / (gt_comp.sum(dim=(2, 3, 4)) + 1e-6)
                    best_local = int(inside_score[0].argmax().item())
                    pred_prob_i = pred_prob_full[:, idxs[best_local]]
                    if pred_logits_full is not None:
                        pred_logits_i = pred_logits_full[:, idxs[best_local]]
                    else:
                        pred_logits_i = torch.logit(pred_prob_i.clamp(min=1e-4, max=1.0 - 1e-4))
                    seg_i, seg_i_parts = seg_loss(pred_logits_i, gt_comp)
                    match_loss_sum = match_loss_sum + seg_i
                    for k in seg_parts_accum:
                        seg_parts_accum[k] = seg_parts_accum[k] + seg_i_parts[k]
                    match_terms += 1

                unmatched_gt_count += len(unmatched_gt)
                if match_terms > 0:
                    loss_seg_match = loss_seg_match + (match_loss_sum / float(match_terms))

                pred_org_prob = pred_prob_full[:, idxs].amax(dim=1)
                pred_org_logits = torch.logit(pred_org_prob.clamp(min=1e-4, max=1.0 - 1e-4))
                seg_union_i, _ = seg_loss(pred_org_logits, gt_org)
                loss_seg_union = loss_seg_union + seg_union_i

                used_organs += 1

            gt_voxels = gt_union.sum()
            if used_organs > 0:
                loss_seg_match = loss_seg_match / float(used_organs)
                loss_seg_union = loss_seg_union / float(used_organs)
                loss_seg = 0.75 * loss_seg_match + 0.25 * loss_seg_union
                seg_terms = matched_count + unmatched_pred_count + unmatched_gt_count
                seg_parts = {k: v / float(max(seg_terms, 1)) for k, v in seg_parts_accum.items()}
            lesion_match_iou = ct.new_tensor(matched_iou_sum / max(matched_count, 1))
            matched_lesions = ct.new_tensor(float(matched_count))
            unmatched_pred_lesions = ct.new_tensor(float(unmatched_pred_count))
            unmatched_gt_lesions = ct.new_tensor(float(unmatched_gt_count))
            lesion_precision = ct.new_tensor(float(matched_count) / max(float(matched_count + unmatched_pred_count), 1.0))
            lesion_recall = ct.new_tensor(float(matched_count) / max(float(matched_count + unmatched_gt_count), 1.0))

        # total (paper): L_total = δ(λseg Lseg + λweak Lweak) + (1-δ) Lweak
        if candidate_only:
            if has_mask.item() > 0.5:
                loss_total = loss_loc
            else:
                loss_total = ver_scores.sum() * 0.0
        else:
            lam_loc = 0.20
            lam_seg = 0.78 * refine_alpha
            lam_weak_base = 0.03 * refine_alpha
            lam_weak_report = 0.10 * report_term_alpha
            delta = has_mask.view(-1).mean()
            loss_total = (
                delta * (
                    lam_loc * loss_loc
                    + lam_seg * loss_seg
                    + lam_weak_base * loss_weak_base
                    + lam_weak_report * loss_weak_report
                )
                + (1.0 - delta) * (
                    lam_weak_base * loss_weak_base
                    + lam_weak_report * loss_weak_report
                )
            )

        out = {
            "loss_total": loss_total,
            "loss_loc": loss_loc,
            "loss_loc_rank": loss_loc_rank,
            "loss_loc_dense": loss_loc_dense,
            "loss_seg": loss_seg,
            "loss_seg_match": loss_seg_match,
            "loss_seg_union": loss_seg_union,
            "loss_coarse": ct.new_zeros(()),
            "loss_coarse_aux": ct.new_zeros(()),
            "loss_weak": loss_weak,
            "loss_weak_base": loss_weak_base,
            "loss_weak_report": loss_weak_report,
            "loss_org": weak_parts["Lorg"],
            "loss_ver": weak_parts["Lver"],
            "loss_vol": weak_parts["Lv"],
            "loss_vol_under": weak_parts["Lv_under"],
            "loss_hu": weak_parts["Lmu"],
            "loss_smooth": weak_parts["Lsmooth"],
            "loss_compact": weak_parts["Lcompact"],
            "loc_best_iou": loc_best_iou.detach(),
            "loc_best_recall": loc_best_recall.detach(),
            "loc_best_fbeta": loc_best_fbeta.detach(),
            "dense_loc_iou": dense_loc_iou.detach(),
            "chosen_iou": chosen_iou.detach(),
            "cond0_iou": cond0_iou.detach(),
            "pred_union_iou": pred_union_iou.detach(),
            "loss_dice": seg_parts["Ldice"],
            "loss_tversky": seg_parts["Ltv"],
            "loss_bce": seg_parts["Lbce"],
            "loss_rec": seg_parts["Lrec"],
            "loss_pre": seg_parts["Lpre"],
            "loss_over": seg_parts["Lover"],
            "loss_fp": seg_parts["Lfp"],
            "chosen_voxels": chosen_voxels.detach(),
            "cond0_voxels": cond0_voxels.detach(),
            "pred_voxels": pred_voxels.detach(),
            "pred_voxels_01": pred_voxels_01.detach(),
            "pred_mass": pred_mass.detach(),
            "coarse_voxels": ct.new_zeros(()),
            "lesion_match_iou": lesion_match_iou.detach(),
            "matched_lesions": matched_lesions.detach(),
            "unmatched_pred_lesions": unmatched_pred_lesions.detach(),
            "unmatched_gt_lesions": unmatched_gt_lesions.detach(),
            "lesion_precision": lesion_precision.detach(),
            "lesion_recall": lesion_recall.detach(),
            "refine_on": ct.new_tensor(1.0 if refinement_enabled else 0.0),
            "refine_alpha": ct.new_tensor(refine_alpha),
            "report_term_alpha": ct.new_tensor(report_term_alpha),
            "candidate_only": ct.new_tensor(1.0 if candidate_only else 0.0),
            "gt_voxels": gt_voxels.detach(),
            "valid": True,
            "num_lesions": N,
        }
        if return_vis:
            pred_type_masks = []
            gt_type_masks = []
            for org in ["kidney", "pancreas", "liver"]:
                idxs = []
                for support_key, support_idxs in organ_to_indices.items():
                    if support_key == org or (org == "kidney" and support_key.startswith("kidney")):
                        idxs.extend(support_idxs)
                if len(idxs) > 0:
                    pred_org = (pred_prob_full[:, idxs].amax(dim=1) > 0.5).float()
                else:
                    pred_org = torch.zeros_like(ct)
                pred_type_masks.append(pred_org[0, 0].detach().cpu())
                gt_type_masks.append((lesion_masks[org][0, 0] > 0.5).float().detach().cpu())
            out["vis_ct"] = ct[0, 0].detach().cpu()
            out["vis_pred_type_masks"] = torch.stack(pred_type_masks, dim=0)
            out["vis_gt_type_masks"] = torch.stack(gt_type_masks, dim=0)
            out["vis_gt_union"] = gt_union_full[0, 0].detach().cpu()
            out["vis_chosen_union"] = chosen_union_full[0, 0].detach().cpu()
            out["vis_cond0_union"] = cond0_union[0, 0].detach().cpu()
            out["vis_pred_union"] = pred_union[0, 0].detach().cpu()
        return out
