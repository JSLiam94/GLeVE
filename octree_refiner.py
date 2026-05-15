from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _best_num_groups(num_channels: int, max_groups: int = 8) -> int:
    num_groups = min(max_groups, num_channels)
    while num_channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return max(num_groups, 1)


def _prob_to_logit(p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    p = p.clamp(min=eps, max=1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


@dataclass
class OctNode:
    h0: int
    h1: int
    w0: int
    w1: int
    d0: int
    d1: int
    level: int


def _clamp_box(h0, h1, w0, w1, d0, d1, H, W, D):
    h0 = max(0, min(h0, H))
    h1 = max(0, min(h1, H))
    w0 = max(0, min(w0, W))
    w1 = max(0, min(w1, W))
    d0 = max(0, min(d0, D))
    d1 = max(0, min(d1, D))
    return h0, h1, w0, w1, d0, d1


def _split_octants(node: OctNode) -> List[OctNode]:
    hm = (node.h0 + node.h1) // 2
    wm = (node.w0 + node.w1) // 2
    dm = (node.d0 + node.d1) // 2
    L = node.level + 1
    return [
        OctNode(node.h0, hm, node.w0, wm, node.d0, dm, L),
        OctNode(node.h0, hm, node.w0, wm, dm, node.d1, L),
        OctNode(node.h0, hm, wm, node.w1, node.d0, dm, L),
        OctNode(node.h0, hm, wm, node.w1, dm, node.d1, L),
        OctNode(hm, node.h1, node.w0, wm, node.d0, dm, L),
        OctNode(hm, node.h1, node.w0, wm, dm, node.d1, L),
        OctNode(hm, node.h1, wm, node.w1, node.d0, dm, L),
        OctNode(hm, node.h1, wm, node.w1, dm, node.d1, L),
    ]


class ResidualRefine3D(nn.Module):
    """Shared lightweight residual refinement head."""

    def __init__(self, C: int, hidden: int = 128):
        super().__init__()
        num_groups = _best_num_groups(hidden)
        self.net = nn.Sequential(
            nn.Conv3d(C + 1, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, hidden),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, hidden),
            nn.GELU(),
            nn.Conv3d(hidden, 1, 1),
        )

    def forward(self, feat: torch.Tensor, cond_mask: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([feat, cond_mask], dim=1))


class OcReOctree(nn.Module):
    """
    Efficient octree autoregressive refinement.

    The design balances voxel precision and throughput:
      - level-to-level autoregressive conditioning
      - batched refinement of same-shape active nodes within each level
      - higher-resolution encoder features on the final octree level
      - one lightweight boundary-band voxel correction pass at the end
    """

    def __init__(
        self,
        feat_channels: int = 128,
        hidden_channels: int = 128,
        depth: int = 3,
        min_size: int = 16,
        active_thr: float = 0.01,
        final_highres_channels: int = 0,
        boundary_channels: int = 0,
        boundary_band_width: int = 1,
        boundary_refine_strength: float = 0.5,
    ):
        super().__init__()
        self.depth = depth
        self.min_size = min_size
        self.active_thr = active_thr
        self.boundary_band_width = max(int(boundary_band_width), 0)
        self.boundary_refine_strength = float(max(boundary_refine_strength, 0.0))

        self.refiner = ResidualRefine3D(C=feat_channels, hidden=hidden_channels)

        self.final_proj = None
        if final_highres_channels > 0:
            self.final_proj = nn.Sequential(
                nn.Conv3d(final_highres_channels, feat_channels, 1, bias=False),
                nn.GroupNorm(_best_num_groups(feat_channels), feat_channels),
                nn.GELU(),
            )

        self.boundary_proj = None
        self.boundary_refiner = None
        if boundary_channels > 0:
            boundary_hidden = max(hidden_channels // 2, 32)
            self.boundary_proj = nn.Sequential(
                nn.Conv3d(boundary_channels, feat_channels, 1, bias=False),
                nn.GroupNorm(_best_num_groups(feat_channels), feat_channels),
                nn.GELU(),
            )
            self.boundary_refiner = ResidualRefine3D(C=feat_channels, hidden=boundary_hidden)

    def _node_feat_shape(
        self,
        node: OctNode,
        feat_shape: Tuple[int, int, int],
        out_size: Tuple[int, int, int],
    ) -> Tuple[int, int, int]:
        h, w, d = feat_shape
        H, W, D = out_size
        sh = h / float(H)
        sw = w / float(W)
        sd = d / float(D)
        fh0 = int(node.h0 * sh)
        fh1 = max(fh0 + 1, int(node.h1 * sh))
        fw0 = int(node.w0 * sw)
        fw1 = max(fw0 + 1, int(node.w1 * sw))
        fd0 = int(node.d0 * sd)
        fd1 = max(fd0 + 1, int(node.d1 * sd))
        return fh1 - fh0, fw1 - fw0, fd1 - fd0

    def _crop_feature(
        self,
        feat_full: torch.Tensor,
        node: OctNode,
        out_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        h, w, d = (int(v) for v in feat_full.shape[-3:])
        H, W, D = out_size
        sh = h / float(H)
        sw = w / float(W)
        sd = d / float(D)
        fh0 = int(node.h0 * sh)
        fh1 = max(fh0 + 1, int(node.h1 * sh))
        fw0 = int(node.w0 * sw)
        fw1 = max(fw0 + 1, int(node.w1 * sw))
        fd0 = int(node.d0 * sd)
        fd1 = max(fd0 + 1, int(node.d1 * sd))
        return feat_full[:, :, fh0:fh1, fw0:fw1, fd0:fd1]

    def _feature_pack_for_level(
        self,
        node: OctNode,
        level: int,
        feat_full: torch.Tensor,
        final_feat_full: torch.Tensor | None,
        parent_prob_full: torch.Tensor,
        out_size: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int], Tuple[int, int, int]]:
        use_final = (
            final_feat_full is not None
            and self.final_proj is not None
            and level == max(self.depth - 1, 0)
        )
        feat_source = final_feat_full if use_final else feat_full
        feat_crop = self._crop_feature(feat_source, node, out_size)
        if use_final:
            feat_crop = self.final_proj(feat_crop)

        cond_crop = parent_prob_full[:, :, node.h0:node.h1, node.w0:node.w1, node.d0:node.d1]
        cond_down = F.interpolate(cond_crop, size=feat_crop.shape[-3:], mode="trilinear", align_corners=False)
        full_shape = (node.h1 - node.h0, node.w1 - node.w0, node.d1 - node.d0)
        return feat_crop, cond_down, tuple(feat_crop.shape[-3:]), full_shape

    def _refine_level_batched(
        self,
        level: int,
        nodes: List[OctNode],
        feat_full: torch.Tensor,
        final_feat_full: torch.Tensor | None,
        parent_prob_full: torch.Tensor,
        out_size: Tuple[int, int, int],
    ) -> List[Tuple[OctNode, torch.Tensor, Tuple[int, int, int]]]:
        if not nodes:
            return []

        buckets: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], List[Tuple[OctNode, torch.Tensor, torch.Tensor]]] = {}
        feat_shape_map: Dict[Tuple[int, int, int, int, int, int, int], Tuple[int, int, int]] = {}

        for node in nodes:
            feat_crop, cond_down, feat_shape, full_shape = self._feature_pack_for_level(
                node=node,
                level=level,
                feat_full=feat_full,
                final_feat_full=final_feat_full,
                parent_prob_full=parent_prob_full,
                out_size=out_size,
            )
            key = (feat_shape, full_shape)
            buckets.setdefault(key, []).append((node, feat_crop, cond_down))
            feat_shape_map[(node.h0, node.h1, node.w0, node.w1, node.d0, node.d1, node.level)] = feat_shape

        results: List[Tuple[OctNode, torch.Tensor, Tuple[int, int, int]]] = []
        for key, entries in buckets.items():
            feat_batch = torch.cat([entry[1] for entry in entries], dim=0)
            cond_batch = torch.cat([entry[2] for entry in entries], dim=0)
            logits_batch = _prob_to_logit(cond_batch) + self.refiner(feat_batch, cond_batch)

            full_shape = key[1]
            if tuple(logits_batch.shape[-3:]) != full_shape:
                logits_batch = F.interpolate(logits_batch, size=full_shape, mode="trilinear", align_corners=False)

            for idx, (node, _, _) in enumerate(entries):
                node_key = (node.h0, node.h1, node.w0, node.w1, node.d0, node.d1, node.level)
                results.append((node, logits_batch[idx:idx + 1], feat_shape_map[node_key]))
        return results

    def _boundary_band(self, prob_crop: torch.Tensor) -> torch.Tensor:
        hard = (prob_crop > 0.5).float()
        width = self.boundary_band_width
        if width <= 0:
            return torch.zeros_like(hard)
        kernel = 2 * width + 1
        dilated = F.max_pool3d(hard, kernel_size=kernel, stride=1, padding=width)
        eroded = 1.0 - F.max_pool3d(1.0 - hard, kernel_size=kernel, stride=1, padding=width)
        return (dilated - eroded).clamp(min=0.0, max=1.0)

    def _apply_boundary_refine(
        self,
        out_logits: torch.Tensor,
        boundary_feat_full: torch.Tensor | None,
        roi: OctNode,
        out_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        if (
            boundary_feat_full is None
            or self.boundary_proj is None
            or self.boundary_refiner is None
            or self.boundary_refine_strength <= 0.0
            or self.boundary_band_width <= 0
        ):
            return out_logits

        prob_full = torch.sigmoid(out_logits)
        prob_crop = prob_full[:, :, roi.h0:roi.h1, roi.w0:roi.w1, roi.d0:roi.d1]
        band = self._boundary_band(prob_crop)
        if float(band.max().item()) <= 0.0:
            return out_logits

        feat_crop = self._crop_feature(boundary_feat_full, roi, out_size)
        feat_crop = self.boundary_proj(feat_crop)
        if tuple(feat_crop.shape[-3:]) != tuple(prob_crop.shape[-3:]):
            feat_crop = F.interpolate(feat_crop, size=prob_crop.shape[-3:], mode="trilinear", align_corners=False)

        delta = self.boundary_refiner(feat_crop, prob_crop)
        refined_crop = out_logits[:, :, roi.h0:roi.h1, roi.w0:roi.w1, roi.d0:roi.d1]
        refined_crop = refined_crop + self.boundary_refine_strength * delta * band
        out_logits[:, :, roi.h0:roi.h1, roi.w0:roi.w1, roi.d0:roi.d1] = refined_crop
        return out_logits

    def forward(
        self,
        feat_full: torch.Tensor,
        cond0_full: torch.Tensor,
        roi_box: Tuple[int, int, int, int, int, int],
        out_size: Tuple[int, int, int],
        final_feat_full: torch.Tensor | None = None,
        boundary_feat_full: torch.Tensor | None = None,
    ) -> torch.Tensor:
        H, W, D = out_size
        h0, h1, w0, w1, d0, d1 = roi_box
        h0, h1, w0, w1, d0, d1 = _clamp_box(h0, h1, w0, w1, d0, d1, H, W, D)
        root = OctNode(h0, h1, w0, w1, d0, d1, level=0)

        out_logits = _prob_to_logit(cond0_full.float())
        parent_pred = cond0_full.float()
        nodes = [root]
        feat_shape = tuple(int(x) for x in feat_full.shape[-3:])

        for level in range(self.depth):
            active_nodes: List[OctNode] = []
            for node in nodes:
                occ_prior = parent_pred[:, :, node.h0:node.h1, node.w0:node.w1, node.d0:node.d1].mean().item()
                if occ_prior >= self.active_thr:
                    active_nodes.append(node)
            if not active_nodes:
                break

            refined_nodes = self._refine_level_batched(
                level=level,
                nodes=active_nodes,
                feat_full=feat_full,
                final_feat_full=final_feat_full,
                parent_prob_full=parent_pred,
                out_size=out_size,
            )

            next_nodes: List[OctNode] = []
            for node, node_logits, node_feat_shape in refined_nodes:
                out_logits[:, :, node.h0:node.h1, node.w0:node.w1, node.d0:node.d1] = node_logits

                feature_too_small = min(node_feat_shape) <= 1
                node_is_small = (
                    (node.h1 - node.h0) <= self.min_size
                    and (node.w1 - node.w0) <= self.min_size
                    and (node.d1 - node.d0) <= self.min_size
                )
                if feature_too_small or node_is_small:
                    continue

                if level >= self.depth - 1:
                    continue

                node_prob = torch.sigmoid(node_logits).mean().item()
                if node_prob < self.active_thr:
                    continue
                next_nodes.extend(_split_octants(node))

            if not next_nodes:
                break

            parent_pred = torch.sigmoid(out_logits)
            nodes = next_nodes

        _ = feat_shape  # documents that traversal resolution still bounds splitting
        out_logits = self._apply_boundary_refine(
            out_logits=out_logits,
            boundary_feat_full=boundary_feat_full,
            roi=root,
            out_size=out_size,
        )
        return out_logits
