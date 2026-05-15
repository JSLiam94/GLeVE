from __future__ import annotations

import argparse
import csv
import math
import os
import random
from contextlib import nullcontext
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from dataset_abdomenatlas import (
    LESION_IDS,
    ORGAN_IDS,
    _lesion_support_key,
    _limit_graph_lesions,
    _load_graph_json,
    _load_text_embedding_cached,
    _make_binary,
)
from gleve_model import GLeVETrainModel, load_model_state_flexible
from utils_nii import hu_window_and_normalize, load_nii, load_nii_with_spacing

try:
    from scipy import ndimage as ndi
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    raise ImportError(
        "test_gleve.py requires scipy for HD95/ASSD/CCLS/Lesion_recall. "
        "Please install scipy in this environment."
    ) from exc


CLASS_NAMES = ("kidney", "pancreas", "liver")
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
SUPPORT_INFERENCE_SPECS = (
    ("kidney_right", "kidney", ("kidney", "kidney_right")),
    ("kidney_left", "kidney", ("kidney", "kidney_left")),
    ("pancreas", "pancreas", ("pancreas",)),
    ("liver", "liver", ("liver",)),
)
WORST_SURFACE_DISTANCE_MM = 373.13
METRIC_FALLBACKS = {
    "dice": 0.0,
    "hd95": WORST_SURFACE_DISTANCE_MM,
    "assd": WORST_SURFACE_DISTANCE_MM,
    "ccls": 0.0,
    "lesion_recall": 0.0,
}
DEFAULT_SEED = 2026


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cc_structure(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        return ndi.generate_binary_structure(3, 1)
    if connectivity in (18, 26):
        return np.ones((3, 3, 3), dtype=bool)
    raise ValueError(f"Unsupported connectivity: {connectivity}")


def component_masks(bin_mask: np.ndarray, connectivity: int, min_voxels: int = 0) -> list[np.ndarray]:
    labeled, n_comp = ndi.label(bin_mask.astype(bool), structure=cc_structure(connectivity))
    out: list[np.ndarray] = []
    for cid in range(1, n_comp + 1):
        comp = labeled == cid
        if min_voxels > 0 and int(np.count_nonzero(comp)) < min_voxels:
            continue
        out.append(comp)
    return out


def centroid_mm(mask: np.ndarray, spacing: Iterable[float] | None) -> np.ndarray:
    spacing_xyz = [1.0, 1.0, 1.0]
    if spacing is not None:
        spacing_xyz = [float(v) for v in spacing]
    vox = np.argwhere(mask)
    if vox.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=float)
    c_zyx = vox.mean(axis=0)
    return np.array(
        [
            c_zyx[2] * spacing_xyz[0],
            c_zyx[1] * spacing_xyz[1],
            c_zyx[0] * spacing_xyz[2],
        ],
        dtype=float,
    )


def dice_mask(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = int(np.count_nonzero(a & b))
    av = int(np.count_nonzero(a))
    bv = int(np.count_nonzero(b))
    den = av + bv
    if den == 0:
        return 1.0
    return (2.0 * inter) / den


def surface_metrics_binary(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Iterable[float],
) -> tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0, 0.0
    if not pred.any() or not gt.any():
        return WORST_SURFACE_DISTANCE_MM, WORST_SURFACE_DISTANCE_MM

    footprint = ndi.generate_binary_structure(rank=3, connectivity=1)
    pred_surface = np.logical_xor(
        pred,
        ndi.binary_erosion(pred, structure=footprint, border_value=0),
    )
    gt_surface = np.logical_xor(
        gt,
        ndi.binary_erosion(gt, structure=footprint, border_value=0),
    )
    if not pred_surface.any() or not gt_surface.any():
        return WORST_SURFACE_DISTANCE_MM, WORST_SURFACE_DISTANCE_MM

    spacing = tuple(float(v) for v in spacing)
    dt_to_gt = ndi.distance_transform_edt(~gt_surface, sampling=spacing)
    dt_to_pred = ndi.distance_transform_edt(~pred_surface, sampling=spacing)
    dist_pred_to_gt = dt_to_gt[pred_surface]
    dist_gt_to_pred = dt_to_pred[gt_surface]

    hd95 = max(
        float(np.percentile(dist_pred_to_gt, 95.0)),
        float(np.percentile(dist_gt_to_pred, 95.0)),
    )
    assd = 0.5 * (float(dist_pred_to_gt.mean()) + float(dist_gt_to_pred.mean()))
    return hd95, assd


def compute_ccls(
    gt_bin: np.ndarray,
    pred_bin: np.ndarray,
    spacing: Iterable[float] | None,
    connectivity: int = 26,
    min_pred_voxels: int = 60,
    dice_match_threshold: float = 0.01,
    d0_mm: float = 20.0,
) -> tuple[float, float]:
    gt_comps = component_masks(gt_bin, connectivity=connectivity, min_voxels=1)
    pred_comps = component_masks(pred_bin, connectivity=connectivity, min_voxels=min_pred_voxels)
    n_gt = len(gt_comps)
    n_pred = len(pred_comps)
    if n_gt == 0:
        return (1.0, 1.0) if n_pred == 0 else (0.0, 0.0)
    if n_pred == 0:
        return 0.0, 0.0

    gt_cent = [centroid_mm(comp, spacing) for comp in gt_comps]
    pred_cent = [centroid_mm(comp, spacing) for comp in pred_comps]

    dice_mat = np.zeros((n_gt, n_pred), dtype=float)
    score_mat = np.zeros((n_gt, n_pred), dtype=float)
    for i, gt_comp in enumerate(gt_comps):
        for j, pred_comp in enumerate(pred_comps):
            dice = dice_mask(gt_comp, pred_comp)
            dice_mat[i, j] = dice
            if dice < dice_match_threshold:
                continue
            dist = float(np.linalg.norm(gt_cent[i] - pred_cent[j]))
            score_mat[i, j] = dice * np.exp(-dist / max(d0_mm, 1e-6))

    row_idx, col_idx = linear_sum_assignment(-score_mat)
    gt_scores = np.zeros(n_gt, dtype=float)
    matched_gt = 0
    for r, c in zip(row_idx.tolist(), col_idx.tolist()):
        if score_mat[r, c] <= 0:
            continue
        if dice_mat[r, c] < dice_match_threshold:
            continue
        gt_scores[r] = score_mat[r, c]
        matched_gt += 1

    ccls = float(np.mean(gt_scores))
    lesion_recall = float(matched_gt / n_gt)
    return ccls, lesion_recall


def sanitize_metric(metric_name: str, value: Any) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float(METRIC_FALLBACKS[metric_name])
    if math.isfinite(numeric):
        return numeric
    return float(METRIC_FALLBACKS[metric_name])


def load_test_case_ids(test_csv: str, exclude_ids_path: str = "") -> list[str]:
    with open(test_csv, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return []

    first = rows[0]
    if "BDMAP ID" in first:
        key = "BDMAP ID"
    elif "BDMAP_ID" in first:
        key = "BDMAP_ID"
    else:
        raise ValueError("test_csv must contain column 'BDMAP ID' or 'BDMAP_ID'.")

    ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        case_id = str(row.get(key, "")).strip()
        if not case_id or case_id in seen:
            continue
        ids.append(case_id)
        seen.add(case_id)

    if exclude_ids_path:
        with open(exclude_ids_path, "r", encoding="utf-8") as f:
            exclude_ids = {line.strip() for line in f if line.strip()}
        ids = [case_id for case_id in ids if case_id not in exclude_ids]
    return ids


def build_crop_meta(
    original_shape: tuple[int, int, int],
    patch_shape: tuple[int, int, int],
    full_start: tuple[int, int, int],
    full_end: tuple[int, int, int],
    patch_start: tuple[int, int, int],
    patch_end: tuple[int, int, int],
    mode: str,
) -> Dict[str, Any]:
    return {
        "mode": str(mode),
        "original_shape": [int(v) for v in original_shape],
        "patch_shape": [int(v) for v in patch_shape],
        "full_start": [int(v) for v in full_start],
        "full_end": [int(v) for v in full_end],
        "patch_start": [int(v) for v in patch_start],
        "patch_end": [int(v) for v in patch_end],
    }


def crop_or_pad_from_start_with_meta(
    x: np.ndarray,
    start: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> tuple[np.ndarray, Dict[str, Any]]:
    h, w, d = x.shape
    ph, pw, pd = patch_size
    sh = min(max(int(start[0]), 0), max(h - ph, 0))
    sw = min(max(int(start[1]), 0), max(w - pw, 0))
    sd = min(max(int(start[2]), 0), max(d - pd, 0))

    eh = min(sh + ph, h)
    ew = min(sw + pw, w)
    ed = min(sd + pd, d)

    out = np.zeros(patch_size, dtype=x.dtype)
    out[:eh - sh, :ew - sw, :ed - sd] = x[sh:eh, sw:ew, sd:ed]
    meta = build_crop_meta(
        original_shape=(h, w, d),
        patch_shape=patch_size,
        full_start=(sh, sw, sd),
        full_end=(eh, ew, ed),
        patch_start=(0, 0, 0),
        patch_end=(eh - sh, ew - sw, ed - sd),
        mode="start",
    )
    return out, meta


def generate_crop_starts_covering_mask(
    mask: np.ndarray,
    patch_size: tuple[int, int, int],
    overlap_ratio: float,
) -> list[tuple[int, int, int]]:
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return []

    overlap_ratio = min(max(float(overlap_ratio), 0.0), 0.95)
    h, w, d = mask.shape
    mins = idx.min(axis=0)
    maxs = idx.max(axis=0) + 1

    def axis_starts(lo: int, hi: int, crop: int, size: int) -> list[int]:
        max_start = max(size - crop, 0)
        if hi - lo <= crop:
            center = (lo + hi) // 2
            return [max(min(center - crop // 2, max_start), 0)]

        step = max(1, int(round(crop * (1.0 - overlap_ratio))))
        first = max(min(lo, max_start), 0)
        last = max(min(hi - crop, max_start), 0)
        starts = list(range(first, last + 1, step))
        if not starts:
            starts = [first]
        if starts[-1] != last:
            starts.append(last)
        return sorted(set(int(v) for v in starts))

    starts_h = axis_starts(int(mins[0]), int(maxs[0]), int(patch_size[0]), int(h))
    starts_w = axis_starts(int(mins[1]), int(maxs[1]), int(patch_size[1]), int(w))
    starts_d = axis_starts(int(mins[2]), int(maxs[2]), int(patch_size[2]), int(d))
    return [(sh, sw, sd) for sh, sw, sd in product(starts_h, starts_w, starts_d)]


def filter_graph_by_supports(
    graph: Dict[str, Any],
    text_emb: np.ndarray,
    allowed_supports: Iterable[str],
) -> tuple[Dict[str, Any], np.ndarray]:
    allowed = {str(v) for v in allowed_supports}
    nodes = graph["nodes"]

    keep_lesion_ids: set[str] = set()
    keep_organs: set[str] = set()
    for node in nodes:
        if str(node.get("type")) != "lesion":
            continue
        if _lesion_support_key(node) not in allowed:
            continue
        lesion_id = str(node.get("lesion_id") or node["id"])
        keep_lesion_ids.add(lesion_id)
        if node.get("organ") is not None:
            keep_organs.add(str(node.get("organ")))

    keep_node_ids: set[str] = set()
    keep_indices: list[int] = []
    for idx, node in enumerate(nodes):
        node_id = str(node["id"])
        node_type = str(node.get("type"))
        lesion_id = str(node.get("lesion_id") or node_id)
        if node_type == "lesion" and lesion_id in keep_lesion_ids:
            keep_node_ids.add(node_id)
            keep_indices.append(idx)
        elif node_type == "attr" and lesion_id in keep_lesion_ids:
            keep_node_ids.add(node_id)
            keep_indices.append(idx)
        elif node_type == "organ" and str(node.get("organ")) in keep_organs:
            keep_node_ids.add(node_id)
            keep_indices.append(idx)

    filtered_nodes = [nodes[i] for i in keep_indices]
    filtered_edges = [
        e for e in graph.get("edges", [])
        if str(e.get("source")) in keep_node_ids and str(e.get("target")) in keep_node_ids
    ]
    filtered_graph = {
        "case_id": graph.get("case_id"),
        "nodes": filtered_nodes,
        "edges": filtered_edges,
        "meta": dict(graph.get("meta", {})),
    }
    filtered_emb = text_emb[keep_indices] if keep_indices else text_emb[:0]
    return filtered_graph, filtered_emb


def build_full_organ_masks_from_labelmap(full_mask: np.ndarray) -> dict[str, np.ndarray]:
    # For offline testing we derive organ priors from the label map because it is
    # already available for metric computation. In real deployment/inference, these
    # priors must come from an external organ segmentation model instead of GT labels.
    return {
        "kidney": _make_binary(full_mask, ORGAN_IDS["kidney"]),
        "kidney_right": _make_binary(full_mask, ORGAN_IDS["kidney_right"]),
        "kidney_left": _make_binary(full_mask, ORGAN_IDS["kidney_left"]),
        "pancreas": _make_binary(full_mask, ORGAN_IDS["pancreas"]),
        "liver": _make_binary(full_mask, ORGAN_IDS["liver"]),
    }


def graph_has_lesion_nodes(graph: Dict[str, Any]) -> bool:
    return any(str(node.get("type")) == "lesion" for node in graph.get("nodes", []))


def build_inference_batch(
    ct_crop: np.ndarray,
    organ_masks_crop: Dict[str, np.ndarray],
    graph: Dict[str, Any],
    text_emb: np.ndarray,
    spacing: Iterable[float],
) -> Dict[str, Any]:
    patch_shape = tuple(int(v) for v in ct_crop.shape)
    zero_mask = torch.zeros((1, 1, *patch_shape), dtype=torch.float32)
    lesion_masks = {
        "kidney": zero_mask.clone(),
        "kidney_right": zero_mask.clone(),
        "kidney_left": zero_mask.clone(),
        "pancreas": zero_mask.clone(),
        "liver": zero_mask.clone(),
    }
    organ_masks = {
        key: torch.from_numpy(mask[None, None, ...].astype(np.float32))
        for key, mask in organ_masks_crop.items()
    }
    return {
        "ct": torch.from_numpy(ct_crop[None, None, ...].astype(np.float32)),
        "mask_all": torch.zeros((1, 1, *patch_shape), dtype=torch.int64),
        "organ_masks": organ_masks,
        "lesion_masks": lesion_masks,
        "graph": graph,
        "text_emb": torch.from_numpy(text_emb.astype(np.float32, copy=False)),
        "has_mask": torch.zeros((1, 1), dtype=torch.float32),
        "spacing": torch.tensor(tuple(float(v) for v in spacing), dtype=torch.float32).view(1, 3),
        "skip_sample": False,
    }


def reconstruct_full_mask(pred_patch: np.ndarray, crop_meta: Dict[str, Any]) -> np.ndarray:
    original_shape = tuple(int(v) for v in crop_meta["original_shape"])
    full_start = tuple(int(v) for v in crop_meta["full_start"])
    full_end = tuple(int(v) for v in crop_meta["full_end"])
    patch_start = tuple(int(v) for v in crop_meta["patch_start"])
    patch_end = tuple(int(v) for v in crop_meta["patch_end"])

    out = np.zeros(original_shape, dtype=np.uint8)
    out[
        full_start[0]:full_end[0],
        full_start[1]:full_end[1],
        full_start[2]:full_end[2],
    ] = (
        pred_patch[
            patch_start[0]:patch_end[0],
            patch_start[1]:patch_end[1],
            patch_start[2]:patch_end[2],
        ] > 0.5
    ).astype(np.uint8)
    return out


def save_mask_nifti(mask: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ref_img.header.copy()
    header.set_data_dtype(np.uint8)
    img = nib.Nifti1Image(mask.astype(np.uint8), affine=ref_img.affine, header=header)
    nib.save(img, str(out_path))


def make_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv) if args.out_csv else save_dir / "test_metrics_per_case.csv"
    out_summary = Path(args.out_summary) if args.out_summary else save_dir / "test_metrics_summary.csv"
    pred_root = None
    if args.save_pred_masks:
        pred_root = Path(args.pred_root) if args.pred_root else save_dir / "predictions"
        pred_root.mkdir(parents=True, exist_ok=True)
    return out_csv, out_summary, pred_root


def summarize_rows(rows: list[dict[str, Any]], out_summary: Path) -> None:
    metrics_list = ["dice", "hd95", "assd", "ccls", "lesion_recall"]
    cohort_specs = (
        ("all", lambda row: True),
        ("gt_positive_only", lambda row: int(row["gt_nonzero"]) > 0),
        ("gt_negative_only", lambda row: int(row["gt_nonzero"]) == 0),
    )
    agg: dict[str, dict[str, dict[str, list[float]]]] = {}
    for row in rows:
        cls = row["class"]
        cls_entry = agg.setdefault(
            cls,
            {
                cohort_name: {metric: [] for metric in metrics_list}
                for cohort_name, _ in cohort_specs
            },
        )
        for cohort_name, predicate in cohort_specs:
            if not predicate(row):
                continue
            for metric_name in metrics_list:
                cls_entry[cohort_name][metric_name].append(
                    sanitize_metric(metric_name, row[metric_name])
                )

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with out_summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "cohort", "metric", "count", "mean", "std", "mean\u00b1std"])
        for cls in CLASS_NAMES:
            cls_metrics = agg.get(cls, {})
            for cohort_name, _ in cohort_specs:
                cohort_metrics = cls_metrics.get(cohort_name, {})
                for metric_name in metrics_list:
                    arr = np.asarray(cohort_metrics.get(metric_name, []), dtype=float)
                    if arr.size == 0:
                        writer.writerow([cls, cohort_name, metric_name, 0, "nan", "nan", "nan\u00b1nan"])
                        continue
                    mean = float(np.mean(arr))
                    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
                    writer.writerow(
                        [
                            cls,
                            cohort_name,
                            metric_name,
                            int(arr.size),
                            f"{mean:.6f}",
                            f"{std:.6f}",
                            f"{mean:.6f}\u00b1{std:.6f}",
                        ]
                    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", type=str, default="data/test.csv")
    ap.add_argument("--image_root", type=str, default="data/altas/Image_only")
    ap.add_argument("--mask_root", type=str, default="data/altas/combined_labels")
    ap.add_argument("--graph_root", type=str, default="data/gleve_offline/graphs")
    ap.add_argument("--emb_root", type=str, default="data/gleve_offline/embeddings")
    ap.add_argument("--exclude_ids", type=str, default="data/gleve_offline/failed_ids.txt")
    ap.add_argument("--load_ckpt", type=str, default="ckpts_default/best_ckpt.pt")
    ap.add_argument("--save_dir", type=str, default="ckpts_default")
    ap.add_argument("--out_csv", type=str, default="")
    ap.add_argument("--out_summary", type=str, default="")
    ap.add_argument("--save_pred_masks", action="store_true")
    ap.add_argument("--pred_root", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    ap.add_argument("--base_chan", type=int, default=64)
    ap.add_argument("--feat_dim", type=int, default=128)
    ap.add_argument("--ver_hidden_dim", type=int, default=256)
    ap.add_argument("--ocre_hidden_dim", type=int, default=128)
    ap.add_argument("--query_bank_size", type=int, default=16)
    ap.add_argument("--topk_candidates", type=int, default=4)
    ap.add_argument("--oc_depth", type=int, default=8)
    ap.add_argument("--oc_min_size", type=int, default=6)
    ap.add_argument("--candidate_warmup_epochs", type=int, default=10)
    ap.add_argument("--refine_ramp_epochs", type=int, default=5)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--disable_persistent_workers", action="store_true")
    ap.add_argument("--sample_timeout_sec", type=float, default=0.0)
    ap.add_argument("--lesion_crop_prob", type=float, default=1.0)
    ap.add_argument("--amp", type=str, default="bf16", choices=["off", "fp16", "bf16"])
    ap.add_argument("--infer_epoch", type=int, default=0, help="0 means auto-select from ckpt/config.")
    ap.add_argument("--min_pred_voxels", type=int, default=60)
    ap.add_argument("--ccls_connectivity", type=int, default=26, choices=[6, 18, 26])
    ap.add_argument("--dice_match_threshold", type=float, default=0.01)
    ap.add_argument("--d0_mm", type=float, default=20.0)
    ap.add_argument("--max_lesions_per_case", type=int, default=15)
    ap.add_argument("--crop_overlap_ratio", type=float, default=0.5)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    set_global_seed(args.seed)
    out_csv, out_summary, pred_root = make_output_paths(args)

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    case_ids = load_test_case_ids(args.test_csv, args.exclude_ids or "")
    patch_size = tuple(int(v) for v in args.patch_size)

    model = GLeVETrainModel(
        d_text=768,
        M=args.query_bank_size,
        topK=args.topk_candidates,
        num_classes=1,
        base_chan=args.base_chan,
        feat_dim=args.feat_dim,
        ver_hidden_dim=args.ver_hidden_dim,
        ocre_hidden_dim=args.ocre_hidden_dim,
        oc_depth=args.oc_depth,
        oc_min_size=args.oc_min_size,
        candidate_warmup_epochs=args.candidate_warmup_epochs,
        refine_ramp_epochs=args.refine_ramp_epochs,
        use_checkpoint=False,
    ).to(device)

    ckpt = torch.load(args.load_ckpt, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    load_report = load_model_state_flexible(model, state_dict)
    model.eval()

    ckpt_epoch = int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0
    auto_epoch = max(ckpt_epoch, int(args.candidate_warmup_epochs) + int(args.refine_ramp_epochs))
    infer_epoch = int(args.infer_epoch) if int(args.infer_epoch) > 0 else auto_epoch

    if device.type == "cuda" and args.amp != "off":
        amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
        amp_ctx = lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        amp_ctx = nullcontext

    print(
        f"[info] loaded checkpoint: {args.load_ckpt}\n"
        f"[info] model params loaded: {load_report['loaded']}/{load_report['total']}, "
        f"shape_skipped={len(load_report['skipped_shape'])}, "
        f"name_skipped={len(load_report['skipped_missing'])}, "
        f"still_missing={len(load_report['missing_after'])}\n"
        f"[info] infer_epoch={infer_epoch} (ckpt_epoch={ckpt_epoch})\n"
        f"[info] testing {len(case_ids)} cases"
    )

    rows: list[dict[str, Any]] = []
    fieldnames = [
        "id",
        "class",
        "dice",
        "hd95",
        "assd",
        "ccls",
        "lesion_recall",
        "pred_nonzero",
        "gt_nonzero",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    skipped_invalid = 0
    invalid_windows = 0
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            for case_id in tqdm(case_ids, desc="Test", dynamic_ncols=True):
                ct_path = Path(args.image_root) / case_id / "ct.nii.gz"
                mask_path = Path(args.mask_root) / case_id / "combined_labels.nii.gz"
                graph_path = Path(args.graph_root) / f"{case_id}.graph.json"
                emb_path = Path(args.emb_root) / f"{case_id}.text_emb.npz"

                try:
                    ct_full, spacing = load_nii_with_spacing(str(ct_path))
                    ct_full = hu_window_and_normalize(ct_full)
                    full_mask = np.asarray(load_nii(str(mask_path), dtype=None), dtype=np.int16)
                    ref_img = nib.load(str(mask_path))

                    graph = _load_graph_json(str(graph_path))
                    text_emb = _load_text_embedding_cached(str(emb_path))
                    if text_emb.shape[0] != len(graph["nodes"]):
                        raise ValueError(
                            f"Embedding rows ({text_emb.shape[0]}) do not match graph nodes ({len(graph['nodes'])}) "
                            f"for case {case_id}."
                        )
                    graph, text_emb = _limit_graph_lesions(graph, text_emb, args.max_lesions_per_case)
                except Exception as exc:
                    print(f"[warn] failed to load case {case_id}: {exc}")
                    skipped_invalid += 1
                    continue

                full_organ_masks = build_full_organ_masks_from_labelmap(full_mask)
                pred_full_by_class = {
                    class_name: np.zeros(full_mask.shape, dtype=np.uint8)
                    for class_name in CLASS_NAMES
                }

                for support_key, target_class, allowed_supports in SUPPORT_INFERENCE_SPECS:
                    support_mask_full = full_organ_masks[support_key]
                    if not support_mask_full.any():
                        continue

                    graph_support, text_emb_support = filter_graph_by_supports(graph, text_emb, allowed_supports)
                    if not graph_has_lesion_nodes(graph_support):
                        continue

                    crop_starts = generate_crop_starts_covering_mask(
                        support_mask_full,
                        patch_size=patch_size,
                        overlap_ratio=args.crop_overlap_ratio,
                    )
                    for crop_start in crop_starts:
                        ct_crop, crop_meta = crop_or_pad_from_start_with_meta(ct_full, crop_start, patch_size)
                        organ_masks_crop = {
                            key: crop_or_pad_from_start_with_meta(mask_one, crop_start, patch_size)[0]
                            for key, mask_one in full_organ_masks.items()
                        }
                        batch = build_inference_batch(
                            ct_crop=ct_crop,
                            organ_masks_crop=organ_masks_crop,
                            graph=graph_support,
                            text_emb=text_emb_support,
                            spacing=spacing,
                        )

                        batch["ct"] = batch["ct"].to(device, non_blocking=True)
                        batch["mask_all"] = batch["mask_all"].to(device, non_blocking=True)
                        batch["has_mask"] = batch["has_mask"].to(device, non_blocking=True)
                        batch["spacing"] = batch["spacing"].to(device, non_blocking=True)
                        batch["text_emb"] = batch["text_emb"].to(device, non_blocking=True)
                        batch["organ_masks"] = {
                            key: value.to(device, non_blocking=True) for key, value in batch["organ_masks"].items()
                        }
                        batch["lesion_masks"] = {
                            key: value.to(device, non_blocking=True) for key, value in batch["lesion_masks"].items()
                        }

                        with amp_ctx():
                            out = model(batch, epoch=infer_epoch, return_vis=True)
                        if not out.get("valid", True):
                            invalid_windows += 1
                            continue

                        pred_patch_masks = out["vis_pred_type_masks"].detach().cpu().numpy().astype(np.uint8)
                        pred_full = reconstruct_full_mask(pred_patch_masks[CLASS_TO_INDEX[target_class]], crop_meta)
                        pred_full_by_class[target_class] = np.maximum(pred_full_by_class[target_class], pred_full)

                if pred_root is not None:
                    for class_name in CLASS_NAMES:
                        save_mask_nifti(
                            pred_full_by_class[class_name],
                            ref_img,
                            pred_root / case_id / "predictions" / f"{class_name}.nii.gz",
                        )

                for class_name in CLASS_NAMES:
                    gt_bin = np.isin(full_mask, LESION_IDS[class_name])
                    pred_bin = pred_full_by_class[class_name].astype(bool)

                    pred_sum = int(np.count_nonzero(pred_bin))
                    gt_sum = int(np.count_nonzero(gt_bin))
                    dice = sanitize_metric("dice", dice_mask(pred_bin, gt_bin))
                    hd95, assd = surface_metrics_binary(pred_bin, gt_bin, spacing)
                    ccls, lesion_recall = compute_ccls(
                        gt_bin=gt_bin,
                        pred_bin=pred_bin,
                        spacing=spacing,
                        connectivity=args.ccls_connectivity,
                        min_pred_voxels=args.min_pred_voxels,
                        dice_match_threshold=args.dice_match_threshold,
                        d0_mm=args.d0_mm,
                    )
                    hd95 = sanitize_metric("hd95", hd95)
                    assd = sanitize_metric("assd", assd)
                    ccls = sanitize_metric("ccls", ccls)
                    lesion_recall = sanitize_metric("lesion_recall", lesion_recall)

                    row = {
                        "id": case_id,
                        "class": class_name,
                        "dice": f"{dice:.6f}",
                        "hd95": f"{hd95:.6f}",
                        "assd": f"{assd:.6f}",
                        "ccls": f"{ccls:.6f}",
                        "lesion_recall": f"{lesion_recall:.6f}",
                        "pred_nonzero": pred_sum,
                        "gt_nonzero": gt_sum,
                    }
                    writer.writerow(row)
                    rows.append(row)
                f.flush()

    summarize_rows(rows, out_summary)
    print(f"[info] skipped_invalid_cases={skipped_invalid}")
    print(f"[info] invalid_windows={invalid_windows}")
    print(f"[info] wrote {out_csv}")
    print(f"[info] wrote {out_summary}")
    if pred_root is not None:
        print(f"[info] wrote prediction masks under {pred_root}")


if __name__ == "__main__":
    main()
