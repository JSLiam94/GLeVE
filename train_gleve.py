# train_gleve.py
from __future__ import annotations
import json
import os
import random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import argparse
import glob
from contextlib import nullcontext
from typing import Dict, Any
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset_abdomenatlas import AbdomenAtlasGLeVEDataset, estimate_physical_stats
from gleve_model import GLeVETrainModel, load_model_state_flexible
from utils_nii import collate_fn


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


def make_worker_init_fn(base_seed: int):
    def _seed_worker(worker_id: int) -> None:
        worker_seed = (int(base_seed) + int(worker_id)) % (2 ** 32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _seed_worker




def _binary_dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    inter = float(np.logical_and(pred, gt).sum())
    denom = float(pred.sum() + gt.sum())
    return (2.0 * inter + eps) / (denom + eps)


def _surface_metrics_binary(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: np.ndarray,
) -> tuple[float, float]:
    try:
        from scipy import ndimage
    except ImportError as e:
        raise ImportError("Validation metrics HD95/ASSD require scipy. Install scipy in the training environment.") from e

    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0, 0.0
    if not pred.any() or not gt.any():
        return float("nan"), float("nan")

    footprint = ndimage.generate_binary_structure(rank=3, connectivity=1)
    pred_surface = np.logical_xor(pred, ndimage.binary_erosion(pred, structure=footprint, border_value=0))
    gt_surface = np.logical_xor(gt, ndimage.binary_erosion(gt, structure=footprint, border_value=0))
    if not pred_surface.any() or not gt_surface.any():
        return float("nan"), float("nan")

    dt_to_gt = ndimage.distance_transform_edt(~gt_surface, sampling=spacing)
    dt_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    dist_pred_to_gt = dt_to_gt[pred_surface]
    dist_gt_to_pred = dt_to_pred[gt_surface]
    hd95 = max(float(np.percentile(dist_pred_to_gt, 95.0)), float(np.percentile(dist_gt_to_pred, 95.0)))
    assd = 0.5 * (float(dist_pred_to_gt.mean()) + float(dist_gt_to_pred.mean()))
    return float(hd95), float(assd)


def evaluate_validation(
    model_for_eval: torch.nn.Module,
    val_dl: DataLoader,
    device: torch.device,
    epoch: int,
    save_dir: str,
) -> Dict[str, float]:
    was_training = model_for_eval.training
    model_for_eval.eval()

    dice_values: list[float] = []
    hd95_values: list[float] = []
    assd_values: list[float] = []
    skipped_invalid = 0

    with torch.no_grad():
        for batch in tqdm(val_dl, desc=f"Val {epoch:03d}", leave=False, dynamic_ncols=True):
            if batch.get("skip_sample", False):
                skipped_invalid += 1
                continue
            out = model_for_eval(batch, epoch=epoch, return_vis=True)
            if not out.get("valid", True):
                skipped_invalid += 1
                continue

            pred_union = out["vis_pred_union"].numpy().astype(np.uint8)
            gt_union = out["vis_gt_union"].numpy().astype(np.uint8)
            spacing = batch["spacing"][0].cpu().numpy().astype(np.float64)

            dice_values.append(_binary_dice(pred_union, gt_union))
            hd95, assd = _surface_metrics_binary(pred_union, gt_union, spacing)
            hd95_values.append(hd95)
            assd_values.append(assd)

    if was_training:
        model_for_eval.train()

    metrics = {
        "epoch": int(epoch),
        "num_cases": int(len(dice_values)),
        "skipped_invalid": int(skipped_invalid),
        "dice": float(np.mean(dice_values)) if dice_values else float("nan"),
        "hd95": float(np.nanmean(hd95_values)) if hd95_values and not np.all(np.isnan(hd95_values)) else float("nan"),
        "assd": float(np.nanmean(assd_values)) if assd_values and not np.all(np.isnan(assd_values)) else float("nan"),
        "hd95_valid_cases": int(np.isfinite(np.asarray(hd95_values, dtype=np.float64)).sum()) if hd95_values else 0,
        "assd_valid_cases": int(np.isfinite(np.asarray(assd_values, dtype=np.float64)).sum()) if assd_values else 0,
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "val_metrics.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
    return metrics


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed():
    if not is_distributed():
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_mean(value: float, device: torch.device, world_size: int) -> float:
    if world_size == 1:
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / world_size


def sync_skip_step(local_skip: bool, device: torch.device, world_size: int) -> bool:
    if world_size == 1:
        return local_skip
    t = torch.tensor([1 if local_skip else 0], dtype=torch.int64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


def _is_cuda_oom_error(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _gather_objects(local_obj: Any, world_size: int) -> list[Any]:
    if world_size == 1:
        return [local_obj]
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_obj)
    return gathered


def _append_oom_records(save_dir: str, records: list[Dict[str, Any]]) -> None:
    if not records:
        return
    os.makedirs(save_dir, exist_ok=True)
    jsonl_path = os.path.join(save_dir, "oom_samples.jsonl")
    txt_path = os.path.join(save_dir, "oom_case_ids.txt")

    with open(jsonl_path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    existing_case_ids: set[str] = set()
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            existing_case_ids = {line.strip() for line in f if line.strip()}

    new_case_ids = []
    for record in records:
        case_id = str(record.get("case_id", "")).strip()
        if not case_id or case_id in existing_case_ids:
            continue
        existing_case_ids.add(case_id)
        new_case_ids.append(case_id)

    if new_case_ids:
        with open(txt_path, "a", encoding="utf-8") as f:
            for case_id in new_case_ids:
                f.write(case_id + "\n")


def _all_reduce_grads(model: torch.nn.Module, world_size: int) -> None:
    if world_size == 1:
        return
    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def _flush_optimizer_step(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    grad_accum_steps: int,
    backward_in_window: int,
    world_size: int,
    grads_are_synced: bool,
) -> None:
    if backward_in_window <= 0:
        return
    if world_size > 1 and not grads_are_synced:
        _all_reduce_grads(model, world_size)
    if backward_in_window != grad_accum_steps:
        scale = grad_accum_steps / float(backward_in_window)
        for param in model.parameters():
            if param.grad is not None:
                param.grad.mul_(scale)
    if scaler.is_enabled():
        scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
    if scaler.is_enabled():
        scaler.step(opt)
        scaler.update()
    else:
        opt.step()
    opt.zero_grad(set_to_none=True)


def find_latest_ckpt(save_dir: str) -> str | None:
    pattern = os.path.join(save_dir, "gleve_ep*.pt")
    ckpt_paths = glob.glob(pattern)
    if not ckpt_paths:
        return None
    return max(ckpt_paths, key=os.path.getmtime)


def _load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}")
    return data


def _save_json_file(path: str, data: Dict[str, Any]) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def _octree_param_count(model: torch.nn.Module) -> int:
    return sum(
        int(param.numel())
        for name, param in model.named_parameters()
        if name.startswith("ocre.") and param.requires_grad
    )


def _save_rgb_image(path: str, rgb: np.ndarray) -> None:
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
        return
    except ImportError:
        if path.lower().endswith(".png"):
            path = os.path.splitext(path)[0] + ".ppm"
        h, w, _ = rgb.shape
        with open(path, "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
            f.write(rgb.tobytes())


_FONT_5X7 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
}


def _draw_text(rgb: np.ndarray, text: str, top: int, left: int, color: np.ndarray, scale: int = 2) -> None:
    text = text.upper()
    h, w, _ = rgb.shape
    x = left
    for ch in text:
        glyph = _FONT_5X7.get(ch, _FONT_5X7[" "])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit != "1":
                    continue
                y0 = top + gy * scale
                y1 = min(y0 + scale, h)
                x0 = x + gx * scale
                x1 = min(x0 + scale, w)
                if y0 < h and x0 < w:
                    rgb[y0:y1, x0:x1] = color
        x += (5 + 1) * scale


def _add_column_titles(canvas: np.ndarray, title_specs: list[tuple[str, int, int]]) -> np.ndarray:
    title_h = 24
    out = np.full((canvas.shape[0] + title_h, canvas.shape[1], 3), 255, dtype=np.uint8)
    out[title_h:] = canvas
    for title, x, w in title_specs:
        text_w = len(title) * 12 - 2
        text_x = x + max((w - text_w) // 2, 2)
        _draw_text(out, title, top=5, left=text_x, color=np.array([16, 16, 16], dtype=np.uint8), scale=2)
    return out


def _add_row_label(row: np.ndarray, label: str, label_w: int = 70) -> np.ndarray:
    out = np.full((row.shape[0], row.shape[1] + label_w, 3), 255, dtype=np.uint8)
    out[:, label_w:] = row
    text_w = len(label) * 12 - 2
    text_x = max((label_w - text_w) // 2, 2)
    text_y = max((row.shape[0] - 14) // 2, 2)
    _draw_text(out, label, top=text_y, left=text_x, color=np.array([16, 16, 16], dtype=np.uint8), scale=2)
    return out


def save_lesion_vis(
    vis_dir: str,
    case_id: str,
    epoch: int,
    step: int,
    ct: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    gt_union: torch.Tensor,
    chosen_union: torch.Tensor,
    cond0_union: torch.Tensor,
    pred_union: torch.Tensor,
    metrics: Dict[str, float] | None = None,
) -> None:
    colors = np.array(
        [
            [255, 64, 64],   # kidney lesion
            [64, 220, 64],   # pancreas lesion
            [64, 128, 255],  # liver lesion
        ],
        dtype=np.uint8,
    )

    def project_rgb(type_masks: torch.Tensor) -> np.ndarray:
        masks_2d = (type_masks > 0.5).any(dim=-1).cpu().numpy().astype(np.uint8)  # (3,H,W)
        h, w = masks_2d.shape[1:]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(colors):
            rgb[masks_2d[idx] > 0] = color
        return rgb

    gt_rgb = project_rgb(gt_masks)
    pred_rgb = project_rgb(pred_masks)
    gap = np.full((gt_rgb.shape[0], 12, 3), 255, dtype=np.uint8)
    proj_canvas = np.concatenate([gt_rgb, gap, pred_rgb], axis=1)
    proj_canvas = _add_column_titles(
        proj_canvas,
        [
            ("GT TYPES", 0, gt_rgb.shape[1]),
            ("PRED TYPES", gt_rgb.shape[1] + 12, pred_rgb.shape[1]),
        ],
    )
    os.makedirs(vis_dir, exist_ok=True)
    base_name = f"ep{epoch:03d}_step{step:06d}_{case_id}"
    _save_rgb_image(os.path.join(vis_dir, f"{base_name}_types.png"), proj_canvas)

    ct_np = np.clip(ct.cpu().numpy().astype(np.float32), 0.0, 1.0)
    gt_union_np = (gt_union > 0.5).cpu().numpy().astype(np.uint8)
    chosen_union_np = (chosen_union > 0.5).cpu().numpy().astype(np.uint8)
    cond0_union_np = (cond0_union > 0.5).cpu().numpy().astype(np.uint8)
    pred_union_np = (pred_union > 0.5).cpu().numpy().astype(np.uint8)

    def pick_center() -> tuple[int, int, int]:
        for mask_np in [gt_union_np, pred_union_np, cond0_union_np, chosen_union_np]:
            coords = np.argwhere(mask_np > 0)
            if coords.size == 0:
                continue
            lo = coords.min(axis=0)
            hi = coords.max(axis=0)
            center = ((lo + hi) // 2).astype(np.int64)
            return int(center[0]), int(center[1]), int(center[2])
        h, w, d = ct_np.shape
        return h // 2, w // 2, d // 2

    def slice_plane(vol: np.ndarray, plane: str, idx: int) -> np.ndarray:
        if plane == "axial":
            sl = vol[:, :, idx]
        elif plane == "coronal":
            sl = vol[:, idx, :]
        else:
            sl = vol[idx, :, :]
        return np.ascontiguousarray(np.rot90(sl))

    def ct_to_rgb(sl: np.ndarray) -> np.ndarray:
        gray = np.clip(sl * 255.0, 0.0, 255.0).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

    def overlay(ct_sl: np.ndarray, gt_sl: np.ndarray, cur_sl: np.ndarray) -> np.ndarray:
        rgb = ct_to_rgb(ct_sl).astype(np.float32)
        gt_idx = gt_sl > 0
        cur_idx = cur_sl > 0
        both_idx = gt_idx & cur_idx
        rgb[gt_idx] = 0.55 * rgb[gt_idx] + 0.45 * np.array([64, 220, 64], dtype=np.float32)
        rgb[cur_idx] = 0.55 * rgb[cur_idx] + 0.45 * np.array([255, 64, 64], dtype=np.float32)
        rgb[both_idx] = 0.35 * rgb[both_idx] + 0.65 * np.array([255, 220, 64], dtype=np.float32)
        return np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    def pad_width(rgb: np.ndarray, target_w: int) -> np.ndarray:
        if rgb.shape[1] >= target_w:
            return rgb
        pad_left = (target_w - rgb.shape[1]) // 2
        pad_right = target_w - rgb.shape[1] - pad_left
        return np.pad(rgb, ((0, 0), (pad_left, pad_right), (0, 0)), constant_values=255)

    ch, cw, cd = pick_center()
    slice_specs = [("axial", cd), ("coronal", cw), ("sagittal", ch)]
    row_gap = 10
    col_gap = 8
    row_gap_canvas = np.full((row_gap, 1, 3), 255, dtype=np.uint8)
    rows = []
    for plane, idx in slice_specs:
        ct_sl = slice_plane(ct_np, plane, idx)
        gt_sl = slice_plane(gt_union_np, plane, idx)
        chosen_sl = slice_plane(chosen_union_np, plane, idx)
        cond0_sl = slice_plane(cond0_union_np, plane, idx)
        pred_sl = slice_plane(pred_union_np, plane, idx)

        panels = [
            overlay(ct_sl, gt_sl, np.zeros_like(gt_sl, dtype=np.uint8)),
            overlay(ct_sl, gt_sl, chosen_sl),
            overlay(ct_sl, gt_sl, cond0_sl),
            overlay(ct_sl, gt_sl, pred_sl),
        ]
        gap_rgb = np.full((panels[0].shape[0], col_gap, 3), 255, dtype=np.uint8)
        row = panels[0]
        for panel in panels[1:]:
            row = np.concatenate([row, gap_rgb, panel], axis=1)
        rows.append(_add_row_label(row, plane))

    max_w = max(row.shape[1] for row in rows)
    rows = [pad_width(row, max_w) for row in rows]
    overlay_canvas = rows[0]
    for row in rows[1:]:
        overlay_canvas = np.concatenate(
            [overlay_canvas, np.tile(row_gap_canvas, (1, max_w, 1)), row],
            axis=0,
        )
    label_w = 70
    panel_w = (rows[0].shape[1] - label_w - 3 * col_gap) // 4
    overlay_canvas = _add_column_titles(
        overlay_canvas,
        [
            ("GT", label_w, panel_w),
            ("CHOSEN", label_w + panel_w + col_gap, panel_w),
            ("COND0", label_w + 2 * (panel_w + col_gap), panel_w),
            ("PRED", label_w + 3 * (panel_w + col_gap), panel_w),
        ],
    )
    _save_rgb_image(os.path.join(vis_dir, f"{base_name}_overlay.png"), overlay_canvas)

    if metrics is not None:
        with open(os.path.join(vis_dir, f"{base_name}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="/data/IID_train_has_lesion.csv")
    ap.add_argument("--val_csv", type=str, default="data/val.csv", help="Optional validation csv. When set, run validation every --eval_every epochs.")
    ap.add_argument("--image_root", type=str, default="data/altas/Image_only")
    ap.add_argument("--mask_root", type=str, default="data/altas/combined_labels")
    ap.add_argument("--graph_root", type=str, default="data/gleve_offline/graphs")
    ap.add_argument("--emb_root", type=str, default="data/gleve_offline/embeddings")
    ap.add_argument("--exclude_ids", type=str, default="data/gleve_offline/failed_ids.txt", help="Path to failed_ids.txt to exclude from training")
    ap.add_argument("--mask_ratio", type=float, default=1.0)
    ap.add_argument("--lesion_crop_prob", type=float, default=1.0, help="Probability of sampling lesion-centered crops on lesion-positive cases.")
    ap.add_argument("--supervised_gt_crop_epochs", type=int, default=50, help="For supervised cases only, use GT-near crops during the first N epochs; later switch to inference-like organ crops.")
    ap.add_argument("--max_lesions_per_case", type=int, default=15, help="Keep at most this many lesion nodes per cropped case, sorted by reported volume_cc. Set 0 to disable.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs and at the final epoch.")
    ap.add_argument("--eval_every", type=int, default=5, help="Run validation every N epochs when --val_csv is set.")
    ap.add_argument("--save_dir", type=str, default="./ckpts_default")
    ap.add_argument("--load_ckpt", type=str, default="", help="Optional checkpoint path (xxx.pt) to warm-start or resume from.")
    ap.add_argument("--auto_resume", action="store_true", help="If set and --load_ckpt is empty, resume from the latest checkpoint in --save_dir.")
    ap.add_argument("--resume_training_state", action="store_true", help="Restore optimizer/scheduler/scaler too. Use only if architecture is unchanged.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Global random seed for training and validation.")
    ap.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    ap.add_argument("--base_chan", type=int, default=64)
    ap.add_argument("--feat_dim", type=int, default=128)
    ap.add_argument("--ver_hidden_dim", type=int, default=256)
    ap.add_argument("--ocre_hidden_dim", type=int, default=128)
    ap.add_argument("--query_bank_size", type=int, default=16)
    ap.add_argument("--topk_candidates", type=int, default=4)
    ap.add_argument("--oc_depth", type=int, default=8)
    ap.add_argument("--oc_min_size", type=int, default=6)
    ap.add_argument("--candidate_warmup_epochs", type=int, default=10, help="Train only candidate selection for the first N epochs, then enable segmentation.")
    ap.add_argument("--refine_ramp_epochs", type=int, default=5, help="Linearly ramp segmentation/coarse refinement losses for the first N epochs after warmup.")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--disable_persistent_workers", action="store_true")
    ap.add_argument("--sample_timeout_sec", type=float, default=80.0, help="Skip a sample if dataset loading/preprocess exceeds this many seconds. 0 disables timeout.")
    ap.add_argument("--physical_scan_cases", type=int, default=128, help="Randomly scan up to N cases to calibrate voxel-volume/HU weak supervision scales.")
    ap.add_argument("--physical_stats_json", type=str, default="", help="Optional JSON cache for physical stats. If present, load it directly; otherwise scan and save the generated stats.")
    ap.add_argument("--amp", type=str, default="bf16", choices=["off", "fp16", "bf16"])
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    ckpt_group = ap.add_mutually_exclusive_group()
    ckpt_group.add_argument(
        "--enable_activation_checkpointing",
        dest="activation_checkpointing",
        action="store_true",
        help="Enable activation checkpointing in the 3D backbone. This reduces VRAM but slows training.",
    )
    ckpt_group.add_argument(
        "--disable_activation_checkpointing",
        dest="activation_checkpointing",
        action="store_false",
        help="Compatibility flag. Activation checkpointing is disabled by default for higher throughput.",
    )
    ap.add_argument("--vis_every", type=int, default=500, help="Save 2D lesion GT/pred visualization every N steps on rank 0.")
    ap.add_argument("--use_compile", action="store_true", help="Enable torch.compile. Disabled by default because some large 3D configs can trigger Triton illegal memory access.")
    ap.set_defaults(activation_checkpointing=False)
    args = ap.parse_args()
    grad_accum_steps = max(int(args.grad_accum_steps), 1)

    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    set_global_seed(args.seed)

    if is_main:
        os.makedirs(args.save_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    physical_stats: Dict[str, Any] = {}
    physical_stats_json = args.physical_stats_json.strip()
    physical_stats_cache_path = physical_stats_json or os.path.join(args.save_dir, "physical_stats.json")
    if world_size == 1 or is_main:
        if physical_stats_json and os.path.exists(physical_stats_json):
            try:
                physical_stats = _load_json_file(physical_stats_json)
                print(f"[physical] loaded json: {physical_stats_json}")
            except Exception as e:
                print(f"[warn] failed to load physical stats json '{physical_stats_json}': {e}")
                physical_stats = {}

        if not physical_stats and args.physical_scan_cases > 0:
            try:
                physical_stats = estimate_physical_stats(
                    train_csv=args.train_csv,
                    image_root=args.image_root,
                    mask_root=args.mask_root,
                    graph_root=args.graph_root,
                    max_cases=args.physical_scan_cases,
                    seed=args.seed,
                )
                _save_json_file(physical_stats_cache_path, physical_stats)
                print(f"[physical] saved json: {physical_stats_cache_path}")
            except Exception as e:
                print(f"[warn] physical scan skipped: {e}")
                physical_stats = {}
        elif not physical_stats and physical_stats_json:
            print(f"[warn] physical stats json not found and physical_scan_cases <= 0: {physical_stats_json}")
    if world_size > 1:
        obj_list = [physical_stats]
        dist.broadcast_object_list(obj_list, src=0)
        physical_stats = dict(obj_list[0] or {})
    if is_main and physical_stats:
        print(f"[physical] {json.dumps(physical_stats, ensure_ascii=False)}")

    ds = AbdomenAtlasGLeVEDataset(
        train_csv=args.train_csv,
        image_root=args.image_root,
        mask_root=args.mask_root,
        graph_root=args.graph_root,
        emb_root=args.emb_root,
        exclude_ids_path=args.exclude_ids,
        mask_ratio=args.mask_ratio,
        seed=args.seed,
        patch_size=tuple(args.patch_size),
        lesion_crop_prob=args.lesion_crop_prob,
        sample_timeout_sec=args.sample_timeout_sec,
        max_lesions_per_case=args.max_lesions_per_case,
        supervised_gt_crop_epochs=args.supervised_gt_crop_epochs,
    )
    val_ds = None
    val_dl = None
    if is_main and args.val_csv.strip():
        val_ds = AbdomenAtlasGLeVEDataset(
            train_csv=args.val_csv,
            image_root=args.image_root,
            mask_root=args.mask_root,
            graph_root=args.graph_root,
            emb_root=args.emb_root,
            exclude_ids_path=args.exclude_ids,
            mask_ratio=1.0,
            seed=args.seed,
            patch_size=tuple(args.patch_size),
            lesion_crop_prob=1.0,
            deterministic_crop=True,
            sample_timeout_sec=args.sample_timeout_sec,
            max_lesions_per_case=args.max_lesions_per_case,
            supervised_gt_crop_epochs=0,
        )
        val_generator = torch.Generator()
        val_generator.manual_seed(int(args.seed))
        val_dl_kwargs = dict(
            dataset=val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0 and not args.disable_persistent_workers),
            worker_init_fn=make_worker_init_fn(int(args.seed)),
            generator=val_generator,
        )
        if args.num_workers > 0:
            val_dl_kwargs["prefetch_factor"] = args.prefetch_factor
        val_dl = DataLoader(**val_dl_kwargs)

    sampler = DistributedSampler(
        ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(args.seed),
    ) if world_size > 1 else None
    train_generator = torch.Generator()
    train_generator.manual_seed(int(args.seed))
    dl_kwargs = dict(
        dataset=ds,
        batch_size=1,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0 and not args.disable_persistent_workers),
        worker_init_fn=make_worker_init_fn(int(args.seed)),
        generator=train_generator,
    )
    if args.num_workers > 0:
        dl_kwargs["prefetch_factor"] = args.prefetch_factor
    dl = DataLoader(**dl_kwargs)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if world_size > 1 else args.device)
    else:
        device = torch.device("cpu")

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
        physical_stats=physical_stats,
        use_checkpoint=args.activation_checkpointing,
    ).to(device)

    if is_main:
        print(
            f"[octree] depth={args.oc_depth} min_size={args.oc_min_size} "
            f"hidden={args.ocre_hidden_dim} trainable_params={_octree_param_count(model)}"
        )

    if args.use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp == "fp16"))

    resume_ckpt = args.load_ckpt.strip() or None
    if not resume_ckpt and args.auto_resume:
        resume_ckpt = find_latest_ckpt(args.save_dir)
    start_epoch = 1
    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location=device)
        model_report = load_model_state_flexible(model, ckpt["model"])
        ckpt_model_cfg = ckpt.get("model_config") or {}
        restored_training_state = False
        if args.resume_training_state:
            try:
                if "opt" in ckpt:
                    opt.load_state_dict(ckpt["opt"])
                if "sched" in ckpt:
                    sched.load_state_dict(ckpt["sched"])
                if "scaler" in ckpt and scaler.is_enabled():
                    scaler.load_state_dict(ckpt["scaler"])
                restored_training_state = True
            except Exception as e:
                if is_main:
                    print(f"Skipped optimizer/scheduler/scaler restore due to mismatch: {e}")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if is_main:
            print(f"Loaded checkpoint: {resume_ckpt} (next epoch: {start_epoch})")
            if ckpt_model_cfg:
                tracked_keys = ("oc_depth", "oc_min_size", "ocre_hidden_dim")
                changed = []
                for key in tracked_keys:
                    old_v = ckpt_model_cfg.get(key, None)
                    new_v = getattr(args, key, None)
                    if old_v is not None and new_v is not None and old_v != new_v:
                        changed.append(f"{key}: ckpt={old_v} current={new_v}")
                if changed:
                    print("[octree] config changed since checkpoint:")
                    for item in changed:
                        print(f"  {item}")
                    print("[octree] note: oc_depth/oc_min_size change forward behavior only; they are not tensor weights, so checkpoint loading can still be fully successful.")
            print(
                f"Model params loaded: {model_report['loaded']}/{model_report['total']}, "
                f"shape_skipped={len(model_report['skipped_shape'])}, "
                f"name_skipped={len(model_report['skipped_missing'])}, "
                f"still_missing={len(model_report['missing_after'])}"
            )
            if model_report["skipped_shape"]:
                print("Shape-mismatched params:")
                for name, old_shape, new_shape in model_report["skipped_shape"][:16]:
                    print(f"  {name}: ckpt{old_shape} != model{new_shape}")
                if len(model_report["skipped_shape"]) > 16:
                    print(f"  ... and {len(model_report['skipped_shape']) - 16} more")
            if restored_training_state:
                print("Restored optimizer/scheduler/scaler state.")
                print(f"Resumed optimizer LR: {opt.param_groups[0]['lr']:.6g}")
            else:
                print("Warm-start only; optimizer/scheduler/scaler were reset.")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if device.type == "cuda" and args.amp != "off":
        amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
        amp_ctx = lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        amp_ctx = nullcontext

    model.train()
    epoch_iter = range(start_epoch, args.epochs + 1)
    epoch_bar = tqdm(epoch_iter, desc="Training", dynamic_ncols=True) if is_main else epoch_iter
    global_step = 0
    vis_dir = os.path.join(args.save_dir, "vis")
    last_val_metrics: Dict[str, float] | None = None

    for ep in epoch_bar:
        ds.set_epoch(ep)
        if val_ds is not None:
            val_ds.set_epoch(ep)
        if sampler is not None:
            sampler.set_epoch(ep)

        loss_meter = 0.0
        valid_steps = 0
        backward_in_window = 0
        skipped_empty = 0
        skipped_nonfinite = 0
        skipped_timeout = 0
        skipped_oom = 0
        opt.zero_grad(set_to_none=True)

        iter_bar = tqdm(dl, desc=f"Epoch {ep:03d}", leave=False, dynamic_ncols=True) if is_main else dl
        for it, batch in enumerate(iter_bar, 1):
            global_step += 1
            local_skip = bool(batch.get("skip_sample", False))
            skip_this_step = sync_skip_step(local_skip, device, world_size)
            if skip_this_step:
                skipped_timeout += 1
                if is_main:
                    if local_skip:
                        case_id = str(batch.get("case_id", "unknown"))
                        reason = str(batch.get("skip_reason", "timeout"))
                        iter_bar.write(f"[skip] epoch={ep:03d} step={global_step:06d} case={case_id} reason={reason}")
                    else:
                        iter_bar.write(f"[skip] epoch={ep:03d} step={global_step:06d} reason=timeout_on_other_rank")
                    iter_bar.set_postfix(
                        {
                            "lr": f"{opt.param_groups[0]['lr']:.2e}",
                            "skip_timeout": skipped_timeout,
                        },
                        refresh=False,
                    )
                continue
            need_vis = is_main and args.vis_every > 0 and (global_step % args.vis_every == 0)
            should_sync = (
                world_size == 1
                or grad_accum_steps == 1
                or ((backward_in_window + 1) % grad_accum_steps == 0)
            )
            ddp_sync_ctx = nullcontext
            if world_size > 1 and grad_accum_steps > 1 and hasattr(model, "no_sync") and not should_sync:
                ddp_sync_ctx = model.no_sync
            with ddp_sync_ctx():
                out = None
                local_oom = False
                try:
                    with amp_ctx():
                        out = model(batch, epoch=ep, return_vis=need_vis)
                except RuntimeError as e:
                    if not _is_cuda_oom_error(e):
                        raise
                    local_oom = True
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                skip_oom_step = sync_skip_step(local_oom, device, world_size)
                if skip_oom_step:
                    skipped_oom += 1
                    out = None
                    local_oom_record = None
                    if local_oom:
                        local_oom_record = {
                            "epoch": int(ep),
                            "step": int(global_step),
                            "rank": int(rank),
                            "case_id": str(batch.get("case_id", "unknown")),
                            "reason": "cuda_oom",
                        }
                    oom_records = [
                        record
                        for record in _gather_objects(local_oom_record, world_size)
                        if isinstance(record, dict) and record.get("case_id")
                    ]
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if is_main and oom_records:
                        _append_oom_records(args.save_dir, oom_records)
                    if is_main:
                        case_ids = ",".join(str(record["case_id"]) for record in oom_records) if oom_records else "unknown"
                        reason = "cuda_oom" if local_oom else "cuda_oom_on_other_rank"
                        iter_bar.write(f"[skip] epoch={ep:03d} step={global_step:06d} cases={case_ids} reason={reason}")
                        iter_bar.set_postfix(
                            {
                                "lr": f"{opt.param_groups[0]['lr']:.2e}",
                                "skip_oom": skipped_oom,
                            },
                            refresh=False,
                        )
                    continue

                if not out.get("valid", True):
                    skipped_empty += 1
                    if is_main:
                        iter_bar.set_postfix(refresh=False)
                    continue

                loss = out["loss_total"]
                if not loss.requires_grad:
                    skipped_empty += 1
                    if is_main:
                        iter_bar.set_postfix(refresh=False)
                    continue

                if not torch.isfinite(loss):
                    skipped_nonfinite += 1
                    if is_main:
                        iter_bar.set_postfix(refresh=False)
                    continue

                loss_to_backward = loss / grad_accum_steps
                if scaler.is_enabled():
                    scaler.scale(loss_to_backward).backward()
                else:
                    loss_to_backward.backward()

            backward_in_window += 1
            should_step = backward_in_window % grad_accum_steps == 0
            if should_step:
                _flush_optimizer_step(
                    model=model,
                    opt=opt,
                    scaler=scaler,
                    grad_accum_steps=grad_accum_steps,
                    backward_in_window=backward_in_window,
                    world_size=world_size,
                    grads_are_synced=should_sync,
                )
                backward_in_window = 0

            loss_meter += float(loss.item())
            valid_steps += 1
            if need_vis and "vis_pred_type_masks" in out and "vis_gt_type_masks" in out:
                save_lesion_vis(
                    vis_dir=vis_dir,
                    case_id=str(batch["case_id"]),
                    epoch=ep,
                    step=global_step,
                    ct=out["vis_ct"],
                    gt_masks=out["vis_gt_type_masks"],
                    pred_masks=out["vis_pred_type_masks"],
                    gt_union=out["vis_gt_union"],
                    chosen_union=out["vis_chosen_union"],
                    cond0_union=out["vis_cond0_union"],
                    pred_union=out["vis_pred_union"],
                    metrics={
                        "epoch": int(ep),
                        "step": int(global_step),
                        "lr": float(opt.param_groups[0]["lr"]),
                        "loc_best_iou": float(out["loc_best_iou"].item()),
                        "loc_best_recall": float(out["loc_best_recall"].item()),
                        "loc_best_fbeta": float(out["loc_best_fbeta"].item()),
                        "loss_loc_rank": float(out["loss_loc_rank"].item()),
                        "loss_loc_dense": float(out["loss_loc_dense"].item()),
                        "dense_loc_iou": float(out["dense_loc_iou"].item()),
                        "chosen_iou": float(out["chosen_iou"].item()),
                        "cond0_iou": float(out["cond0_iou"].item()),
                        "pred_union_iou": float(out["pred_union_iou"].item()),
                        "refine_alpha": float(out["refine_alpha"].item()),
                        "report_term_alpha": float(out["report_term_alpha"].item()),
                        "loss_tversky": float(out["loss_tversky"].item()),
                        "loss_weak_base": float(out["loss_weak_base"].item()),
                        "loss_weak_report": float(out["loss_weak_report"].item()),
                        "loss_vol": float(out["loss_vol"].item()),
                        "loss_vol_under": float(out["loss_vol_under"].item()),
                        "loss_hu": float(out["loss_hu"].item()),
                        "lesion_match_iou": float(out["lesion_match_iou"].item()),
                        "lesion_precision": float(out["lesion_precision"].item()),
                        "lesion_recall": float(out["lesion_recall"].item()),
                        "matched_lesions": float(out["matched_lesions"].item()),
                        "unmatched_pred_lesions": float(out["unmatched_pred_lesions"].item()),
                        "unmatched_gt_lesions": float(out["unmatched_gt_lesions"].item()),
                        "chosen_voxels": float(out["chosen_voxels"].item()),
                        "cond0_voxels": float(out["cond0_voxels"].item()),
                        "pred_voxels": float(out["pred_voxels"].item()),
                        "gt_voxels": float(out["gt_voxels"].item()),
                    },
                )
            avg_loss = loss_meter / max(valid_steps, 1)
            if is_main:
                postfix = {
                    "lr": f"{opt.param_groups[0]['lr']:.2e}",
                    "loss": f"{avg_loss:.4f}",
                    "sup": int(float(batch["has_mask"].item())),
                    "lesions": int(out.get("num_lesions", 0)),
                    "gt_vox": int(float(out["gt_voxels"].item())),
                }
                if float(out["candidate_only"].item()) > 0.5:
                    postfix.update(
                        loc_rec=f"{float(out['loc_best_recall'].item()):.4f}",
                        loc_f2=f"{float(out['loc_best_fbeta'].item()):.4f}",
                        dense_iou=f"{float(out['dense_loc_iou'].item()):.4f}",
                        chosen_iou=f"{float(out['chosen_iou'].item()):.4f}",
                        chosen_vox=int(float(out["chosen_voxels"].item())),
                    )
                else:
                    postfix.update(
                        a=f"{float(out['refine_alpha'].item()):.2f}",
                        wa=f"{float(out['report_term_alpha'].item()):.2f}",
                        chosen_iou=f"{float(out['chosen_iou'].item()):.4f}",
                        cond0_iou=f"{float(out['cond0_iou'].item()):.4f}",
                        cond0_vox=int(float(out["cond0_voxels"].item())),
                        pred_vox=int(float(out["pred_voxels"].item())),
                        pred_iou=f"{float(out['pred_union_iou'].item()):.4f}",
                        match_iou=f"{float(out['lesion_match_iou'].item()):.4f}",
                        det_r=f"{float(out['lesion_recall'].item()):.4f}",
                        lvol=f"{float(out['loss_vol'].item()):.3f}",
                        lhu=f"{float(out['loss_hu'].item()):.3f}",
                    )
                iter_bar.set_postfix(postfix, refresh=False)

        if backward_in_window > 0:
            _flush_optimizer_step(
                model=model,
                opt=opt,
                scaler=scaler,
                grad_accum_steps=grad_accum_steps,
                backward_in_window=backward_in_window,
                world_size=world_size,
                grads_are_synced=False,
            )
            backward_in_window = 0

        sched.step()

        epoch_loss = reduce_mean(loss_meter / max(valid_steps, 1), device, world_size)
        epoch_valid_steps = reduce_mean(float(valid_steps), device, world_size)
        epoch_skipped_empty = reduce_mean(float(skipped_empty), device, world_size)
        epoch_skipped_nonfinite = reduce_mean(float(skipped_nonfinite), device, world_size)
        epoch_skipped_timeout = reduce_mean(float(skipped_timeout), device, world_size)
        epoch_skipped_oom = reduce_mean(float(skipped_oom), device, world_size)

        should_eval = bool(val_dl is not None and (ep % max(args.eval_every, 1) == 0 or ep == args.epochs))
        if world_size > 1:
            dist.barrier()
        if is_main and should_eval:
            model_for_eval = model.module if isinstance(model, DDP) else model
            last_val_metrics = evaluate_validation(
                model_for_eval=model_for_eval,
                val_dl=val_dl,
                device=device,
                epoch=ep,
                save_dir=args.save_dir,
            )
            print(
                f"[VAL] epoch={ep:03d} dice={last_val_metrics['dice']:.4f} "
                f"hd95={last_val_metrics['hd95']:.4f} assd={last_val_metrics['assd']:.4f} "
                f"cases={last_val_metrics['num_cases']}"
            )
        if world_size > 1:
            dist.barrier()

        if is_main:
            ckpt_name = ""
            if ep % max(args.save_every, 1) == 0 or ep == args.epochs:
                model_to_save = model.module if isinstance(model, DDP) else model
                ckpt = {
                    "epoch": ep,
                    "model": model_to_save.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "model_config": {
                        "oc_depth": args.oc_depth,
                        "oc_min_size": args.oc_min_size,
                        "ocre_hidden_dim": args.ocre_hidden_dim,
                    },
                    "mask_ratio": args.mask_ratio,
                    "candidate_warmup_epochs": args.candidate_warmup_epochs,
                    "refine_ramp_epochs": args.refine_ramp_epochs,
                    "physical_stats": physical_stats,
                    "world_size": world_size,
                }
                ckpt_name = f"gleve_ep{ep:03d}.pt"
                torch.save(ckpt, os.path.join(args.save_dir, ckpt_name))
            postfix = dict(
                cases=len(ds),
                loss=f"{epoch_loss:.4f}",
                valid_steps=f"{epoch_valid_steps:.1f}",
                skipped_empty=f"{epoch_skipped_empty:.1f}",
                skipped_nonfinite=f"{epoch_skipped_nonfinite:.1f}",
                skipped_timeout=f"{epoch_skipped_timeout:.1f}",
                skipped_oom=f"{epoch_skipped_oom:.1f}",
                refresh=False,
            )
            if ckpt_name:
                postfix["last_ckpt"] = ckpt_name
            if last_val_metrics is not None and int(last_val_metrics.get("epoch", -1)) == ep:
                postfix["val_dice"] = f"{last_val_metrics['dice']:.4f}"
                postfix["val_hd95"] = f"{last_val_metrics['hd95']:.4f}"
                postfix["val_assd"] = f"{last_val_metrics['assd']:.4f}"
            epoch_bar.set_postfix(**postfix)

    cleanup_distributed()


if __name__ == "__main__":
    main()
